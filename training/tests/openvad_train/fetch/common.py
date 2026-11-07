# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from email.message import Message
from http.client import HTTPMessage
from typing import Any

import pytest

from fetch_support import FakeResp, archive_with, http_error, plain_tar, tar_with
from openvad_train.fetch import common


@pytest.mark.parametrize(("env", "expected"), [(None, "data"), ("rel", "rel")])
def test_data_dir_is_absolute(monkeypatch, tmp_path, env, expected):
    monkeypatch.chdir(tmp_path)
    if env is None:
        monkeypatch.delenv("OPENVAD_DATA_DIR", raising=False)
    else:
        monkeypatch.setenv("OPENVAD_DATA_DIR", env)
    assert common.data_dir().is_absolute()
    assert common.data_dir().resolve() == (tmp_path / expected).resolve()


def test_download_checksum_mismatch_raises(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"openvad")
    with pytest.raises(ValueError, match="0" * 32):
        common.download(src.as_uri(), tmp_path / "out.bin", md5="0" * 32)
    assert not (tmp_path / "out.bin").exists()
    assert not (tmp_path / "out.bin.part").exists()


def test_download_truncated_close_keeps_part_for_resume(monkeypatch, tmp_path):
    seen = {}

    def fake_urlopen(req, **kw):
        seen.update(kw)
        return FakeResp([b"abc"], length=7)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ConnectionError, match="7 bytes missing"):
        common.download("https://x/f", tmp_path / "f.bin")
    assert (tmp_path / "f.bin.part").read_bytes() == b"abc"
    assert not (tmp_path / "f.bin").exists()
    assert seen["timeout"] == common.TIMEOUT_S


def test_hf_tree_refuses_a_paginated_listing(monkeypatch):
    headers = Message()
    headers["Link"] = '<https://huggingface.co/api/next>; rel="next"'
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *_, **__: FakeResp([b"[]"], headers=headers))
    with pytest.raises(RuntimeError, match="paginated"):
        common.hf_tree("o/r", "rev")


def test_download_416_on_complete_part_finishes(monkeypatch, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(b"openvad")

    def fake_urlopen(req, **_):
        raise http_error(416)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    out = common.download("https://x/f", tmp_path / "f.bin", md5=hashlib.md5(b"openvad").hexdigest())
    assert out.read_bytes() == b"openvad"
    assert not (tmp_path / "f.bin.part").exists()


def test_download_416_on_stale_part_restarts(monkeypatch, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(b"stale-and-too-long")
    calls = []

    def fake_urlopen(req, **_):
        calls.append(req.get_header("Range"))
        if len(calls) == 1:
            raise http_error(416)
        return FakeResp([b"openvad"])

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    out = common.download("https://x/f", tmp_path / "f.bin", md5=hashlib.md5(b"openvad").hexdigest())
    assert out.read_bytes() == b"openvad"
    assert calls == ["bytes=18-", None]


@pytest.mark.parametrize(
    ("content_range", "requests"),
    [("bytes */18", ["bytes=18-"]), ("bytes */7", ["bytes=18-", None]), (None, ["bytes=18-", None])],
)
def test_download_416_without_a_hash_trusts_only_a_matching_size(monkeypatch, tmp_path, content_range, requests):
    part = tmp_path / "f.bin.part"
    part.write_bytes(b"stale-and-too-long")
    calls = []

    def fake_urlopen(req, **_):
        calls.append(req.get_header("Range"))
        if len(calls) == 1:
            raise http_error(416, content_range)
        return FakeResp([b"openvad"])

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    out = common.download("https://x/f", tmp_path / "f.bin")
    assert out.read_bytes() == (b"stale-and-too-long" if content_range == "bytes */18" else b"openvad")
    assert calls == requests
    assert not part.exists()


def test_download_sends_headers_on_resume(monkeypatch, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(b"ope")
    seen = []

    def fake_urlopen(req, **_):
        seen.append((req.get_header("Authorization"), req.get_header("Range")))
        return FakeResp([b"nvad"], status=206)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    out = common.download("https://x/f", tmp_path / "f.bin", headers={"Authorization": "Bearer t"})
    assert out.read_bytes() == b"openvad"
    assert seen == [("Bearer t", "bytes=3-")]


def _length(size):
    headers = Message()
    headers["Content-Length"] = str(size)
    return headers


def _range_server(monkeypatch, data, fail_starts=(), short_starts=(), seen=None):
    """Fake urlopen serving `data` by Range only (HEAD is refused); it requires a timeout on every request.

    Pieces starting in `fail_starts` always fail, those in `short_starts` end after one byte. Returns the piece starts
    requested; the size probe is not among them. `seen` collects every request's Authorization.
    """
    requested = []

    def fake_urlopen(req, *, timeout):
        assert timeout
        assert req.get_method() == "GET"
        if seen is not None:
            seen.append(req.get_header("Authorization"))
        if req.get_header("Range") == "bytes=0-0":
            headers = Message()
            headers["Content-Range"] = f"bytes 0-0/{len(data)}"
            return FakeResp([data[:1]], status=206, headers=headers)
        start, end = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
        requested.append(start)
        if start in fail_starts:
            raise ConnectionResetError("reset")
        return FakeResp([data[start : start + 1 if start in short_starts else end + 1]], status=206)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(common, "PIECE", 4)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    return requested


def test_download_ranges_resumes_only_missing_pieces(monkeypatch, tmp_path):
    data = b"0123456789abcdefghij"
    _range_server(monkeypatch, data, fail_starts={8})
    dest = tmp_path / "a.tar"
    with pytest.raises(ConnectionError, match="1 pieces failed"):
        common.download_ranges("https://x/a.tar", dest, jobs=2)
    assert not dest.exists()
    assert sorted(map(int, (tmp_path / "a.tar.pieces").read_text().split())) == [0, 4, 12, 16]
    requested = _range_server(monkeypatch, data)
    assert common.download_ranges("https://x/a.tar", dest, jobs=2).read_bytes() == data
    assert requested == [8]
    assert not (tmp_path / "a.tar.pieces").exists()


def test_fetch_piece_retries_a_transient_error_with_growing_backoff(monkeypatch, tmp_path):
    part = tmp_path / "a.part"
    part.write_bytes(bytes(4))
    answers = [http_error(500), ConnectionResetError("reset"), FakeResp([b"data"], status=206)]
    sleeps = []

    def fake_urlopen(req, **_):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    monkeypatch.setattr(common, "PIECE_TRIES", 3)
    common._fetch_piece("https://x/a", part, 0, 4)
    assert part.read_bytes() == b"data"
    assert 0 < sleeps[0] < sleeps[1]


@pytest.mark.parametrize("code", [401, 403])
def test_fetch_piece_raises_an_expired_link_at_once(monkeypatch, tmp_path, code):
    calls = []

    def fake_urlopen(req, **_):
        calls.append(1)
        raise http_error(code)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(common.time, "sleep", lambda _: pytest.fail("must not back off"))
    with pytest.raises(urllib.error.HTTPError) as err:
        common._fetch_piece("https://x/a", tmp_path / "a.part", 0, 4)
    assert err.value.code == code
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("new_url", "keeps"),
    [("https://hf.co/g", True), ("https://cdn.example/f", False), ("http://hf.co/g", False)],
)
def test_redirect_keeps_authorization_only_on_the_same_https_host(new_url, keeps):
    handler = common._DropAuthOnCrossHostRedirect()
    req = urllib.request.Request("https://hf.co/f", headers={"Authorization": "Bearer t", "Range": "bytes=0-"})
    new = handler.redirect_request(req, io.BytesIO(), 302, "Found", HTTPMessage(), new_url)
    assert new is not None
    assert new.has_header("Authorization") == keeps
    assert new.get_header("Range") == "bytes=0-"


def test_the_auth_dropping_redirect_handler_is_installed():
    # The handler is tested alone above; this checks that urlopen uses it instead of urllib's default.
    opener = urllib.request._opener  # pyright: ignore[reportAttributeAccessIssue]
    assert opener is not None
    assert any(isinstance(h, common._DropAuthOnCrossHostRedirect) for h in opener.handlers)


def test_download_ranges_rejects_ignored_range(monkeypatch, tmp_path):
    def fake_urlopen(req, **_):
        return FakeResp([b"0123456789"], headers=_length(10))

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    with pytest.raises(ConnectionError, match="ignored the range"):
        common.download_ranges("https://x/a.tar", tmp_path / "a.tar")


def test_download_ranges_sends_headers_on_every_request(monkeypatch, tmp_path):
    data = b"0123456789abcdefghij"
    seen = []
    requested = _range_server(monkeypatch, data, seen=seen)
    out = common.download_ranges(
        "https://x/a.tar",
        tmp_path / "a.tar",
        sha256=hashlib.sha256(data).hexdigest(),
        headers={"Authorization": "Bearer t"},
    )
    assert out.read_bytes() == data
    assert sorted(requested) == [0, 4, 8, 12, 16]
    assert seen == ["Bearer t"] * 6


@pytest.mark.parametrize("algo", ["md5", "sha256"])
def test_download_ranges_hash_mismatch_restarts_cleanly(monkeypatch, tmp_path, algo):
    _range_server(monkeypatch, b"0123456789")
    expected: dict[str, Any] = {algo: "0" * 32}
    with pytest.raises(ValueError, match=f"{algo} mismatch"):
        common.download_ranges("https://x/a.zip", tmp_path / "a.zip", **expected)
    assert not list(tmp_path.iterdir())


def test_short_piece_is_not_logged_and_fails(monkeypatch, tmp_path):
    # Without the length check the hole stays zero-filled and, with no hash, is renamed to dest.
    _range_server(monkeypatch, b"0123456789ab", short_starts={4})
    with pytest.raises(ConnectionError, match="1 pieces failed"):
        common.download_ranges("https://x/a.tar", tmp_path / "a.tar", jobs=1)
    assert not (tmp_path / "a.tar").exists()
    assert sorted(map(int, (tmp_path / "a.tar.pieces").read_text().split())) == [0, 8]


def _record_synced_offsets(monkeypatch):
    # The file position of the synced descriptor shows the bytes were written and flushed before the sync.
    synced = []
    monkeypatch.setattr(common.os, "fsync", lambda fd: synced.append(os.lseek(fd, 0, os.SEEK_CUR)))
    return synced


def test_download_syncs_the_part_before_the_rename(monkeypatch, tmp_path):
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *_, **__: FakeResp([b"openvad"]))
    synced = _record_synced_offsets(monkeypatch)
    common.download("https://x/f", tmp_path / "f.bin")
    assert synced == [7]
    assert (tmp_path / "f.bin").read_bytes() == b"openvad"


def test_pieces_are_synced_before_they_are_logged(monkeypatch, tmp_path):
    _range_server(monkeypatch, b"0123456789ab")
    synced = _record_synced_offsets(monkeypatch)
    common.download_ranges("https://x/a.tar", tmp_path / "a.tar", jobs=1)
    assert synced == [4, 8, 12]


@pytest.mark.parametrize(
    ("fetch", "answer"),
    [
        pytest.param(common.download, {"length": 7}, id="download-truncated"),
        pytest.param(common.download_ranges, {"status": 206, "headers": _length(8)}, id="ranges-short-piece"),
        pytest.param(common.download_ranges, {"headers": _length(8)}, id="ranges-ignored"),
    ],
)
def test_error_messages_hide_the_query_string(monkeypatch, tmp_path, fetch, answer):
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *_, **__: FakeResp([b"0"], **answer))
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    with pytest.raises(ConnectionError) as err:
        fetch("https://x/a?sig=SECRET", tmp_path / "a.tar")
    assert "SECRET" not in str(err.value)
    assert "https://x/a" in str(err.value)


def test_download_ranges_keeps_a_bad_dest_and_says_how_to_recover(tmp_path):
    (tmp_path / "a.tar").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match=r"sha256 mismatch.*delete it to refetch"):
        common.download_ranges("https://x/a.tar", tmp_path / "a.tar", sha256="0" * 64)
    assert (tmp_path / "a.tar").read_bytes() == b"corrupt"


def test_stale_pieces_log_is_dropped_when_the_size_changes(monkeypatch, tmp_path):
    # A log from an older, different-sized file must not mark pieces of the new one done.
    data = b"0123456789ab"
    (tmp_path / "a.tar.part").write_bytes(b"\0" * 20)
    (tmp_path / "a.tar.pieces").write_text("0\n4\n")
    _range_server(monkeypatch, data)
    assert common.download_ranges("https://x/a.tar", tmp_path / "a.tar", jobs=1).read_bytes() == data


def test_resume_that_server_ignores_restarts_the_part(monkeypatch, tmp_path):
    # A 200 answer to a Range request must overwrite the part, not append to it.
    (tmp_path / "f.bin.part").write_bytes(b"ope")
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda req, **_: FakeResp([b"openvad"], status=200))
    assert common.download("https://x/f", tmp_path / "f.bin").read_bytes() == b"openvad"


def test_download_replaces_a_dest_that_fails_its_hash(monkeypatch, tmp_path):
    (tmp_path / "f.bin").write_bytes(b"corrupt")
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda req, **_: FakeResp([b"openvad"]))
    out = common.download("https://x/f", tmp_path / "f.bin", md5=hashlib.md5(b"openvad").hexdigest())
    assert out.read_bytes() == b"openvad"


def test_download_returns_a_valid_dest_without_a_request(monkeypatch, tmp_path):
    (tmp_path / "f.bin").write_bytes(b"openvad")
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *_, **__: pytest.fail("must not download"))
    out = common.download("https://x/f", tmp_path / "f.bin", sha256=hashlib.sha256(b"openvad").hexdigest())
    assert out.read_bytes() == b"openvad"


def test_http_error_on_resume_keeps_the_part_and_raises(monkeypatch, tmp_path):
    # Only a 416 can mean the part is complete; any other error must not rename it to dest.
    (tmp_path / "f.bin.part").write_bytes(b"ope")

    def fake_urlopen(req, **_):
        raise http_error(500)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        common.download("https://x/f", tmp_path / "f.bin")
    assert (tmp_path / "f.bin.part").read_bytes() == b"ope"
    assert not (tmp_path / "f.bin").exists()


def test_hf_token_from_the_token_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert common.hf_token() is None
    (tmp_path / "token").write_text("hf_abc\n")
    assert common.hf_token() == "hf_abc"


def test_tar_filter_extracts_without_setting_mtime(tmp_path):
    tar_with(tmp_path / "a.tar", "d/f.txt", b"hello", mtime=1_000_000_000)
    out = tmp_path / "out"
    with tarfile.open(tmp_path / "a.tar") as tar:
        tar.extractall(out, filter=common.tar_filter)
    f = out / "d/f.txt"
    assert f.read_bytes() == b"hello"
    assert abs(f.stat().st_mtime - time.time()) < 60


def test_tar_filter_rejects_unsafe_members(tmp_path):
    tar_with(tmp_path / "a.tar", "../escape.txt")
    with tarfile.open(tmp_path / "a.tar") as tar, pytest.raises(tarfile.FilterError):
        tar.extractall(tmp_path / "out", filter=common.tar_filter)


def test_manifest_round_trip(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"12345")
    common.write_manifest(tmp_path, {"dataset": "x", "licence": "CC BY 4.0"})
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert (m["files"], m["bytes"], m["dataset"]) == (1, 5, "x")
    assert common.is_complete(tmp_path)
    (tmp_path / "b.txt").write_bytes(b"1")
    assert not common.is_complete(tmp_path)


def test_same_count_other_size_is_incomplete(tmp_path):
    # A file truncated in place keeps the count and changes only the bytes.
    (tmp_path / "a.txt").write_bytes(b"12345")
    common.write_manifest(tmp_path, {"dataset": "x"})
    (tmp_path / "a.txt").write_bytes(b"12")
    assert not common.is_complete(tmp_path)


def test_corrupt_manifest_reads_as_empty(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    assert common.read_manifest(tmp_path) == {}
    assert not common.is_complete(tmp_path)


def test_stray_files_do_not_break_completeness(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"12345")
    common.write_manifest(tmp_path, {"dataset": "x"})
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "sub/__pycache__").mkdir(parents=True)
    (tmp_path / "sub/__pycache__/m.pyc").write_bytes(b"junk")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden/f").write_bytes(b"junk")
    assert common.is_complete(tmp_path)


def test_check_archive_end_rejects_a_bad_gzip_crc(tmp_path):
    archive = tmp_path / "a.tar.gz"
    archive_with(archive, {f"f{i}": os.urandom(5000) for i in range(3)})
    raw = bytearray(archive.read_bytes())
    raw[-8] ^= 0xFF  # the CRC32 in the gzip trailer
    archive.write_bytes(raw)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "out", filter=common.tar_filter)
        with pytest.raises(RuntimeError, match="delete it to refetch"):
            common.check_archive_end(tar, archive)


def test_check_archive_end_rejects_a_truncated_plain_tar(tmp_path):
    archive = tmp_path / "a.tar"
    plain_tar(archive, hole=True)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "out", filter=common.tar_filter)
        with pytest.raises(RuntimeError, match="ends at byte"):
            common.check_archive_end(tar, archive)


@pytest.mark.parametrize("gz", [False, True])
def test_check_archive_end_accepts_a_good_archive(tmp_path, gz):
    archive = tmp_path / ("a.tar.gz" if gz else "a.tar")
    if gz:
        archive_with(archive, {f"f{i}": os.urandom(5000) for i in range(3)})
    else:
        plain_tar(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "out", filter=common.tar_filter)
        common.check_archive_end(tar, archive)


def test_download_ranges_keyboard_interrupt_cancels_and_reraises(monkeypatch, tmp_path):
    _range_server(monkeypatch, b"0123456789abcdefghij")  # five pieces of four bytes
    started, release, ran = threading.Event(), threading.Event(), []

    def blocked_piece(url, part, start, end, headers=None):
        ran.append(start)
        started.set()
        assert release.wait(5)  # opens once the four queued pieces are cancelled

    def interrupted(futures):
        def on_done(_):
            if sum(f.cancelled() for f in futures) == len(futures) - 1:
                release.set()

        for future in futures:
            future.add_done_callback(on_done)
        assert started.wait(5)
        raise KeyboardInterrupt

    monkeypatch.setattr(common, "_fetch_piece", blocked_piece)
    monkeypatch.setattr(common.concurrent.futures, "as_completed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        common.download_ranges("https://x/a.tar", tmp_path / "a.tar", jobs=1)
    assert ran == [0]  # one piece was running; the queued ones were cancelled, none started
    assert not (tmp_path / "a.tar").exists()


def test_take_share_lock_refuses_a_lock_held_by_a_live_process(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    common.cache_dir().mkdir()
    path = common.take_share_lock("x")
    assert path.read_text() == f"host={socket.gethostname()} pid={os.getpid()}"
    with pytest.raises(SystemExit, match=f"x: held by host {socket.gethostname()}, pid {os.getpid()}"):
        common.take_share_lock("x")


def test_take_share_lock_takes_over_from_a_dead_process_on_this_host(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    common.cache_dir().mkdir()
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    (common.cache_dir() / "x.fetching").write_text(f"host={socket.gethostname()} pid={dead.stdout.strip()}")
    assert common.take_share_lock("x").read_text().endswith(f"pid={os.getpid()}")


@pytest.mark.parametrize("held", ["host=elsewhere pid=1", "not a lock file"])
def test_take_share_lock_never_deletes_a_lock_it_cannot_judge(monkeypatch, tmp_path, held):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    common.cache_dir().mkdir()
    path = common.cache_dir() / "x.fetching"
    path.write_text(held)
    with pytest.raises(SystemExit, match="held by host"):
        common.take_share_lock("x")
    assert path.read_text() == held
