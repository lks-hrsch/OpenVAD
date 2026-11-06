import fcntl
import os
import socket
import subprocess
from pathlib import Path

import pytest

from openvad_train.fetch import __main__ as cli
from openvad_train.fetch import common


@pytest.fixture(autouse=True)
def _isolated_locks(monkeypatch, tmp_path):
    # The per-user lock lives under HOME; keep every run off the real home, temp dir and data dir.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))


def _fetch_lock(name: str) -> Path:
    path = Path.home() / ".cache/openvad" / f"fetch-{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _share_lock(name: str) -> Path:
    return common.cache_dir() / f"{name}.fetching"


def _hold_share_lock(name: str, content: str) -> Path:
    path = _share_lock(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_main_refuses_to_refetch_a_mismatching_manifest(monkeypatch, tmp_path):
    root = common.raw_dir("librispeech")
    root.mkdir(parents=True)
    (root / "a.txt").write_bytes(b"12345")
    common.write_manifest(root, {"dataset": "x"})
    (root / "b.txt").write_bytes(b"12")
    monkeypatch.setitem(cli.FETCHERS, "librispeech", lambda: pytest.fail("must not refetch"))
    with pytest.raises(
        RuntimeError, match=r"expected 1 files, 5 bytes; found 2 files, 7 bytes.*delete .*manifest.json"
    ):
        cli.main(["librispeech"])


def test_main_resumes_without_manifest(monkeypatch):
    partial = common.raw_dir("librispeech") / "partial.flac"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x")
    ran = []
    monkeypatch.setitem(cli.FETCHERS, "librispeech", lambda: ran.append(partial.exists()))
    cli.main(["librispeech"])
    assert ran == [True]


def test_main_refuses_a_dataset_that_is_already_being_fetched(monkeypatch, tmp_path):
    monkeypatch.setitem(cli.FETCHERS, "librispeech", lambda: pytest.fail("must not run"))
    with _fetch_lock("librispeech").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="librispeech: another fetch"):
            cli.main(["librispeech"])


@pytest.mark.parametrize(
    "holder",
    [
        "host=other pid={dead}",  # a pid from another host says nothing here, even if it is dead here
        "garbage",
        "host={this} pid={me}",
        "host={this} pid=1",  # alive under another user
    ],
)
def test_main_refuses_a_held_share_lock(monkeypatch, holder):
    proc = subprocess.Popen(["true"])
    proc.wait()
    content = holder.format(this=socket.gethostname(), me=os.getpid(), dead=proc.pid)
    monkeypatch.setitem(cli.FETCHERS, "librispeech", lambda: pytest.fail("must not run"))
    path = _hold_share_lock("librispeech", content)
    with pytest.raises(SystemExit) as exc:
        cli.main(["librispeech"])
    assert str(path) in str(exc.value)
    assert path.read_text() == content


def test_main_takes_over_a_dead_lock_from_this_host(monkeypatch):
    proc = subprocess.Popen(["true"])
    proc.wait()
    path = _hold_share_lock("librispeech", f"host={socket.gethostname()} pid={proc.pid}")
    ran = []
    monkeypatch.setitem(cli.FETCHERS, "librispeech", lambda: ran.append(path.read_text()))
    cli.main(["librispeech"])
    assert ran == [f"host={socket.gethostname()} pid={os.getpid()}"]
    assert not path.exists()


def test_main_removes_the_share_lock_when_the_fetcher_raises(monkeypatch):
    def boom():
        assert _share_lock("librispeech").exists()
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.FETCHERS, "librispeech", boom)
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["librispeech"])
    assert not _share_lock("librispeech").exists()


@pytest.mark.parametrize(("name", "raw"), [("ava", "ava-speech"), ("alignments", "librispeech-alignments")])
def test_main_removes_leftover_markers_from_a_complete_tree(monkeypatch, name, raw):
    root = common.raw_dir(raw)
    root.mkdir(parents=True)
    (root / "a.txt").write_bytes(b"12345")
    common.write_manifest(root, {"dataset": "x"})
    markers = [root / ".medium-extracted", root / ".setup-done", root / ".generate-done"]
    for marker in markers:
        marker.touch()
    monkeypatch.setitem(cli.FETCHERS, name, lambda: pytest.fail("must not refetch"))
    cli.main([name])
    assert not any(marker.exists() for marker in markers)
    assert (root / "a.txt").exists()
