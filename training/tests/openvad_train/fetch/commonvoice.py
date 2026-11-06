# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import datetime as dt
import hashlib
import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from fetch_support import FakeResp, archive_with, clean_run, http_error, orphan_archive
from openvad_train.fetch import common, commonvoice


def test_commonvoice_daily_limit_waits_for_the_reset_and_asks_again(monkeypatch):
    now = dt.datetime(2026, 12, 31, 22, 0, tzinfo=dt.UTC)  # the wait crosses a month and a year

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    answers = [http_error(429), http_error(429), io.BytesIO(b'{"downloadUrl": "u"}')]

    def fake_urlopen(req, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    sleeps = []
    monkeypatch.setattr(commonvoice.dt, "datetime", Clock)
    monkeypatch.setattr(commonvoice.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(commonvoice.time, "sleep", sleeps.append)
    assert commonvoice.download_session("abc", "key") == {"downloadUrl": "u"}
    assert sleeps == [2 * 3600 + 600] * 2


def test_commonvoice_reuses_a_valid_saved_session(monkeypatch, tmp_path):
    session_file = tmp_path / "s.json"
    expires = (dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    session_file.write_text(json.dumps({"downloadUrl": "old", "expiresAt": expires}))
    posts, downloads = [], []
    monkeypatch.setattr(commonvoice, "download_session", lambda *a: posts.append(a))
    monkeypatch.setattr(commonvoice, "download_ranges", lambda url, dest, **kw: downloads.append((url, kw)))
    commonvoice.download_archive("abc", "key", "h" * 64, tmp_path / "a.tar.gz", session_file)
    assert posts == []
    assert downloads == [("old", {"sha256": "h" * 64, "headers": {"User-Agent": commonvoice.USER_AGENT}})]


def test_commonvoice_asks_for_a_session_when_expired(monkeypatch, tmp_path):
    session_file = tmp_path / "s.json"
    session_file.write_text(json.dumps({"downloadUrl": "old", "expiresAt": "2000-01-01T00:00:00Z"}))
    new = {"downloadUrl": "new", "checksum": "h" * 64}
    calls = []

    def fake_ranges(url, dest, **kw):
        # Saved before the download starts: a kill mid-download must not cost another daily download.
        calls.append((url, kw["sha256"], json.loads(session_file.read_text())))

    monkeypatch.setattr(commonvoice, "download_session", lambda *a: new)
    monkeypatch.setattr(commonvoice, "download_ranges", fake_ranges)
    commonvoice.download_archive("abc", "key", "h" * 64, tmp_path / "a.tar.gz", session_file)
    assert calls == [("new", "h" * 64, new)]


def test_commonvoice_session_without_expiry_is_reused_until_a_403(monkeypatch, tmp_path):
    session_file = tmp_path / "s.json"
    session_file.write_text(json.dumps({"downloadUrl": "https://x/old"}))
    data = b"0123456789"
    urls = []

    def fake_urlopen(req, **_):
        urls.append(req.full_url)
        if req.get_header("Range") == "bytes=0-0":
            headers = Message()
            headers["Content-Range"] = f"bytes 0-0/{len(data)}"
            return FakeResp([data[:1]], status=206, headers=headers)
        if req.full_url == "https://x/old":  # the link expires mid-download
            raise http_error(403)
        start, end = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
        return FakeResp([data[start : end + 1]], status=206)

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(commonvoice, "download_session", lambda *a: {"downloadUrl": "https://x/new"})
    monkeypatch.setattr(common.time, "sleep", lambda _: pytest.fail("must not back off"))
    archive = tmp_path / "a.tar.gz"
    commonvoice.download_archive("abc", "key", hashlib.sha256(data).hexdigest(), archive, session_file)
    assert archive.read_bytes() == data
    assert "https://x/old" in urls
    assert json.loads(session_file.read_text()) == {"downloadUrl": "https://x/new"}


@pytest.mark.parametrize("content", ['{"downloadUr', "{}"])
def test_commonvoice_unusable_saved_session_is_ignored(tmp_path, content):
    session_file = tmp_path / "s.json"
    session_file.write_text(content)
    assert commonvoice.saved_session(session_file) is None


@pytest.mark.parametrize("saved", [True, False])
def test_commonvoice_other_download_errors_keep_the_link(monkeypatch, tmp_path, saved):
    session_file = tmp_path / "s.json"
    if saved:
        session_file.write_text(json.dumps({"downloadUrl": "old"}))
    asked = []

    def fake_ranges(url, dest, **kw):
        raise http_error(500)

    monkeypatch.setattr(commonvoice, "download_session", lambda *a: asked.append(1) or {"downloadUrl": "new"})
    monkeypatch.setattr(commonvoice, "download_ranges", fake_ranges)
    with pytest.raises(urllib.error.HTTPError):
        commonvoice.download_archive("abc", "key", "h" * 64, tmp_path / "a.tar.gz", session_file)
    assert len(asked) == (0 if saved else 1)
    assert session_file.exists()


def test_commonvoice_session_with_another_checksum_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(commonvoice, "download_session", lambda *a: {"downloadUrl": "new", "checksum": "x" * 64})
    with pytest.raises(RuntimeError, match="MDC now serves"):
        commonvoice.download_archive("abc", "key", "h" * 64, tmp_path / "a.tar.gz", tmp_path / "s.json")


def _tsv(rows):
    return "\n".join("\t".join(r) for r in [["client_id", "path", "sentence"], *rows]).encode() + b"\n"


def test_commonvoice_extracts_only_validated_clips(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    # The unbalanced quote: csv's default quoting would swallow the rows after it and lose their clips silently.
    validated = _tsv([["s1", "a.mp3", '"hi'], ["s2", "c.mp3", "x"]])
    # Clips before the metadata, as the archive order is not known.
    members = {"cv/xx/clips/a.mp3": b"a", "cv/xx/clips/b.mp3": b"b", "cv/xx/clips/c.mp3": b"c"}
    metadata = {"cv/xx/validated.tsv": validated, "cv/xx/invalidated.tsv": _tsv([["s3", "b.mp3", "y"]])}
    archive_with(archive, members | metadata)
    locale_dir, missing = commonvoice.extract_validated(archive, tmp_path / "out")
    assert (locale_dir, missing) == (tmp_path / "out/cv/xx", 0)
    assert sorted(p.name for p in (locale_dir / "clips").iterdir()) == ["a.mp3", "c.mp3"]
    assert (locale_dir / "invalidated.tsv").exists()


def _cv_archive(path, clips, validated):
    members = {f"cv/xx/clips/{c}": b"x" for c in clips} | {"cv/xx/validated.tsv": _tsv(validated)}
    archive_with(path, members)


def test_commonvoice_fetch_keeps_the_archive_when_nothing_was_extracted(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(commonvoice, "CV_LOCALES", {"xx": ("id", "0" * 64)})
    monkeypatch.setattr(commonvoice, "mdc_api_key", lambda: "key")

    def fake_download_archive(dataset_id, key, sha256, archive, session_file):
        archive.parent.mkdir(parents=True, exist_ok=True)
        _cv_archive(archive, ["a.mp3"], [["s1", "other.mp3", "x"]])

    monkeypatch.setattr(commonvoice, "download_archive", fake_download_archive)
    with pytest.raises(RuntimeError, match="none of the 1 validated clips"):
        commonvoice.fetch_commonvoice()
    cache = tmp_path / "cache"
    assert (cache / f"common-voice-{commonvoice.CV_RELEASE}-xx.tar.gz").exists()
    assert not (common.raw_dir("common-voice") / ".xx-extracted").exists()


def test_extract_validated_counts_missing_clips(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    _cv_archive(archive, ["a.mp3"], [["s1", "a.mp3", "x"], ["s2", "b.mp3", "y"], ["s3", "c.mp3", "z"]])
    assert commonvoice.extract_validated(archive, tmp_path / "out")[1] == 2


def test_extract_validated_skips_clips_already_on_disk(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    members = {"cv/xx/clips/a.mp3": b"aaa", "cv/xx/clips/b.mp3": b"bbb", "cv/xx/clips/c.mp3": b"ccc"}
    archive_with(archive, members | {"cv/xx/validated.tsv": _tsv([["s", f"{c}.mp3", "x"] for c in "abc"])})
    clips = tmp_path / "out/cv/xx/clips"
    clips.mkdir(parents=True)
    (clips / "a.mp3").write_bytes(b"AAA")  # full size: kept as is, even with other content
    (clips / "b.mp3").write_bytes(b"b")  # cut short by a kill: extracted again
    _, missing = commonvoice.extract_validated(archive, tmp_path / "out")
    assert missing == 0
    assert [(clips / f"{c}.mp3").read_bytes() for c in "abc"] == [b"AAA", b"bbb", b"ccc"]


def _fake_cv_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(commonvoice, "CV_LOCALES", {"xx": ("id", "0" * 64)})
    monkeypatch.setattr(commonvoice, "mdc_api_key", lambda: "key")
    members = {f"cv/xx/{s}.tsv": _tsv([["s1", "a.mp3", "x"]]) for s in commonvoice.CV_SPLITS}
    return members | {"cv/xx/clips/a.mp3": b"a"}


def test_fetch_commonvoice_one_locale_succeeds(monkeypatch, tmp_path):
    members = _fake_cv_fetch(monkeypatch, tmp_path)

    def fake_download_archive(dataset_id, key, sha256, archive, session_file):
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive_with(archive, members)
        session_file.write_text("{}")

    monkeypatch.setattr(commonvoice, "download_archive", fake_download_archive)
    commonvoice.fetch_commonvoice()
    manifest = clean_run("common-voice")
    assert [x["url"] for x in manifest["sources"]] == ["https://mozilladatacollective.com/datasets/id"]
    assert manifest["locales"]["xx"]["missing_clips"] == 0
    assert (common.raw_dir("common-voice") / "xx/cv/xx/clips/a.mp3").exists()


def test_cached_archive_with_a_bad_hash_is_downloaded_again(monkeypatch, tmp_path):
    members = _fake_cv_fetch(monkeypatch, tmp_path)
    archive = tmp_path / "cache" / f"common-voice-{commonvoice.CV_RELEASE}-xx.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"truncated")
    downloads = []

    def fake_download_archive(dataset_id, key, sha256, archive, session_file):
        assert not archive.exists()
        downloads.append(archive)
        archive_with(archive, members)

    monkeypatch.setattr(commonvoice, "download_archive", fake_download_archive)
    commonvoice.fetch_commonvoice()
    assert downloads == [archive]
    assert (common.raw_dir("common-voice") / "xx/cv/xx/clips/a.mp3").exists()


def test_second_fresh_session_403_propagates(monkeypatch, tmp_path):
    session_file = tmp_path / "s.json"
    asked = []

    def fake_ranges(url, dest, **kw):
        raise http_error(403)

    monkeypatch.setattr(commonvoice, "download_session", lambda *a: asked.append(1) or {"downloadUrl": "u"})
    monkeypatch.setattr(commonvoice, "download_ranges", fake_ranges)
    with pytest.raises(urllib.error.HTTPError):
        commonvoice.download_archive("abc", "key", "h" * 64, tmp_path / "a.tar.gz", session_file)
    assert len(asked) == 2
    assert not session_file.exists()


def test_session_file_is_private(monkeypatch, tmp_path):
    session_file = tmp_path / "s.json"
    # A world-readable file from older code, ignored because it has expired.
    session_file.write_text(json.dumps({"downloadUrl": "u", "expiresAt": "2000-01-01T00:00:00Z"}))
    session_file.chmod(0o644)
    monkeypatch.setattr(commonvoice, "download_session", lambda *a: {"downloadUrl": "u"})
    monkeypatch.setattr(commonvoice, "download_ranges", lambda *a, **kw: None)
    commonvoice.download_archive("abc", "key", "h" * 64, tmp_path / "a.tar.gz", session_file)
    assert session_file.stat().st_mode & 0o777 == 0o600


def test_commonvoice_split_stats_pool_excludes_dev_and_test_speakers(monkeypatch, tmp_path):
    monkeypatch.setattr(commonvoice, "CV_SPEAKER_CAP_S", 2 * 3600)
    train = [["s1", f"{i}.mp3", "x"] for i in range(3)]
    dev, test = [["s3", "7.mp3", "x"]], [["s4", "8.mp3", "x"]]
    validated = [*train, ["s2", "6.mp3", "x"], *dev, *test]
    for split, rows in {"train": train, "dev": dev, "test": test, "validated": validated}.items():
        (tmp_path / f"{split}.tsv").write_bytes(_tsv(rows))
    durations = "clip\tduration[ms]\n" + "".join(f"{r[1]}\t3600000\n" for r in validated)  # one hour per clip
    (tmp_path / "clip_durations.tsv").write_text(durations)
    stats = commonvoice.split_stats(tmp_path)
    assert stats["train"] == {"clips": 3, "speakers": 1}
    # s3 and s4 are held out, so s1 (3 h) and s2 (1 h) form the pool; the 2 h cap leaves 2 h + 1 h.
    assert stats["train_pool"] == {"speakers": 2, "hours": 4.0, "top_speaker_hours": [3.0, 1.0], "hours_under_cap": 3.0}


def _seed_cv_locale(locale_dir):
    for split in commonvoice.CV_SPLITS:
        (locale_dir / f"{split}.tsv").parent.mkdir(parents=True, exist_ok=True)
        (locale_dir / f"{split}.tsv").write_bytes(_tsv([["s1", "a.mp3", "x"]]))


def test_cached_valid_archive_is_kept_and_not_downloaded_again(monkeypatch, tmp_path):
    members = _fake_cv_fetch(monkeypatch, tmp_path)
    archive = tmp_path / "cache" / f"common-voice-{commonvoice.CV_RELEASE}-xx.tar.gz"
    archive.parent.mkdir()
    archive_with(archive, members)
    monkeypatch.setattr(commonvoice, "CV_LOCALES", {"xx": ("id", common.file_hash(archive, "sha256"))})
    monkeypatch.setattr(commonvoice, "download_archive", lambda *a: pytest.fail("must not download"))
    commonvoice.fetch_commonvoice()
    out = common.raw_dir("common-voice")
    assert (out / "xx/cv/xx/clips/a.mp3").exists()
    clean_run("common-voice")
    assert not archive.exists()  # deleted after its clips are out, not before


@pytest.mark.parametrize(("marker_text", "expected"), [("3", 3), ("", None)])
def test_commonvoice_marker_skip_reports_the_missing_clips_it_holds(monkeypatch, tmp_path, marker_text, expected):
    _fake_cv_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(commonvoice, "download_archive", lambda *a: pytest.fail("must not download"))
    out = common.raw_dir("common-voice")
    _seed_cv_locale(out / "xx/cv/xx")
    (out / ".xx-extracted").write_text(marker_text)  # an empty marker comes from older code
    session_file = common.cache_dir() / f"common-voice-{commonvoice.CV_RELEASE}-xx.session.json"
    orphan_archive(f"common-voice-{commonvoice.CV_RELEASE}-xx.tar.gz")
    session_file.write_text("{}")
    commonvoice.fetch_commonvoice()
    manifest = clean_run("common-voice")
    assert manifest["locales"]["xx"]["missing_clips"] == expected
