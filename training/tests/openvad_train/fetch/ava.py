# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import hashlib
import subprocess
from pathlib import Path

import pytest

from fetch_support import FakeResp, clean_run
from openvad_train.fetch import ava, common
from openvad_train.fetch.common import file_hash


LABELS = "b,900,905.6,NO_SPEECH\na,900,901,CLEAN_SPEECH\na,901,910,NO_SPEECH\nc,900,950,SPEECH_WITH_NOISE\n"


def test_mirror_map_reports_missing():
    labels = "b,900,905.6,NO_SPEECH\na,900,901,CLEAN_SPEECH\na,901,910,NO_SPEECH\nc,900,950,X\nc,950,960,X\n"
    found, missing = ava.ava_mirror_map(ava.ava_video_ids(labels), "a.mkv\nb.mp4\nzz.webm\n")
    assert found == {"a": "a.mkv", "b": "b.mp4"}
    assert missing == ["c"]


def test_ffmpeg_cmd_cuts_labelled_span_without_resampling(tmp_path):
    cmd = ava.ava_ffmpeg_cmd("https://x/a.mkv", tmp_path / "a.flac")
    assert cmd[cmd.index("-ss") + 1] == "900"
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-t") + 1] == "900"
    assert cmd.index("-t") > cmd.index("-i")
    assert not {"-to", "-ar", "-ac"} & set(cmd)
    assert cmd[cmd.index("-rw_timeout") + 1] == ava.RW_TIMEOUT_US
    assert cmd.index("-rw_timeout") < cmd.index("-i")
    assert cmd[cmd.index("-vn") : cmd.index("-vn") + 3] == ["-vn", "-c:a", "flac"]


def test_fetch_ava_reports_cause_per_failed_video(monkeypatch, tmp_path):
    _, out = _ava_stub(monkeypatch, tmp_path, {})

    def fake_run(cmd, **_):
        if cmd[0] == "ffprobe" and cmd[-1].endswith("a.mkv"):
            raise subprocess.CalledProcessError(1, cmd, stderr="HTTP error 403 Forbidden\n")
        if "-i" in cmd:
            Path(cmd[-1]).write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, cmd, stderr="Connection reset")
        return subprocess.CompletedProcess(cmd, 0, stdout="0.0\n")

    monkeypatch.setattr(ava.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="ffmpeg/ffprobe") as err:
        ava.fetch_ava()
    assert "a: ffprobe exited 1: HTTP error 403 Forbidden" in str(err.value)
    assert "b: ffmpeg exited 1: Connection reset" in str(err.value)
    assert not list((out / "audio").iterdir())


def _ava_stub(monkeypatch, tmp_path, durations):
    """Fake download and ffmpeg/ffprobe; `durations` maps video id to the cut's length, default full."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ava, "require_ffmpeg", lambda: None)
    cmds: list[list[str]] = []

    def fake_download(url, dest, **_):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("a,900,901,X\nb,900,901,X\nc,900,901,X\n" if url == ava.AVA_LABELS else "a.mkv\nb.mkv\n")
        return dest

    def fake_run(cmd, **_):
        cmds.append(cmd)
        if cmd[:2] == ["ffmpeg", "-version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version 7.1 Copyright\nbuilt with clang\n")
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(Path(cmd[cmd.index("-i") + 1]).name.encode())
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        if "format=duration" in cmd:
            stem = Path(cmd[-1]).name.split(".")[0]
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{durations.get(stem, 900.0)}\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="0.5\n" if cmd[-1].endswith("a.mkv") else "N/A\n")

    monkeypatch.setattr(ava, "download", fake_download)
    monkeypatch.setattr(ava.subprocess, "run", fake_run)
    return cmds, tmp_path / "datasets/raw/ava-speech"


def test_fetch_ava_records_provenance_and_uses_timeouts(monkeypatch, tmp_path):
    cmds, out = _ava_stub(monkeypatch, tmp_path, {})
    ava.fetch_ava()
    m = clean_run("ava-speech")
    assert m["ffmpeg"] == "ffmpeg version 7.1 Copyright"
    assert m["audio"] == {
        "a": {"sha256": file_hash(out / "audio/a.flac", "sha256"), "start_time": 0.5},
        "b": {"sha256": file_hash(out / "audio/b.flac", "sha256")},
    }
    assert (m["videos"], m["missing"]) == (2, 1)
    assert (out / "missing.txt").read_text() == "c\n"
    probes = [c for c in cmds if "format=start_time" in c]
    assert len(probes) == 2
    assert all(c[c.index("-rw_timeout") + 1] == ava.RW_TIMEOUT_US and c[-1].startswith("https://") for c in probes)
    assert all(c.index("-rw_timeout") < len(c) - 2 for c in probes)


def test_fetch_ava_short_cut_is_failed_and_removed(monkeypatch, tmp_path):
    _, out = _ava_stub(monkeypatch, tmp_path, {"a": 898.0})
    with pytest.raises(RuntimeError, match="ffmpeg/ffprobe") as err:
        ava.fetch_ava()
    assert "a: short output: 898.0 s" in str(err.value)
    assert not list((out / "audio").glob("a*"))
    assert (out / "audio/b.flac").exists()
    assert not (out / "manifest.json").exists()


def test_fetch_ava_short_finished_file_is_reported_not_deleted(monkeypatch, tmp_path):
    _, out = _ava_stub(monkeypatch, tmp_path, {"a": 5.0})
    (out / "audio").mkdir(parents=True)
    (out / "audio/a.flac").write_bytes(b"short")
    with pytest.raises(RuntimeError, match=r"a: short output: 5\.0 s"):
        ava.fetch_ava()
    assert (out / "audio/a.flac").read_bytes() == b"short"


@pytest.mark.parametrize("changed", ["labels", "file list"])
def test_fetch_ava_rejects_a_changed_list(monkeypatch, tmp_path, changed):
    served = {ava.AVA_LABELS: b"a,900,901,X\n", ava.AVA_FILE_LIST: b"a.mkv\n"}
    pins = {url: hashlib.sha256(body).hexdigest() for url, body in served.items()}
    pins[ava.AVA_LABELS if changed == "labels" else ava.AVA_FILE_LIST] = "0" * 64
    monkeypatch.setattr(ava, "AVA_LABELS_SHA256", pins[ava.AVA_LABELS])
    monkeypatch.setattr(ava, "AVA_FILE_LIST_SHA256", pins[ava.AVA_FILE_LIST])
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ava, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda req, **_: FakeResp([served[req.full_url]]))

    def no_subprocess(cmd, **_):
        pytest.fail(f"subprocess.run called: {cmd}")

    monkeypatch.setattr(ava.subprocess, "run", no_subprocess)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ava.fetch_ava()
    assert not (tmp_path / "datasets/raw/ava-speech/manifest.json").exists()
