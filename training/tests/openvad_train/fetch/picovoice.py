# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import subprocess
import tempfile
from pathlib import Path

import pytest

from fetch_support import archive_with, clean_run
from openvad_train.fetch import common, picovoice


def test_picovoice_cmd_is_isolated_and_pinned(tmp_path):
    cmd = picovoice.picovoice_cmd(tmp_path / "tc", tmp_path / "demand", tmp_path / "s.wav", tmp_path / "l.txt")
    uv_flags = cmd[: cmd.index("python")]
    assert "--no-project" in uv_flags
    assert set(picovoice.PICOVOICE_DEPS) <= set(uv_flags)
    assert cmd[-4:] == [str(tmp_path / p) for p in ("tc", "demand", "s.wav", "l.txt")]


def _picovoice_inputs():
    """Finished LibriSpeech and DEMAND trees, as the Picovoice mixer needs them."""
    for name, rel in (("librispeech", "LibriSpeech/test-clean/f.flac"), ("demand", "DKITCHEN/ch01.wav")):
        root = common.raw_dir(name)
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_bytes(b"x")
        common.write_manifest(root, {"dataset": name})


def _picovoice_stub(monkeypatch, tmp_path):
    downloads, runs = [], []
    (tmp_path / "tmp").mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))  # the DEMAND view goes under tmp_path

    def fake_download(url, dest, **_):
        downloads.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_with(dest, {f"voice-activity-benchmark-{picovoice.PICOVOICE_REV}/mixer.py": b"#"})
        return dest

    def fake_run(cmd, *, cwd, check):
        test_clean, view, speech, labels = map(Path, cmd[-4:])
        assert view.parent == tmp_path / "tmp"
        runs.append(
            {"cwd": cwd, "check": check, "test_clean": test_clean, "scenes": sorted(p.name for p in view.iterdir())}
        )
        speech.write_bytes(b"s")
        labels.write_text("2\n")

    monkeypatch.setattr(picovoice, "download", fake_download)
    monkeypatch.setattr(picovoice.subprocess, "run", fake_run)
    return downloads, runs


def test_fetch_picovoice_mixes_librispeech_with_the_demand_scenes_only(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    _picovoice_inputs()
    downloads, runs = _picovoice_stub(monkeypatch, tmp_path)
    picovoice.fetch_picovoice()
    out = common.raw_dir("picovoice")
    assert downloads == [f"picovoice-{picovoice.PICOVOICE_REV}.tar.gz"]
    # The DEMAND manifest is hidden from the mixer, which lists every entry of the folder as a scene.
    assert runs == [
        {
            "cwd": out / "repo",
            "check": True,
            "test_clean": common.raw_dir("librispeech") / "LibriSpeech/test-clean",
            "scenes": ["DKITCHEN"],
        }
    ]
    assert (out / "repo/mixer.py").exists()
    manifest = clean_run("picovoice")
    assert manifest["revision"] == picovoice.PICOVOICE_REV
    assert (out / "benchmark_speech.wav").exists()
    assert (out / "benchmark_labels.txt").exists()
    (out / "manifest.json").unlink()
    picovoice.fetch_picovoice()  # the repo and the mixed audio are kept: a refetch only writes the manifest again
    assert (len(downloads), len(runs)) == (1, 1)
    assert common.is_complete(out)


def test_fetch_picovoice_mixes_again_when_the_labels_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    _picovoice_inputs()
    out = common.raw_dir("picovoice")
    (out / "repo").mkdir(parents=True)
    (out / "benchmark_speech.wav").write_bytes(b"half")  # a run killed between the two outputs
    _, runs = _picovoice_stub(monkeypatch, tmp_path)
    picovoice.fetch_picovoice()
    assert len(runs) == 1
    assert (out / "benchmark_speech.wav").read_bytes() == b"s"
    assert (out / "benchmark_labels.txt").exists()
    clean_run("picovoice")


def test_fetch_picovoice_failed_mix_leaves_no_output_and_reruns(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    _picovoice_inputs()
    _, runs = _picovoice_stub(monkeypatch, tmp_path)
    working_run = picovoice.subprocess.run

    def killed_run(cmd, *, cwd, check):
        speech, labels = map(Path, cmd[-2:])
        speech.write_bytes(b"half")
        labels.write_text("2\n")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(picovoice.subprocess, "run", killed_run)
    with pytest.raises(subprocess.CalledProcessError):
        picovoice.fetch_picovoice()
    out = common.raw_dir("picovoice")
    assert not any((out / name).exists() for name in ("benchmark_speech.wav", "benchmark_labels.txt", "manifest.json"))

    monkeypatch.setattr(picovoice.subprocess, "run", working_run)
    picovoice.fetch_picovoice()
    assert len(runs) == 1
    assert (out / "benchmark_speech.wav").read_bytes() == b"s"
    assert (out / "benchmark_labels.txt").exists()
    assert not list(out.glob("*.part.*"))
    clean_run("picovoice")


@pytest.mark.parametrize("unfinished", ["librispeech", "demand"])
def test_fetch_picovoice_needs_finished_librispeech_and_demand(monkeypatch, tmp_path, unfinished):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    _picovoice_inputs()
    (common.raw_dir(unfinished) / "manifest.json").unlink()
    _, runs = _picovoice_stub(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="fetch `librispeech` and `demand` first"):
        picovoice.fetch_picovoice()
    assert runs == []
    assert not common.raw_dir("picovoice").exists()
