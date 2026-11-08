# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import subprocess
import sys
import zipfile

import pytest

from fetch_support import assert_manifest_shape
from librivad_support import make_upstream, tree_digest
from openvad_train.dataset.sources import librivad as librivad_source
from openvad_train.fetch import common, librivad


@pytest.mark.parametrize("has_manifest", [False, True], ids=["no-manifest", "changed-tree"])
def test_librivad_small_needs_a_complete_medium(monkeypatch, tmp_path, has_manifest):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(librivad.subprocess, "run", lambda *_, **__: pytest.fail("must not generate"))
    medium = common.raw_dir("librivad")
    if has_manifest:
        medium.mkdir(parents=True)
        common.write_manifest(medium, {"dataset": "librivad", "labels_dir": "repo/Files/Labels"})
        (medium / "appeared-later.wav").touch()
    with pytest.raises(RuntimeError, match="just fetch librivad"):
        librivad.fetch_librivad_small()


def test_librivad_files_skips_eval_and_zips():
    tree = [
        {"type": "file", "path": "setup.py"},
        {"type": "file", "path": "Scripts/create_labels.py"},
        {"type": "file", "path": "Files/Noises.zip"},
        {"type": "file", "path": "Eval-ViT-MFCC/mfcc.py"},
        {"type": "directory", "path": "Scripts"},
    ]
    assert librivad.librivad_files(tree) == ["Scripts/create_labels.py", "setup.py"]


def test_librivad_uv_cmd_is_isolated():
    cmd = librivad.librivad_uv_cmd("setup.py", "EU")
    assert cmd[:3] == ["uv", "run", "--no-project"]
    assert cmd[-3:] == ["python", "setup.py", "EU"]


def test_librivad_counts_per_dataset_and_split(tmp_path):
    for rel in (
        "LibriSpeech/train-clean-100/n1/0/a.wav",
        "LibriSpeech/train-clean-100/n2/0/b.wav",
        "LibriSpeechConcat/test-clean/n1/0/c.wav",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    (tmp_path / "LibriSpeech/train-clean-100/n1/0/ignored.txt").touch()
    assert librivad.librivad_counts(tmp_path) == {"LibriSpeech/train-clean-100": 2, "LibriSpeechConcat/test-clean": 1}


def _fake_file(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr(f"{dest.stem}.txt", "x")
    else:
        dest.write_text("x")
    return dest


def _make_wavs(generated, skip=()):
    for d in ("LibriSpeech", "LibriSpeechConcat"):
        for split in ("train-clean-100", "test-clean"):
            if f"{d}/{split}" not in skip:
                wav = generated / d / split / "n/0/a.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                wav.touch()


def test_fetch_librivad_runs_setup_then_generate_and_records_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    tree = [
        {"type": "file", "path": "setup.py"},
        {"type": "file", "path": "Files/Noises.zip", "lfs": {"oid": "aa"}},
        {"type": "file", "path": "Files/Forced_alignments.zip", "lfs": {"oid": "bb"}},
    ]
    downloads: list[tuple[str, str | None]] = []

    def fake_download(url, dest, **kw):
        downloads.append((url, kw.get("sha256")))
        return _fake_file(dest)

    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw["cwd"], kw["check"]))
        repo = kw["cwd"]
        if "setup.py" in cmd:
            (repo / "Files/Labels").mkdir(parents=True)
        else:
            _make_wavs(tmp_path / "datasets/raw/librivad/generated")

    monkeypatch.setattr(librivad, "hf_tree", lambda *_: tree)
    monkeypatch.setattr(librivad, "download", fake_download)
    monkeypatch.setattr(librivad.subprocess, "run", fake_run)
    librivad.fetch_librivad()
    out = tmp_path / "datasets/raw/librivad"
    assert calls[0][0][-2:] == ["setup.py", "EU"]
    assert calls[1][0][-3:] == ["medium", "--save_dir", str(out / "generated")]
    assert all(c[1:] == (out / "repo", True) for c in calls)
    noises = f"https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/{librivad.LIBRIVAD_REV}/Files/Noises.zip"
    assert (noises, "aa") in downloads
    m = json.loads((out / "manifest.json").read_text())
    assert_manifest_shape(m)
    assert m["labels_dir"] == "repo/Files/Labels"
    assert m["counts"]["LibriSpeech/train-clean-100"] == 1
    assert len(m["counts"]) == 4
    assert m["splits"] == ["train-clean-100", "test-clean"]
    assert not list(out.glob(".*"))  # the markers go once the manifest exists
    assert (out / "repo/Files/Noises/Noises.txt").exists()
    assert not (out / "repo/Files/Noises.zip").exists()
    assert {"url": noises, "sha256": "aa"} in m["sources"]
    assert {"url": noises.replace("Files/Noises.zip", "setup.py")} in m["sources"]
    assert m["revision"] == librivad.LIBRIVAD_REV


def _librivad_stub(monkeypatch, tmp_path, *, labels: bool, marker: bool | None = None):
    """Pre-populate a repo as if setup.py ran; return the recorded downloads and script names."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    files = tmp_path / "datasets/raw/librivad/repo/Files"
    for rel in ("Noises/n.wav", "Forced_alignments/a.txt", "Datasets/Flac/x.flac") + (
        ("Labels/l.npy",) if labels else ()
    ):
        (files / rel).parent.mkdir(parents=True, exist_ok=True)
        (files / rel).touch()
    if labels if marker is None else marker:
        (files.parent.parent / ".setup-done").touch()
    tree = [{"type": "file", "path": z, "lfs": {"oid": "aa"}} for z in librivad.LIBRIVAD_ZIPS]
    downloaded: list[str] = []
    scripts: list[str] = []

    def fake_download(url, dest, **_):
        downloaded.append(url)
        return _fake_file(dest)

    def fake_run(cmd, **kw):
        scripts.append(cmd[cmd.index("python") + 1])
        if "setup.py" in cmd:
            (kw["cwd"] / "Files/Labels").mkdir(parents=True, exist_ok=True)
        else:
            _make_wavs(kw["cwd"].parent / "generated")

    monkeypatch.setattr(librivad, "hf_tree", lambda *_: tree)
    monkeypatch.setattr(librivad, "download", fake_download)
    monkeypatch.setattr(librivad.subprocess, "run", fake_run)
    return downloaded, scripts


def test_fetch_librivad_after_a_crash_before_the_manifest_repeats_nothing(monkeypatch, tmp_path):
    downloaded, scripts = _librivad_stub(monkeypatch, tmp_path, labels=False)
    write_manifest = librivad.write_manifest

    def crash(*_):
        raise OSError("share gone")

    monkeypatch.setattr(librivad, "write_manifest", crash)
    with pytest.raises(OSError, match="share gone"):
        librivad.fetch_librivad()
    downloaded.clear()
    monkeypatch.setattr(librivad, "write_manifest", write_manifest)
    librivad.fetch_librivad()
    assert scripts == ["setup.py", "create_LibriVAD.py"]
    assert not [u for u in downloaded if u.endswith(".zip")]
    assert common.is_complete(tmp_path / "datasets/raw/librivad")


def test_fetch_librivad_reruns_setup_when_labels_exist_but_marker_missing(monkeypatch, tmp_path):
    _, scripts = _librivad_stub(monkeypatch, tmp_path, labels=True, marker=False)
    librivad.fetch_librivad()
    assert scripts == ["setup.py", "create_LibriVAD.py"]


def test_fetch_librivad_after_failed_setup_keeps_the_extracted_zips(monkeypatch, tmp_path):
    # setup.py skips a populated folder and deletes its zip even when extraction fails, so we extract and mark.
    downloaded, scripts = _librivad_stub(monkeypatch, tmp_path, labels=False)
    out = tmp_path / "datasets/raw/librivad"
    working_run = librivad.subprocess.run

    def setup_dies(cmd, **_):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(librivad.subprocess, "run", setup_dies)
    with pytest.raises(subprocess.CalledProcessError):
        librivad.fetch_librivad()
    assert sorted(u.rsplit("/", 1)[1] for u in downloaded if u.endswith(".zip")) == [
        "Forced_alignments.zip",
        "Noises.zip",
    ]
    assert (out / "repo/Files/Noises/Noises.txt").exists()
    downloaded.clear()
    (out / "repo/Files/Noises.zip").touch()  # left behind by a crash after the marker
    monkeypatch.setattr(librivad.subprocess, "run", working_run)
    librivad.fetch_librivad()
    assert not [u for u in downloaded if u.endswith(".zip")]
    assert not (out / "repo/Files/Noises.zip").exists()
    assert scripts == ["setup.py", "create_LibriVAD.py"]


def test_fetch_librivad_small_generates_from_the_medium_setup(monkeypatch, tmp_path):
    _librivad_stub(monkeypatch, tmp_path, labels=True)
    librivad.fetch_librivad()
    medium, small = tmp_path / "datasets/raw/librivad", tmp_path / "datasets/raw/librivad-small"
    calls = []

    def fake_run(cmd, *, cwd, check):
        calls.append((cmd, cwd, check))
        _make_wavs(small / "generated")

    monkeypatch.setattr(librivad.subprocess, "run", fake_run)
    librivad.fetch_librivad_small()
    assert [c[1:] for c in calls] == [(medium / "repo", True)]
    assert calls[0][0][-3:] == ["small", "--save_dir", str(small / "generated")]
    m = json.loads((small / "manifest.json").read_text())
    assert_manifest_shape(m)
    assert m["size"] == "small"
    assert (small / m["labels_dir"]).is_dir()
    assert not list(small.glob(".*"))


@pytest.mark.parametrize("empty_dir", [False, True], ids=["missing", "empty"])
def test_fetch_librivad_rejects_missing_or_empty_split(monkeypatch, tmp_path, empty_dir):
    _librivad_stub(monkeypatch, tmp_path, labels=True)
    gen = tmp_path / "datasets/raw/librivad/generated"

    def fake_run(cmd, **_):
        _make_wavs(gen, skip=("LibriSpeechConcat/test-clean",))
        if empty_dir:
            (gen / "LibriSpeechConcat/test-clean").mkdir()

    monkeypatch.setattr(librivad.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no wav files generated for LibriSpeechConcat/test-clean"):
        librivad.fetch_librivad()
    assert not (tmp_path / "datasets/raw/librivad/manifest.json").exists()


def test_fetch_librivad_raises_without_labels_before_generating(monkeypatch, tmp_path):
    _, scripts = _librivad_stub(monkeypatch, tmp_path, labels=False)
    monkeypatch.setattr(librivad.subprocess, "run", lambda cmd, **_: scripts.append(cmd[cmd.index("python") + 1]))
    with pytest.raises(RuntimeError, match="Labels"):
        librivad.fetch_librivad()
    assert "create_LibriVAD.py" not in scripts


@pytest.fixture
def tmp_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_upstream(monkeypatch, tmp_data_dir):
    """The fetched `librivad` and `librispeech` with stand-in upstream scripts, run by this Python instead of uv."""
    make_upstream()
    monkeypatch.setattr(librivad, "librivad_uv_cmd", lambda *args: [sys.executable, *args])


@pytest.mark.parametrize("missing", ["librivad", "librispeech"])
def test_val_refuses_without_the_fetches_it_reads(fake_upstream, missing):
    (common.raw_dir(missing) / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match=f"just fetch {missing}"):
        librivad.fetch_librivad_val()
    assert not common.raw_dir("librivad-val").exists()


def test_val_refuses_without_librivad_setup(tmp_data_dir):
    with pytest.raises(RuntimeError, match="librivad"):
        librivad.fetch_librivad_val()


def test_val_never_touches_the_librivad_manifest_tree(fake_upstream):
    names = ("librivad", "librispeech")
    before = [tree_digest(common.raw_dir(name)) for name in names]
    librivad.fetch_librivad_val()
    assert [tree_digest(common.raw_dir(name)) for name in names] == before


def test_val_counts_are_recorded(fake_upstream):
    librivad.fetch_librivad_val()
    m = common.read_manifest(common.raw_dir("librivad-val"))
    assert m is not None
    assert_manifest_shape(m)
    assert m["counts"] == {"LibriSpeech/dev-clean": 6 * 4, "LibriSpeechConcat/dev-clean": 3 * 4}
    assert m["splits"] == ["dev-clean"]
    assert m["labels_dir"] == "labels"
    assert [s["url"] for s in m["sources"]][1:] == ["https://openslr.example/dev-clean.tar.gz"]
    assert common.is_complete(common.raw_dir("librivad-val"))
    assert not list(common.raw_dir("librivad-val").glob(".*"))  # the working copy is gone


def test_val_replaces_what_a_killed_run_left(fake_upstream):
    out = common.raw_dir("librivad-val")
    for stale in (out / "generated/old.wav", out / "labels/old.npy"):
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.touch()
    librivad.fetch_librivad_val()
    assert not list(out.rglob("old.*"))


def test_val_stops_when_upstream_loses_a_conversion(fake_upstream, monkeypatch):
    # Upstream prints a failed conversion and goes on, so the script compares the counts itself.
    flac = next(common.raw_dir("librispeech").rglob("*.flac"))
    flac.write_text("not a number")
    with pytest.raises(subprocess.CalledProcessError):
        librivad.fetch_librivad_val()
    assert common.read_manifest(common.raw_dir("librivad-val")) is None


def test_val_output_is_what_the_librivad_adapter_reads(fake_upstream):
    librivad.fetch_librivad_val()
    raw = common.raw_dir("librivad")
    report: dict = {}
    found = list(librivad_source.clips(raw, report=report))
    wavs = sorted(common.raw_dir("librivad-val").glob("generated/*/dev-clean/*/*/*.wav"))
    assert len(wavs) == 36
    assert sorted(c.path for c in found) == sorted(w.relative_to(common.data_dir()).as_posix() for w in wavs)
    assert {c.split for c in found} == {"validation"}
    assert report["label_length_mismatch"] == {}
    timings = list(librivad_source.timings(raw, found))
    assert len(timings) == len(found)
    assert all(t.kind == "frame" and t.speech_ms for t in timings)
