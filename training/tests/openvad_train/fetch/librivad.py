import json
import subprocess
import zipfile

import pytest

from fetch_support import assert_manifest_shape
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
