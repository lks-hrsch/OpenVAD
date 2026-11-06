# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Picovoice voice-activity-benchmark test audio, built by its own mixer at a pinned commit (Apache-2.0)."""

import subprocess
import tarfile
import tempfile
from pathlib import Path

from openvad_train.fetch.common import cache_dir, download, is_complete, raw_dir, tar_filter, write_manifest


PICOVOICE_REPO = "Picovoice/voice-activity-benchmark"
PICOVOICE_REV = "38351b25ba883d4ae414bd48e16e8d4999813edf"
# The mixer imports only these, pinned as in requirements.txt at PICOVOICE_REV.
PICOVOICE_DEPS = ("numpy==2.3.3", "soundfile==0.13.1")
# Upstream main() without the engine run: mix test-clean with DEMAND at its default 0 dB.
PICOVOICE_SCRIPT = (
    "import sys; from dataset import Dataset, Datasets; from mixer import create_test_files; "
    "create_test_files(speech_path=sys.argv[3], label_path=sys.argv[4], "
    "speech_dataset=Dataset.create(Datasets.LIBRI_SPEECH, sys.argv[1]), "
    "noise_dataset=Dataset.create(Datasets.DEMAND, sys.argv[2]))"
)


def picovoice_cmd(test_clean: Path, demand: Path, speech: Path, labels: Path) -> list[str]:
    deps = [arg for dep in PICOVOICE_DEPS for arg in ("--with", dep)]
    cmd = ["uv", "run", "--no-project", "--python", "3.12", *deps, "python", "-c", PICOVOICE_SCRIPT]
    return [*cmd, str(test_clean), str(demand), str(speech), str(labels)]


def fetch_picovoice() -> None:
    librispeech, demand = raw_dir("librispeech"), raw_dir("demand")
    if not (is_complete(librispeech) and is_complete(demand)):
        raise RuntimeError("Picovoice mixes LibriSpeech test-clean with DEMAND; fetch `librispeech` and `demand` first")
    out = raw_dir("picovoice")
    repo = out / "repo"
    url = f"https://codeload.github.com/{PICOVOICE_REPO}/tar.gz/{PICOVOICE_REV}"
    if not repo.exists():
        archive = download(url, cache_dir() / f"picovoice-{PICOVOICE_REV}.tar.gz")
        with tarfile.open(archive) as tar:
            tar.extractall(out, filter=tar_filter)
        (out / f"voice-activity-benchmark-{PICOVOICE_REV}").rename(repo)
        archive.unlink()
    speech, labels = out / "benchmark_speech.wav", out / "benchmark_labels.txt"
    if not (speech.exists() and labels.exists()):
        with tempfile.TemporaryDirectory() as view:
            # Upstream lists every entry of the DEMAND folder as an environment, so hide our manifest.
            for scene in sorted(p for p in demand.iterdir() if p.is_dir()):
                (Path(view) / scene.name).symlink_to(scene)
            test_clean = librispeech / "LibriSpeech/test-clean"
            # Final names appear only after the mixer succeeds; soundfile picks the format from the `.wav` suffix.
            speech_part, labels_part = out / "benchmark_speech.part.wav", out / "benchmark_labels.part.txt"
            speech_part.unlink(missing_ok=True)
            labels_part.unlink(missing_ok=True)
            subprocess.run(picovoice_cmd(test_clean, Path(view), speech_part, labels_part), cwd=repo, check=True)
            labels_part.rename(labels)
            speech_part.rename(speech)
    info = {
        "dataset": "picovoice",
        "licence": "Apache-2.0 (code); audio from LibriSpeech test-clean (CC BY 4.0) and DEMAND",
        "use": "evaluation only",
        "revision": PICOVOICE_REV,
        "sources": [{"url": url}],
        "deps": list(PICOVOICE_DEPS),
        "snr_db": 0,
        "labels": "one line per 512-sample frame: 0 silence, 1 unknown (not scored), 2 voice",
    }
    write_manifest(out, info)
