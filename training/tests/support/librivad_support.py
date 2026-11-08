# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A stand-in for the pinned LibriVAD scripts and for a fetched LibriSpeech `dev-clean`, for the val fetcher.

The stand-in scripts have the functions and the path rules of the upstream ones that the fetcher uses (the
`_+_` names, the `Labels` folder found by the first `LibriSpeech` in a path, the output folders of the generator),
with wav files of a few samples. They do not model upstream's wrong labels for a pair across chapters or speakers.
"""

import hashlib
from pathlib import Path

from openvad_train.fetch import common


# The sorted utterances pair up as (1-1-0000, 1-1-0001), (1-2-0000, 2-1-0000) and (2-1-0001, 2-2-0000): the same
# chapter, another speaker and another chapter of the same speaker. The value is the number of samples.
DEV_CLEAN = {
    "1/1/1-1-0000": 3200,
    "1/1/1-1-0001": 3300,
    "1/2/1-2-0000": 3400,
    "2/1/2-1-0000": 3500,
    "2/1/2-1-0001": 3600,
    "2/2/2-2-0000": 3700,
}

_WAV = """
import wave
from pathlib import Path


def write_wav(path: Path, n_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(2 * n_samples))


def n_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()
"""

_TO_WAV = """
from Scripts._wav import write_wav


def process_files_in_parallel(flac_files, flac_root_dir, wav_root_dir):
    for flac in flac_files:
        write_wav(wav_root_dir / flac.relative_to(flac_root_dir).with_suffix(".wav"), int(flac.read_text()))
"""

_CONCAT = """
from pathlib import Path

from Scripts._wav import n_samples, write_wav


def create_LibriSpeechConcat(list_of_paths, silence_path):
    assert Path(silence_path).is_file()
    if len(list_of_paths) % 2:
        list_of_paths.pop()
    for first, second in zip(list_of_paths[::2], list_of_paths[1::2], strict=True):
        path1, path2 = Path(first), Path(second)
        if path1.parts[-3] != path2.parts[-3]:
            name = f"{path1.stem}_+_{path2.name}"
        elif path1.parts[-2] != path2.parts[-2]:
            name = f"{path1.stem}_+_{'-'.join(path2.stem.split('-')[1:])}.wav"
        else:
            name = f"{path1.stem}_+_{path2.stem.split('-')[-1]}.wav"
        parts = list(path2.parts)
        parts[parts.index("LibriSpeech")] = "LibriSpeechConcat"
        parts[-1] = name
        total = n_samples(path1) + n_samples(path2)
        write_wav(Path(*parts), total + total // 4)
"""

_LABELS = """
from pathlib import Path

import numpy as np

from Scripts._wav import n_samples


def create_label_for_file(wav_path_str, alignments_root_dir):
    assert (Path(alignments_root_dir)).is_dir()
    wav_path = Path(wav_path_str)
    parts = list(wav_path.parts)
    parts[parts.index(wav_path.parts[-5]) - 1] = "Labels"
    label_path = Path(*parts).with_suffix(".npy")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    n = n_samples(wav_path)
    label = np.zeros(n, dtype=np.int16)
    label[n // 4 : n // 2] = 1
    np.save(label_path, label)
"""

_GENERATOR = """
import sys
from pathlib import Path

import numpy as np

from Scripts._wav import n_samples, write_wav

assert sys.argv[1] == "medium"
save_dir = Path(sys.argv[sys.argv.index("--save_dir") + 1])
assert Path("Files/Noises/Noises.txt").is_file()
assert Path("Files/Forced_alignments/librispeech_alignments/unaligned.txt").is_file()
for dataset in ("LibriSpeech", "LibriSpeechConcat"):
    for wav in sorted(Path("Files/Datasets", dataset, "dev-clean").glob("*/*/*.wav")):
        # `remove_silence` of upstream finds the label of the clean wav by its path.
        label = np.load(Path("Files/Labels", *wav.parts[2:]).with_suffix(".npy"))
        assert len(label) == n_samples(wav)
        for noise in ("Noise_a", "Noise_b"):
            for snr in ("0", "5"):
                write_wav(save_dir / dataset / "dev-clean" / noise / snr / wav.name, n_samples(wav))
"""

UPSTREAM_SCRIPTS = {
    "Scripts/_wav.py": _WAV,
    "Scripts/create_LibriSpeech_wav.py": _TO_WAV,
    "Scripts/create_LibriSpeechConcat.py": _CONCAT,
    "Scripts/create_labels.py": _LABELS,
    "create_LibriVAD.py": _GENERATOR,
}


def make_upstream() -> None:
    """The `librivad` and `librispeech` fetches under `$OPENVAD_DATA_DIR`, each with its manifest."""
    librivad = common.raw_dir("librivad")
    repo = librivad / "repo"
    for rel, source in UPSTREAM_SCRIPTS.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(source)
    (repo / "requirements.txt").write_text("numpy\n")
    files = repo / "Files"
    for rel, text in (
        ("Noises/Noises.txt", "x"),
        ("Forced_alignments/librispeech_alignments/unaligned.txt", "# test-other\n"),
        ("Datasets/LibriSpeech/train100_silence.wav", "silence"),
        ("Datasets/LibriSpeech/train-clean-100/9/9/9-9-0000.wav", "train"),
    ):
        (files / rel).parent.mkdir(parents=True, exist_ok=True)
        (files / rel).write_text(text)
    common.write_manifest(librivad, {"dataset": "librivad", "labels_dir": "repo/Files/Labels"})
    librispeech = common.raw_dir("librispeech")
    for stem, n in DEV_CLEAN.items():
        flac = librispeech / "LibriSpeech/dev-clean" / f"{stem}.flac"
        flac.parent.mkdir(parents=True, exist_ok=True)
        flac.write_text(str(n))
    sources = [{"url": f"https://openslr.example/{s}.tar.gz", "md5": "aa"} for s in ("train-clean-100", "dev-clean")]
    common.write_manifest(librispeech, {"dataset": "librispeech", "sources": sources})


def tree_digest(root: Path) -> str:
    """A hash over every path, link target and file content under `root`."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        h.update(path.relative_to(root).as_posix().encode())
        if path.is_symlink():
            h.update(str(path.readlink()).encode())
        elif path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()
