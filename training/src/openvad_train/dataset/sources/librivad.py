# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""LibriVAD `medium`: every generated mixture, labelled by the per-sample speech labels of its clean utterance.

Adapter contract: `MFA_ALIGNED`, `REPORTS`, `RESERVES_SPEAKERS`, `clips(raw, languages, report)` and
`timings(raw, clips)`. `raw` is `datasets/raw/librivad`.
The splits follow the LibriSpeech folder an utterance comes from: `train-clean-100` is train, `test-clean` test
(`raw/generated`), and `dev-clean` validation, which only `datasets/raw/librivad-val` holds (the `librivad-val`
fetcher generates it in the same layout, with its labels in `labels/` beside `generated/`). Nothing here reads
LibriSpeech itself.
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from openvad_train.dataset.labels import FRAME_MS, SAMPLES_PER_FRAME
from openvad_train.dataset.schema import Clip, Split, Timing
from openvad_train.dataset.sources import wav_durations


SOURCE = "librivad"
CUTTING = None
MFA_ALIGNED = False
REPORTS = True
# The spec keeps every file of this source, so none of its speakers is reserved for babble.
RESERVES_SPEAKERS = False
LICENCE = "CC BY-NC-SA 4.0"
LANGUAGE = "en"
SAMPLE_RATE = 16000
DATASETS = ("LibriSpeech", "LibriSpeechConcat")
SPLITS: dict[str, Split] = {"train-clean-100": "train", "dev-clean": "validation", "test-clean": "test"}
VAL_SOURCE = "librivad-val"
# `labels_dir` of the librivad manifest: the labels `setup.py` unpacks, `<dataset>/<split folder>/<speaker>/<chapter>`.
LABELS = "repo/Files/Labels"
VAL_LABELS = "labels"
_THREADS = 16


def frame_labels(path: Path) -> np.ndarray:
    """One label per 10 ms from a per-sample `int16` file (1 speech, 0 non-speech): speech if strictly more than
    half of the frame's 160 samples are, so a tie is non-speech. The frame count is `round(samples / 160)` as in
    `rasterise`; samples past the last frame are not counted, and a short last frame counts its missing samples as
    non-speech.
    """
    samples = np.load(path)
    if samples.ndim != 1 or samples.min() < 0 or samples.max() > 1:
        raise ValueError(f"{path}: expected one 0 or 1 per sample, got shape {samples.shape}")
    n_frames = round(len(samples) / SAMPLES_PER_FRAME)
    padded = np.zeros(n_frames * SAMPLES_PER_FRAME, dtype=np.int64)
    kept = min(len(samples), len(padded))
    padded[:kept] = samples[:kept]
    return (padded.reshape(n_frames, SAMPLES_PER_FRAME).sum(axis=1) * 2 > SAMPLES_PER_FRAME).astype(np.uint8)


def _label_of(labels: Path, dataset: str, folder: str, stem: str) -> Path:
    """The label file of a mixture's utterance. It sits in the `<speaker>/<chapter>` folder of the source tree, and
    for a concatenated pair `A_+_B` that is where B is, which the first part of the name does not say (upstream
    `match_labels.py` reads the folders from it, and is wrong for a pair of different chapters or speakers):
    `1995-1836-0014_+_1837-0000` is under `1995/1837`, and `1580-141084-0050_+_1995-1826-0000` under `1995/1826`.
    """
    first, _, second = stem.partition("_+_")
    speaker, chapter, *_ = first.split("-")
    fields = second.split("-")  # `0001`: same chapter; `1837-0000`: same speaker; `1995-1826-0000`: another speaker
    if len(fields) == 2:
        chapter = fields[0]
    elif len(fields) == 3:
        speaker, chapter = fields[:2]
    return labels / dataset / folder / speaker / chapter / f"{stem}.npy"


def _n_samples(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


def clips(raw: Path, languages: set[str] | None = None, report: dict | None = None) -> Iterator[Clip]:
    """One clip per generated wav of `train-clean-100` and `test-clean` under `raw`, and of `dev-clean` under the
    sibling `librivad-val` when it exists. The length is the wav's header. The 54 mixtures of an utterance (nine
    noises at six SNRs) share its label file, which has one value per sample, so a label whose length is not the
    wav's frame count belongs to another signal and the clip is dropped, never trimmed or stretched: the pairs of
    two chapters or speakers in `LibriSpeechConcat` are such (about 1,890 clips). `report["label_length_mismatch"]`
    counts them per split. The speaker is the LibriSpeech speaker (of the first utterance of a concatenated pair),
    whom the folders keep in one split.
    """
    if languages is not None and LANGUAGE not in languages:
        return
    trees: list[tuple[Path, str, tuple[str, ...]]] = [(raw, LABELS, ("train-clean-100", "test-clean"))]
    if (val := raw.with_name(VAL_SOURCE)).is_dir():
        trees.append((val, VAL_LABELS, ("dev-clean",)))
    found = []
    for tree, labels, folders in trees:
        for dataset in DATASETS:
            for folder in folders:
                for wav in sorted((tree / "generated" / dataset / folder).glob("*/*/*.wav")):
                    found.append((tree, dataset, folder, wav, _label_of(tree / labels, dataset, folder, wav.stem)))
    needed = sorted({label for *_, label in found})
    with ThreadPoolExecutor(_THREADS) as pool:
        lengths = dict(zip(needed, pool.map(_n_samples, needed), strict=True))
    durations = wav_durations([wav for _, _, _, wav, _ in found])
    mismatched: Counter[str] = Counter()
    for (tree, dataset, folder, wav, label), duration in zip(found, durations, strict=True):
        if round(duration * SAMPLE_RATE) != lengths[label]:
            mismatched[SPLITS[folder]] += 1
            continue
        noise, snr = wav.parent.parent.name, wav.parent.name
        yield Clip(
            clip_id=f"{dataset}/{folder}/{noise}/{snr}/{wav.stem}",
            source=SOURCE,
            language=LANGUAGE,
            speaker=wav.stem.split("-")[0],
            split=SPLITS[folder],
            path=(Path("datasets/raw") / tree.name / wav.relative_to(tree)).as_posix(),
            offset_s=0.0,
            duration_s=duration,
            licence=LICENCE,
            transcript=None,
        )
    if report is not None:
        report["label_length_mismatch"] = dict(sorted(mismatched.items()))


@lru_cache(maxsize=8192)
def _speech_ms(path: Path) -> tuple[tuple[int, int], ...]:
    """The runs of speech frames as millisecond intervals."""
    edges = np.diff(np.concatenate([[0], frame_labels(path), [0]]).astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return tuple((int(s) * FRAME_MS, int(e) * FRAME_MS) for s, e in zip(starts, ends, strict=True))


def _label_of_clip(raw: Path, clip: Clip) -> Path:
    dataset, folder, _, _, stem = clip.clip_id.split("/")
    tree = raw.parent / Path(clip.path).parts[2]
    return _label_of(tree / (VAL_LABELS if tree.name == VAL_SOURCE else LABELS), dataset, folder, stem)


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    """Frame labels as they are: `kind = "frame"`, so no gap rule is applied to them and they come back from
    `rasterise` unchanged. The 54 mixtures of an utterance read one label file, and the files of a batch are read in
    parallel, since each costs about 100 ms over the share.
    """
    clips = list(clips)
    paths = [_label_of_clip(raw, clip) for clip in clips]
    with ThreadPoolExecutor(_THREADS) as pool:
        list(pool.map(_speech_ms, set(paths)))  # fills the cache
    for clip, path in zip(clips, paths, strict=True):
        yield Timing(clip.clip_id, "frame", list(_speech_ms(path)), [], None, "librivad:labels")
