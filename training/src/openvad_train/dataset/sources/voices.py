# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""VOiCES devkit: every far-field re-recording of a LibriSpeech utterance, labelled by the utterance's alignment.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)` and `timings(raw, clips)`. `raw` is `datasets/raw/voices`.
The devkit ships the source audio (`source-16k`), so the delay between a re-recording and its source is measured on
the files, and the LibriSpeech alignments of `datasets/raw/librispeech-alignments` (Zenodo 2619474, labels only)
are shifted by it. The distractor-only noise recordings (`distant-16k/distractors`) are not used.
"""

import csv
import io
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from openvad_train.dataset.cutting import rebase
from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.schema import Clip, Split, Timing
from openvad_train.dataset.sources.swc import assign_splits
from openvad_train.dataset.textgrid import parse_textgrid


SOURCE = "voices"
CUTTING = None
MFA_ALIGNED = False
RAW_PREFIX = Path("datasets/raw/voices")
LICENCE = "CC BY 4.0"
LANGUAGE = "en"
DEVKIT = "VOiCES_devkit"
ALIGNMENTS = "librispeech-alignments"
# Where each speaker's LibriSpeech utterances come from (ADR 0023): `dev-clean` speakers are dropped, `train-clean-100`
# speakers are train only because LibriVAD has their speech, `train-clean-360` speakers are split 80/10/10.
DROPPED_DATASET = "dev-clean"
TRAIN_ONLY_DATASET = "train-clean-100"
SPLIT_DATASET = "train-clean-360"
# The loudspeaker is metres from the microphone (3 ms per metre), so a larger lag is a wrong correlation peak.
MAX_DELAY_S = 0.25
_THREADS = 16

_NAME = re.compile(r"Lab41-SRI-VOiCES-rm\d-\w+-sp(?P<speaker>\d+)-ch(?P<chapter>\d+)-sg(?P<segment>\d+)-")


def _read_mono(wav: Path) -> tuple[np.ndarray, int]:
    # One request for the whole file: libsndfile's many small reads cost a round trip each over the share.
    audio, rate = sf.read(io.BytesIO(wav.read_bytes()), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def _lag_s(noisy: np.ndarray, source: np.ndarray, rate: int) -> float:
    corr = correlate(noisy, source, mode="full", method="fft")
    lags = correlation_lags(len(noisy), len(source), mode="full")
    near = np.abs(lags) <= MAX_DELAY_S * rate
    return float(lags[near][np.argmax(corr[near])] / rate)


def measure_voices_delay(voices_wav: Path, source_wav: Path) -> float:
    """Seconds by which the re-recording lags its source: the lag, within `MAX_DELAY_S` either way, at the peak of
    their cross-correlation; positive when the re-recording is later. The labels of the source move by this much.
    The devkit's own `references/time_values.csv` holds the same lags with the opposite sign (30 of 30 random
    recordings agreed to the sample).
    """
    (noisy, rate), (source, source_rate) = _read_mono(voices_wav), _read_mono(source_wav)
    if rate != source_rate:
        raise ValueError(f"{voices_wav} is {rate} Hz and {source_wav} {source_rate} Hz")
    return _lag_s(noisy, source, rate)


def _speaker_datasets(root: Path) -> dict[str, str]:
    """Speaker ID (four digits) to the LibriSpeech set the speaker is from."""
    lines = (root / "references/Lab41-SRI-VOiCES-speaker-gender-dataset.tbl").read_text().splitlines()[1:]
    return {fields[0]: fields[2] for line in lines if len(fields := line.split()) == 3}


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    """One clip per re-recording in the devkit's two index files, whole. The splits follow the speakers'
    LibriSpeech sets and so depend only on the devkit: the 224 `train-clean-360` speakers are split by speaker
    over their hours, as SWC splits readers. A speaker's re-recordings all go to one split.
    """
    if languages is not None and LANGUAGE not in languages:
        return
    root = raw / DEVKIT
    datasets = _speaker_datasets(root)
    rows = []
    for index in ("train_index.csv", "test_index.csv"):
        with (root / "references" / index).open(newline="") as f:
            rows += [r for r in csv.DictReader(f) if r["filename"].startswith("distant-16k/speech/")]
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        speaker = f"{int(r['speaker']):04d}"
        if datasets[speaker] == SPLIT_DATASET:
            hours[speaker][LANGUAGE] += int(r["noisy_length"]) / int(r["noisy_sr"]) / 3600
    split_of = assign_splits(hours)
    for r in rows:
        speaker = f"{int(r['speaker']):04d}"
        origin = datasets[speaker]
        if origin == DROPPED_DATASET:
            continue
        if origin not in (TRAIN_ONLY_DATASET, SPLIT_DATASET):
            raise ValueError(f"VOiCES speaker {speaker} is from {origin!r}, which ADR 0023 has no rule for")
        split: Split = "train" if origin == TRAIN_ONLY_DATASET else split_of[speaker]
        yield Clip(
            clip_id=Path(r["filename"]).stem,
            source=SOURCE,
            language=LANGUAGE,
            speaker=speaker,
            split=split,
            path=(RAW_PREFIX / DEVKIT / r["filename"]).as_posix(),
            offset_s=0.0,
            duration_s=int(r["noisy_length"]) / int(r["noisy_sr"]),
            licence=LICENCE,
            transcript=None,
        )


@cache
def _source_words(textgrid: Path) -> list[tuple[int, int]]:
    """The word intervals of one LibriSpeech utterance in ms; the words tier is empty text for silence."""
    tiers = parse_textgrid(textgrid.read_text(encoding="utf-8"))
    words = [(round(a * 1000), round(b * 1000)) for a, b, text in tiers["words"] if text.strip()]
    return [(s, e) for s, e in words if e > s]


def _source_wav(raw: Path, clip: Clip) -> Path:
    match = _NAME.search(clip.clip_id)
    assert match is not None, clip.clip_id
    speaker, chapter, segment = match["speaker"], match["chapter"], match["segment"]
    devkit_split = "test" if "/speech/test/" in clip.path else "train"
    name = f"Lab41-SRI-VOiCES-src-sp{speaker}-ch{chapter}-sg{segment}.wav"
    return raw / DEVKIT / "source-16k" / devkit_split / f"sp{speaker}" / name


def _delays_ms(raw: Path, clips: list[Clip]) -> dict[str, int]:
    """The delay of each recording behind its source in ms. A source is read once for all its recordings (the
    devkit has 32 of each), and sources are handled in parallel.
    """
    by_source: dict[Path, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_source[_source_wav(raw, clip)].append(clip)

    def measure(item: tuple[Path, list[Clip]]) -> dict[str, int]:
        source_wav, group = item
        source, source_rate = _read_mono(source_wav)
        out = {}
        for clip in group:
            noisy, rate = _read_mono(raw / Path(clip.path).relative_to(RAW_PREFIX))
            if rate != source_rate:
                raise ValueError(f"{clip.path} is {rate} Hz and {source_wav} {source_rate} Hz")
            out[clip.clip_id] = round(_lag_s(noisy, source, rate) * 1000)
        return out

    with ThreadPoolExecutor(_THREADS) as pool:
        return {k: v for delays in pool.map(measure, by_source.items()) for k, v in delays.items()}


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    """The source utterance's words, shifted by the measured delay and cut to the recording."""
    clips = list(clips)
    datasets = _speaker_datasets(raw / DEVKIT)
    delays = _delays_ms(raw, clips)
    for clip in clips:
        match = _NAME.search(clip.clip_id)
        assert match is not None, clip.clip_id
        speaker, chapter, segment = match["speaker"], match["chapter"], match["segment"]
        grid = (
            raw.parent
            / ALIGNMENTS
            / datasets[speaker]
            / str(int(speaker))
            / str(int(chapter))
            / f"{int(speaker)}-{int(chapter)}-{int(segment):04d}.TextGrid"
        )
        shifted = [(s + delays[clip.clip_id], e + delays[clip.clip_id]) for s, e in _source_words(grid)]
        speech = apply_gap_rule(rebase(shifted, 0, round(clip.duration_s * 1000)), 0)
        yield Timing(clip.clip_id, "word", speech, [], None, "voices:librispeech-alignments")
