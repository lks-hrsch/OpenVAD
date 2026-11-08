# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The noise sources of the acoustic mix: the file pool (DEMAND, WHAM! and MUSAN), the reserved talkers and the
babble, speech-shaped noise and competing talkers built from them.
"""

import csv
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.linalg import solve_toeplitz

from openvad_train.dataset.channel import apply_rir, load_rir
from openvad_train.dataset.codecs import DecodeError
from openvad_train.dataset.mix import SAMPLE_RATE, read_span, rms, speech_rms
from openvad_train.dataset.schema import Clip


# LibriVAD's categories, keyed by the first letter of the DEMAND scene name.
_CATEGORIES = {
    "D": "domestic",
    "N": "nature",
    "O": "office",
    "P": "public",
    "S": "street",
    "T": "transport",
}
# The hand-reviewed sort of the MUSAN noise files into the nine categories (spec, "Acoustic mix").
MUSAN_CATEGORIES = Path(__file__).with_name("musan_noise_categories.tsv")
# Channels 1 to 12 train, 13 and 14 validation, 15 and 16 test (owner, 2026-11-06).
_CHANNELS = {"train": range(1, 13), "validation": range(13, 15), "test": range(15, 17)}
# WHAM! `tt` is test only and never trains (ADR 0018).
_WHAM_SPLITS = {"train": "tr", "validation": "cv", "test": "tt"}
# Japanese and Mandarin are tokenised by MFA unlike the rest, so their babble is same-language only half the time
# (ADR 0023, owner 2026-11-03). Taiwan Mandarin counts as Mandarin; Cantonese (`yue`) does not.
OPEN_LANGUAGES = frozenset({"ja", "zh", "zh-TW"})
OPEN_LANGUAGE_P = 0.5
_HEADER_WORKERS = 32


class NoiseDraw(NamedTuple):
    file: Path
    channel: int
    offset_n: int
    stereo: bool = False

    def source_id(self, root: Path) -> str:
        """The plan's `noise_source_ids` entry: the path under `root`, and `#<channel>` for a two-channel file."""
        return self.file.relative_to(root).as_posix() + (f"#{self.channel}" if self.stereo else "")


def category_of_demand(scene: str) -> str:
    if scene[:1] not in _CATEGORIES:
        raise ValueError(f"not a DEMAND scene: {scene!r}")
    return _CATEGORIES[scene[0]]


def musan_noise_categories() -> dict[str, str]:
    """Category of every MUSAN noise file ID, or `unused`."""
    with MUSAN_CATEGORIES.open() as f:
        return {row["file"]: row["category"] for row in csv.DictReader(f, delimiter="\t")}


def demand_channels(split: str) -> range:
    if split not in _CHANNELS:
        raise ValueError(f"unknown split: {split!r}")
    return _CHANNELS[split]


def wham_split(split: str) -> str:
    if split not in _WHAM_SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    return _WHAM_SPLITS[split]


class NoisePool:
    """DEMAND under `demand` as `<SCENE>/ch01.wav` to `ch16.wav`, WHAM! under `wham` as `wham_noise/<tr|cv|tt>/*.wav`
    and the MUSAN noise files of the category table under `musan`; any of the three may be absent.

    A category draws a DEMAND scene or a MUSAN file with a probability in proportion to its duration, so MUSAN's
    few minutes add variety without drowning the scenes. A scene takes a channel of the split, MUSAN is shared by
    all splits (ADR 0023). `city` is WHAM!: a random file of the split and one of its two channels. A draw is a
    pure function of the rng and the sorted names, never of directory listing order, so a plan row reproduces it.
    """

    def __init__(self, demand: Path | None = None, wham: Path | None = None, musan: Path | None = None) -> None:
        scenes = (
            sorted(p for p in demand.iterdir() if p.name[:1] in _CATEGORIES and (p / "ch01.wav").is_file())
            if demand
            else []
        )
        self._scenes = {c: [p for p in scenes if category_of_demand(p.name) == c] for c in _CATEGORIES.values()}
        self._musan = {c: [] for c in _CATEGORIES.values()}
        for file_id, category in sorted(musan_noise_categories().items()):
            if musan is not None and category != "unused":
                folder = file_id.removeprefix("noise-").rsplit("-", 1)[0]
                self._musan[category].append(musan / "musan/noise" / folder / f"{file_id}.wav")
        if missing := [f for files in self._musan.values() for f in files if not f.is_file()]:
            raise FileNotFoundError(f"MUSAN is incomplete, {missing[0]} does not exist: run `just fetch musan`")
        self._wham = wham / "wham_noise" if wham else None
        self._wham_files: dict[str, list[Path]] = {}
        self._units_of: dict[str, tuple[list[tuple[Path, bool]], list[int]]] = {}
        # A header read costs about 50 ms cold on the share, which adds up over millions of draws.
        self._frames: dict[Path, int] = {}

    def _n_frames(self, file: Path) -> int:
        if file not in self._frames:
            self._frames[file] = sf.info(file).frames
        return self._frames[file]

    def _units(self, category: str) -> tuple[list[tuple[Path, bool]], list[int]]:
        """The scenes (as their `ch01.wav`, which stands for the scene's length) and MUSAN files of `category`, each
        with whether it is a scene, and their running duration in samples.
        """
        if category not in self._units_of:
            units = [(s / "ch01.wav", True) for s in self._scenes[category]] + [
                (f, False) for f in self._musan[category]
            ]
            self._units_of[category] = (units, list(accumulate(self._n_frames(f) for f, _ in units)))
        return self._units_of[category]

    def _draw_wham(self, split: str, rng: np.random.Generator) -> NoiseDraw:
        if self._wham is None:
            raise ValueError("city noise needs WHAM!, which this pool was given no directory for")
        folder = wham_split(split)
        if folder not in self._wham_files:
            self._wham_files[folder] = sorted((self._wham / folder).glob("*.wav"))
            # One header per file, cold over the share (about 1.5 s each), would cost hours one draw at a time: read
            # the whole split once, in parallel. The counts stay exact, never derived from file sizes.
            with ThreadPoolExecutor(_HEADER_WORKERS) as threads:
                counts = threads.map(lambda f: sf.info(f).frames, self._wham_files[folder])
                self._frames |= dict(zip(self._wham_files[folder], counts, strict=True))
        files = self._wham_files[folder]
        if not files:
            raise ValueError(f"no WHAM! file in {self._wham / folder}")
        file = files[int(rng.integers(len(files)))]
        channel = int(rng.integers(2))
        return NoiseDraw(file, channel, int(rng.integers(self._n_frames(file))), stereo=True)

    def draw(self, category: str, split: str, rng: np.random.Generator) -> NoiseDraw:
        if category == "city":
            return self._draw_wham(split, rng)
        if category not in self._scenes:
            raise ValueError(f"not a noise category drawn from files: {category!r}")
        units, running = self._units(category)
        if not units:
            raise ValueError(f"no noise in category {category!r}")
        unit, is_scene = units[bisect_right(running, rng.random() * running[-1])]
        channels = demand_channels(split)
        if is_scene:
            channel = int(rng.integers(channels.start, channels.stop))
            file = unit.parent / f"ch{channel:02d}.wav"
        else:
            channel, file = 0, unit
        # The offset is uniform over the file; the caller wraps it around with `loop_slice`.
        return NoiseDraw(file, channel, int(rng.integers(self._n_frames(file))))


class TalkerPool:
    """The reserved clips that feed babble, speech-shaped noise and competing talkers, by source, split and language.

    A clip is a whole file (`offset_s == 0`): AMI and VoxPopuli clips are spans of recordings that hold other
    speakers, which looping the file would leak. A draw takes the reserved clips of one source and the row's own
    split, so validation and test never hear a training speaker, and adding a source to a build changes no other
    source's draw. The row's language is used when the source has at least `min_speakers` speakers of it, reserved
    or not, all splits (`speakers`, counted by the index and fixed by the source); a smaller language takes every
    language of the source. Japanese and Mandarin take their own language only half the time. If the same-language
    reserved clips of the split hold fewer speakers than are asked for, the rest come from every language. A draw is
    a pure function of the rng and the sorted clips.
    """

    def __init__(self, clips: Iterable[Clip], speakers: Mapping[tuple[str, str], int], min_speakers: int) -> None:
        self._min_speakers = min_speakers
        self._speakers = dict(speakers)
        self._in_split: dict[tuple[str, str], list[Clip]] = defaultdict(list)
        self._in_language: dict[tuple[str, str, str], list[Clip]] = defaultdict(list)
        for clip in sorted(clips, key=lambda c: (c.source, c.clip_id)):
            if not clip.reserved or clip.offset_s != 0:
                raise ValueError(
                    f"{clip.source}/{clip.clip_id} is not a reserved whole-file clip, so it cannot be a talker"
                )
            self._in_split[clip.source, clip.split].append(clip)
            self._in_language[clip.source, clip.split, clip.language].append(clip)

    def has(self, source: str, split: str) -> bool:
        return bool(self._in_split.get((source, split)))

    def draw(self, source: str, language: str, split: str, n: int, rng: np.random.Generator) -> list[Clip]:
        """`n` clips of different speakers where the pool has them, else of clips not yet drawn, and only then a
        repeated clip (a pool of fewer than `n` clips).
        """
        everyone = self._in_split.get((source, split), [])
        if not everyone:
            raise ValueError(f"no reserved talker of {source} in the {split} split")
        borrow = language in OPEN_LANGUAGES and rng.random() < OPEN_LANGUAGE_P
        enough = self._speakers.get((source, language), 0) >= self._min_speakers
        picked: list[Clip] = []
        if enough and not borrow:
            _add_new_speakers(self._in_language.get((source, split, language), []), picked, n, rng)
        _add_new_speakers(everyone, picked, n, rng)
        _add_unpicked_clips(everyone, picked, n, rng)
        while len(picked) < n:  # fewer clips than asked for
            picked.append(everyone[int(rng.integers(len(everyone)))])
        return picked


def _add_new_speakers(pool: list[Clip], picked: list[Clip], n: int, rng: np.random.Generator) -> None:
    """Random clips of `pool` whose speaker is not in `picked`, until `picked` has `n` (a bounded number of tries)."""
    for _ in range(20 * n):
        if len(picked) == n or not pool:
            return
        clip = pool[int(rng.integers(len(pool)))]
        if clip.speaker not in {c.speaker for c in picked}:
            picked.append(clip)


def _add_unpicked_clips(pool: list[Clip], picked: list[Clip], n: int, rng: np.random.Generator) -> None:
    for _ in range(20 * n):
        if len(picked) == n:
            return
        clip = pool[int(rng.integers(len(pool)))]
        if clip not in picked:
            picked.append(clip)


def talker_offset(clip: Clip, span_n: int, rng: np.random.Generator) -> int:
    """Where in the file of `clip` a read of `span_n` samples starts: anywhere that keeps it inside the clip, else 0
    (the read then wraps around the clip).
    """
    clip_n = round(clip.duration_s * SAMPLE_RATE)
    return int(rng.integers(clip_n - span_n + 1)) if clip_n > span_n else 0


def build_babble(paths: list[str], offsets: list[int], n: int, raw: Path) -> np.ndarray:
    """Talkers summed at equal energy, as LibriVAD's babble: `n` samples of each file at its offset (wrapping), each
    scaled to unit RMS over the whole span, then added and divided by `sqrt(len(paths))`.

    Reserved clips are never aligned, so the energy is over the pauses too. `DecodeError` for a damaged or silent file.
    """
    if len(paths) != len(offsets) or not paths:
        raise ValueError(f"need one offset per talker, got {len(paths)} paths and {len(offsets)} offsets")
    talkers = [read_span(str(raw / path), offset, n) for path, offset in zip(paths, offsets, strict=True)]
    return (np.sum([t / rms(t) for t in talkers], axis=0) / np.sqrt(len(talkers))).astype(np.float32)


def fit_lpc(speech: np.ndarray, order: int = 12) -> np.ndarray:
    """The all-pole filter `[1, a1, ..., a_order]` of the autocorrelation method over the whole of `speech`, so its
    long-term average spectrum, which is what speech-shaped noise copies. Stable by construction.
    """
    x = np.asarray(speech, np.float64)
    r = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1 : len(x) + order]
    if r[0] == 0:
        raise DecodeError("cannot fit an all-pole filter to digital silence")
    r[0] *= 1 + 1e-9  # keeps the matrix positive definite for a band-limited signal
    return np.concatenate([[1.0], solve_toeplitz(r[:order], -r[1 : order + 1])])


_SSN_WARMUP_N = 4096


def build_ssn(lpc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` samples of Gaussian white noise through the all-pole filter `lpc`, at unit RMS."""
    y = np.asarray(signal.lfilter([1.0], lpc, rng.standard_normal(n + _SSN_WARMUP_N)))[_SSN_WARMUP_N:]
    return (y / rms(y)).astype(np.float32)


# Each segment fades in and out over 10 ms inside its own span, so its edges are no cue to the model; with both
# ramps and a body, 40 ms is the shortest a segment can be. The plan keeps to `competing_min_segment_ms`, far above.
_FADE_N = SAMPLE_RATE // 100
MIN_SEGMENT_N = 4 * _FADE_N


class Segment(NamedTuple):
    """One competing-talker segment: samples `start_n` to `end_n` of the sample hold `source_path` read from
    `source_offset_n` (the plan's `segments` entry), through `rir_id` and a low-pass at `lowpass_hz`.
    """

    clip_id: str
    start_n: int
    end_n: int
    rir_id: str | None
    lowpass_hz: float
    source_path: str
    source_offset_n: int


def mix_competing(
    speech: np.ndarray,
    active: np.ndarray,
    segments: list[Segment],
    tir_db: float,
    raw: Path,
    baseline_snr_db: float | None = None,
) -> np.ndarray:
    """`speech` plus the segments' talkers, each reverberated with its RIR and low-passed (ADR 0022), at `tir_db`.

    The TIR is the RMS of `speech` over the `active` samples against the RMS of the talkers over the samples the
    segments cover, which is how loud the competing talker is when it talks. Measured over `active` alone it would
    fall with the coverage and can be zero where the talker does not overlap a word. Segments must not overlap, and
    each is at least `MIN_SEGMENT_N` long: the low-pass cannot filter a handful of samples, and the 10 ms
    raised-cosine ramps at both ends (applied before the TIR is measured) need room. `baseline_snr_db` is as in
    `speech_rms`. `DecodeError` for a talker file that cannot be read.
    """
    if active.dtype != bool or active.shape != speech.shape:
        raise ValueError("active must be a boolean mask the shape of speech")
    if not segments:
        raise ValueError("a competing-talker row needs at least one segment")
    if not active.any() or not speech[active].any():
        raise ValueError("no speech to measure the TIR against")
    track, covered = np.zeros(len(speech), np.float32), np.zeros(len(speech), bool)
    for seg in segments:
        n = seg.end_n - seg.start_n
        if not 0 <= seg.start_n < seg.end_n <= len(speech) or covered[seg.start_n : seg.end_n].any():
            raise ValueError(f"segment {seg.start_n} to {seg.end_n} is empty, overlaps another or leaves {len(speech)}")
        if n < MIN_SEGMENT_N:
            raise ValueError(f"{seg.clip_id}: a segment of {n} samples is under the {MIN_SEGMENT_N} the ramps need")
        if seg.rir_id is None:
            raise ValueError(f"{seg.clip_id}: a competing segment without a rir_id")
        talker = apply_rir(read_span(str(raw / seg.source_path), seg.source_offset_n, n), load_rir(raw / seg.rir_id))
        lowpass = signal.butter(4, seg.lowpass_hz, "lowpass", fs=SAMPLE_RATE, output="sos")
        faded = signal.sosfiltfilt(lowpass, talker)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(_FADE_N) / _FADE_N))
        faded[:_FADE_N] *= ramp
        faded[-_FADE_N:] *= ramp[::-1]
        track[seg.start_n : seg.end_n] = faded
        covered[seg.start_n : seg.end_n] = True
    gain = speech_rms(speech, active, baseline_snr_db) / (rms(track[covered]) * 10 ** (tir_db / 20))
    return (speech + gain * track).astype(np.float32)
