# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The stored channel chain after the acoustic mix: room, handset, AGC, soft clipping and noise suppression (ADR 0021).

Audio is float32 at 16 kHz. Every step keeps the input's length and adds no delay: the RIR is cut at its peak, and
the filters run forwards and backwards, so the labels need no shift for any of them. Only the codec leg lags.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from openvad_train.dataset.mix import SAMPLE_RATE
from openvad_train.dataset.schema import _blake2_64


SUPPRESSION_METHODS = ("wiener", "spectral_subtraction")
_SPLITS = ("train", "validation", "test")


def load_rir(path: Path) -> np.ndarray:
    """The first channel of a 16 kHz RIR file as float32."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: a RIR must be 16 kHz, got {rate}")
    return audio[:, 0]


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """`x` convolved with `rir`, cut at the RIR's peak and scaled to unit energy; same length as `x`.

    Cutting at the peak (the direct path) moves it to lag 0, so an onset of `x` stays where it is and the labels
    need no shift (spec, Labels). The tail runs past the end of speech into padding that is labelled non-speech.
    Unit energy keeps the mean power of broadband audio, so the mix level does not depend on the RIR's gain.
    """
    peak = int(np.argmax(np.abs(rir))) if len(rir) else 0
    direct = np.asarray(rir[peak:], np.float64)
    energy = float(np.sum(direct**2))
    if energy == 0.0 or not np.isfinite(energy):
        raise ValueError("a silent or non-finite RIR has no direct path")
    return signal.fftconvolve(x, direct / np.sqrt(energy))[: len(x)].astype(np.float32)


def _peaking(freq_hz: float, gain_db: float, q: float) -> np.ndarray:
    """One second-order section, the peaking filter of the RBJ audio EQ cookbook."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / SAMPLE_RATE
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return np.concatenate([b, den]) / den[0]


def handset(x: np.ndarray, params: Mapping) -> np.ndarray:
    """High-pass, low-pass and two peaks of a telephone handset: `params` is the plan's `handset` struct.

    The filters run forwards and backwards, so there is no phase shift and the magnitude is squared. The peaks
    are designed at half their gain for that, so the audio sees the drawn `gain_db`; the Butterworth corners
    (second order) end up at -6 dB.
    """
    sos = [
        signal.butter(2, params["highpass_hz"], "highpass", fs=SAMPLE_RATE, output="sos"),
        signal.butter(2, params["lowpass_hz"], "lowpass", fs=SAMPLE_RATE, output="sos"),
        *(_peaking(p["freq_hz"], p["gain_db"] / 2, 1.0)[None] for p in params["peaks"]),
    ]
    return signal.sosfiltfilt(np.concatenate(sos), x).astype(np.float32)


_AGC_BLOCK = 16  # 1 ms: gain follows the level of blocks, not samples, so the loop below is short
_AGC_MAX_GAIN_DB = 30.0  # a hand-set limit; without it digital silence would be raised to the target
_AGC_LIMIT_N = 81  # 5 ms at 16 kHz, odd so the window is centred and the limiter adds no delay


def _limit_peak(y: np.ndarray) -> np.ndarray:
    """`y` with its peak at most 1.0, the gain dipping smoothly around an overshoot and nowhere else.

    The gain that would bring each sample to 1.0 is minimum-filtered and then averaged over the same window, so at
    every sample the average cannot exceed that sample's own need. Audio that never exceeds 1.0 is returned as it is.
    """
    magnitude = np.abs(y)
    if magnitude.max(initial=0.0) <= 1.0:
        return y
    need = np.minimum(1.0, 1.0 / np.maximum(magnitude, 1e-30))
    gain = uniform_filter1d(minimum_filter1d(need, _AGC_LIMIT_N, mode="nearest"), _AGC_LIMIT_N, mode="nearest")
    return np.clip(y * gain, -1.0, 1.0)  # the clip only absorbs floating-point rounding


def agc(x: np.ndarray, params: Mapping) -> np.ndarray:
    """Automatic gain control: the plan's `agc` struct gives the target RMS level and the two time constants.

    The level is the RMS of 1 ms blocks smoothed with `attack_ms` when it rises and `release_ms` when it falls.
    The gain `target / level`, at most +30 dB, is interpolated between block centres. The attack lets the first
    block of a loud onset after a quiet stretch through at the old gain, so the output is limited to a peak of 1.0
    here: `protect_peak` after the chain would otherwise scale the whole sample down for one overshoot.
    """
    n = len(x)
    blocks = -(-n // _AGC_BLOCK)
    padded = np.pad(np.asarray(x, np.float64), (0, blocks * _AGC_BLOCK - n))
    level = np.sqrt(np.mean(padded.reshape(blocks, _AGC_BLOCK) ** 2, axis=1))
    attack, release = np.exp(-1 / params["attack_ms"]), np.exp(-1 / params["release_ms"])  # per 1 ms block
    env = np.empty(blocks)
    smooth = level[0]
    for i, now in enumerate(level.tolist()):
        coeff = attack if now > smooth else release
        smooth = coeff * smooth + (1 - coeff) * now
        env[i] = smooth
    gain = np.minimum(10 ** (params["target_dbfs"] / 20) / np.maximum(env, 1e-12), 10 ** (_AGC_MAX_GAIN_DB / 20))
    sample_gain = np.interp(np.arange(n), (np.arange(blocks) + 0.5) * _AGC_BLOCK, gain)
    return _limit_peak(x * sample_gain).astype(np.float32)


def soft_clip(x: np.ndarray, drive_db: float) -> np.ndarray:
    """A `tanh` saturator that keeps the peak of `x`; `drive_db` is how far into saturation the peak is driven.

    The drive is relative to the peak, not to full scale, so a quiet sample is clipped as hard as a loud one.
    """
    peak = float(np.max(np.abs(x), initial=0.0))
    if peak == 0.0:
        return x.astype(np.float32)
    drive = 10 ** (drive_db / 20)
    return (np.tanh(drive * x / peak) * (peak / np.tanh(drive))).astype(np.float32)


_SUPPRESSION_FLOOR = 10 ** (-15 / 20)
_NOISE_PERCENTILE = 20
# The p-th percentile of an exponential distribution (the power of a noise bin) is -ln(1 - p) of its mean.
_NOISE_BIAS = -1 / np.log(1 - _NOISE_PERCENTILE / 100)


def suppress(x: np.ndarray, method: str) -> np.ndarray:
    """Classical noise suppression on the STFT: `wiener` or `spectral_subtraction` (the plan's `suppression`).

    The noise power of each frequency is the 20th percentile of its frame powers over the whole sample, scaled to a
    mean, which holds for stationary noise and for samples with at least 20% non-speech. That is non-causal: the
    first frame is treated with what the last one shows. It is a deliberate choice, since the stored set is built
    offline. The gain has a floor of -15 dB.
    """
    if method not in SUPPRESSION_METHODS:
        raise ValueError(f"unknown suppression {method!r}, expected one of {SUPPRESSION_METHODS}")
    _, _, spectrum = signal.stft(x, nperseg=512)
    power = np.abs(spectrum) ** 2
    noise = np.percentile(power, _NOISE_PERCENTILE, axis=1, keepdims=True) * _NOISE_BIAS
    snr = power / (noise + 1e-30)
    if method == "wiener":
        prior = np.maximum(snr - 1, 0)
        gain = prior / (1 + prior)
    else:
        gain = np.sqrt(np.clip(1 - 1 / np.maximum(snr, 1e-30), 0, 1))
    _, y = signal.istft(spectrum * np.maximum(gain, _SUPPRESSION_FLOOR), nperseg=512)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)].astype(np.float32)


def split_rooms(rooms: set[str]) -> dict[str, list[str]]:
    """Rooms by split: ranked by a hash of the room, the first tenth validation, the next tenth test, the rest train.

    The split is a property of the RIR sets (a room's rank among all rooms), never of a slice or a language
    choice, and at least one room goes to each of validation and test.
    """
    if len(rooms) < 3:
        raise ValueError(f"need at least 3 rooms for 3 splits, got {len(rooms)}")
    ordered = sorted(rooms, key=lambda r: (_blake2_64("rir-room", r), r))
    tenth = max(1, round(len(ordered) / 10))
    return {"validation": ordered[:tenth], "test": ordered[tenth : 2 * tenth], "train": ordered[2 * tenth :]}


class RirDraw(NamedTuple):
    file: Path
    room: str


class RirPool:
    """The RIRs under `root`, as written by `fetch rirs` (`index.tsv` of path and room), split by room.

    A draw picks a room of the split uniformly and then a RIR of it, so a room with many measurements does not
    dominate. It is a pure function of the rng and the sorted index, never of directory order.
    """

    def __init__(self, root: Path) -> None:
        index = root / "index.tsv"
        if not index.is_file():
            raise FileNotFoundError(f"{index} does not exist: run `just fetch rirs`")
        by_room: dict[str, list[Path]] = {}
        for line in index.read_text().splitlines()[1:]:
            path, room = line.split("\t")
            by_room.setdefault(room, []).append(root / path)
        self._files = {room: sorted(files) for room, files in sorted(by_room.items())}
        self._rooms = split_rooms(set(self._files))
        self._room_of = {file: room for room, files in self._files.items() for file in files}

    def rooms(self, split: str) -> list[str]:
        if split not in self._rooms:
            raise ValueError(f"unknown split {split!r}, expected one of {_SPLITS}")
        return self._rooms[split]

    def draw(self, split: str, rng: np.random.Generator) -> RirDraw:
        rooms = self.rooms(split)
        room = rooms[int(rng.integers(len(rooms)))]
        files = self._files[room]
        return RirDraw(files[int(rng.integers(len(files)))], room)

    def room_of(self, file: Path) -> str:
        return self._room_of[file]
