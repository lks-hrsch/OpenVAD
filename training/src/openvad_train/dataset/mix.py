# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Acoustic mix primitives. Audio is float32 in [-1, 1] at 16 kHz."""

import numpy as np

from openvad_train.dataset.codecs import DecodeError, decode_16k


SAMPLE_RATE = 16_000
_DTMF_ROWS = (697, 770, 852, 941)
_DTMF_COLUMNS = (1209, 1336, 1477, 1633)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, active: np.ndarray, snr_db: float) -> np.ndarray:
    """Speech plus noise, the noise gain set so speech RMS over `active` samples
    is `snr_db` above noise RMS over the same samples. The gain applies to the
    whole noise, including outside `active`.
    """
    if active.dtype != bool:  # integer labels would silently fancy-index
        raise TypeError(f"active must be a boolean mask, got {active.dtype}")
    if not speech.shape == noise.shape == active.shape:
        raise ValueError("speech, noise and active must have the same shape")
    if not active.any():
        raise ValueError("no active samples to measure the SNR on")
    speech_rms, noise_rms = rms(speech[active]), rms(noise[active])
    if speech_rms == 0 or noise_rms == 0:
        raise ValueError("speech or noise is silent over the active samples; the SNR is undefined")
    gain = speech_rms / (noise_rms * 10 ** (snr_db / 20))
    return (speech + gain * noise).astype(np.float32)


def place(speech: np.ndarray, total_n: int, offset_n: int) -> np.ndarray:
    """`speech` in one block at `offset_n` inside `total_n` samples of digital silence."""
    if offset_n < 0 or offset_n + len(speech) > total_n:
        raise ValueError(f"{len(speech)} samples at offset {offset_n} do not fit in {total_n}")
    out = np.zeros(total_n, dtype=np.float32)
    out[offset_n : offset_n + len(speech)] = speech
    return out


def loop_slice(noise: np.ndarray, offset_n: int, n: int) -> np.ndarray:
    """`n` samples from `offset_n`, wrapping around the end when the noise is shorter."""
    if noise.ndim != 1 or len(noise) == 0:
        raise ValueError(f"noise must be non-empty and one-dimensional, got shape {noise.shape}")
    if n < 0:
        raise ValueError(f"cannot take {n} samples")
    return np.take(noise, np.arange(offset_n, offset_n + n), mode="wrap")


def read_span(path: str, offset_n: int, n: int) -> np.ndarray:
    """`n` samples of the speech file at `path` from sample `offset_n`, repeating what the file holds after the offset
    when it ends first.

    The file is the whole clip (a talker is never a span of a longer recording), and the plan starts a read at 0
    when the clip is shorter than the span, so a repeat is of the whole clip.
    `DecodeError` for a file that is damaged, empty after the offset or digital silence, none of which can mix.
    """
    audio = decode_16k(path, offset_n / SAMPLE_RATE, n / SAMPLE_RATE)
    if len(audio) == 0:
        raise DecodeError(f"{path}: no audio after sample {offset_n}")
    audio = loop_slice(audio, 0, n)
    if not audio.any():
        raise DecodeError(f"{path}: digital silence from sample {offset_n}")
    return audio


def dither(n: int, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise of RMS `dbfs`, so padding is never digital silence."""
    return (rng.standard_normal(n) * 10 ** (dbfs / 20)).astype(np.float32)


def protect_peak(x: np.ndarray) -> np.ndarray:
    """Scale the whole sample down only when its peak exceeds 1.0, which keeps the SNR."""
    x = x.astype(np.float32)
    peak = np.abs(x).max(initial=0.0)
    if not np.isfinite(peak):
        raise ValueError("cannot protect the peak of a sample with NaN or infinity")
    return x / peak if peak > 1.0 else x


def overlay_transient(x: np.ndarray, event: np.ndarray, offset_n: int, peak_ratio_db: float) -> np.ndarray:
    """Add `event` at `offset_n`, scaled so its peak is `peak_ratio_db` relative to the peak of `x`.

    An event running past the end is cut. The sum may exceed 1.0; call `protect_peak` after.
    """
    if not 0 <= offset_n < len(x):
        raise ValueError(f"offset {offset_n} is outside a sample of {len(x)}")
    x_peak, event_peak = np.abs(x).max(), np.abs(event).max()
    if x_peak == 0 or event_peak == 0:
        raise ValueError("a silent sample or event has no peak to scale against")
    out = x.astype(np.float32)
    cut = event[: len(x) - offset_n]
    out[offset_n : offset_n + len(cut)] += cut * (x_peak * 10 ** (peak_ratio_db / 20) / event_peak)
    return out


def add_transients(x: np.ndarray, transients: list[dict], rng: np.random.Generator) -> np.ndarray:
    """`x` with the plan's `transients` overlaid in order, each a synthetic click or DTMF digit at its offset and
    peak ratio. The sum can pass 1.0: this belongs before `protect_peak`, never after.
    """
    for t in transients:
        if t["kind"] not in ("click", "dtmf"):
            raise ValueError(f"unknown transient {t['kind']!r}, expected click or dtmf")
        event = synth_click(rng) if t["kind"] == "click" else synth_dtmf(rng)
        x = overlay_transient(x, event, t["offset_n"], t["peak_db"])
    return x


def synth_click(rng: np.random.Generator) -> np.ndarray:
    """A 0.5 to 5 ms burst of exponentially decaying Gaussian noise with unit peak."""
    n = int(rng.integers(SAMPLE_RATE // 2000, SAMPLE_RATE // 200 + 1))
    click = rng.standard_normal(n) * np.exp(-5 * np.arange(n) / n)
    return (click / np.abs(click).max()).astype(np.float32)


def synth_dtmf(rng: np.random.Generator) -> np.ndarray:
    """One DTMF digit of 50 to 150 ms with 5 ms fades and unit peak."""
    n = int(rng.integers(SAMPLE_RATE // 20, SAMPLE_RATE * 3 // 20 + 1))
    t = np.arange(n) / SAMPLE_RATE
    low, high = _DTMF_ROWS[rng.integers(4)], _DTMF_COLUMNS[rng.integers(4)]
    tone = np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t)
    fade = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (SAMPLE_RATE // 200))
    tone *= fade
    return (tone / np.abs(tone).max()).astype(np.float32)


def padded_length(speech_s: float, target: float = 0.33, edge_s: float = 0.0) -> float:
    """Total length in seconds that makes `speech_s` about `target` of the sample.

    `edge_s` is non-speech already in the clip (natural clip edges) and counts
    towards the padding. A clip already at or below `target` is not shortened.
    """
    if speech_s <= 0 or edge_s < 0 or not 0 < target <= 1:
        raise ValueError(f"invalid padding arguments: {speech_s=}, {target=}, {edge_s=}")
    return max(speech_s / target, speech_s + edge_s)


# G(SNR) for SNR = -20 to 100 dB in 1 dB steps: log(mean|x|) - mean(log|x|) of Gamma(0.4) speech
# amplitudes plus Gaussian noise. These are the values of the public Monte Carlo table behind the
# WADA-SNR method of Kim and Stern, "Robust Signal-to-Noise Ratio Estimation Based on Waveform
# Amplitude Distribution Analysis", Interspeech 2008. Copied from lIkesimba9/audiogear
# (src/audiogear/pipeline/metrics/wada_snr.py, MIT licence, checked).
# The accumulate is needed because the Monte Carlo noise makes the third and fourth entries decrease,
# and np.interp requires a non-decreasing table.
# fmt: off
_WADA_G = np.maximum.accumulate(
    np.array(
        [
            0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627,
            0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192, 0.42077252, 0.42452799,
            0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
            0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724,
            0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418,
            0.95655449, 0.9835349, 1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995,
            1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029,
            1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727, 1.37345513, 1.38577122, 1.39733504,
            1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
            1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097, 1.528578,
            1.53389835, 1.5391211, 1.5439065, 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348,
            1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
            1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547,
            1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119,
            1.62390423, 1.62513143, 1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026,
            1.63204102,
        ]
    )
)
# fmt: on
_WADA_DB = np.arange(-20, 101, dtype=np.float64)
_EPS = 1e-10


def wada_snr(x: np.ndarray) -> float:
    """Blind SNR estimate in dB, clamped to [-20, 100] (WADA-SNR, Kim and Stern 2008).

    Input is 16-bit audio as decoded for storage, as float. The clip is
    peak-normalised and magnitudes are clipped to `_EPS`, as in the public
    reference, so exact zeros count as near-silence rather than being dropped:
    the value then does not depend on the float scale, but it does depend on
    the 16-bit quantisation, so never feed it a different decode form.
    Assumes speech-like amplitudes: for a pure tone the estimate is meaningless.
    """
    amplitude = np.abs(x, dtype=np.float64)
    peak = amplitude.max(initial=0.0)
    if peak == 0:
        raise ValueError("a silent signal has no SNR")
    amplitude = np.maximum(amplitude / peak, _EPS)
    g = np.log(amplitude.mean()) - np.log(amplitude).mean()
    return float(np.interp(g, _WADA_G, _WADA_DB))
