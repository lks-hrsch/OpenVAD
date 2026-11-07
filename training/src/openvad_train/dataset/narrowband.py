# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Train-time narrowband: decimate, G.711, interpolate (ADR 0021; spec, train-time narrowband).

Delay. The resampler is linear-phase, so every function that filters returns audio that lags its input by
`label_shift_samples()` samples (66 at 16 kHz, 4.125 ms, 0.4125 frames), and nothing here compensates. The shift
rounds to 0 frames on the 10 ms grid, so the labels stay as they are. A `narrow_leg` sample is delayed twice: its
stored AMR-NB leg carries 5.0 ms for the codec plus 4.125 ms for the resampler, which Task 19 rounds to 9 ms in the
labels and leaves 0.125 ms; this transform adds another 4.125 ms. The residual lag is 4.25 ms, 0.425 frames,
under one frame.

The resampler functions are whole-signal calls; pass the whole sample, never blocks.
"""

from collections.abc import Callable

import numpy as np

from openvad_rs import GROUP_DELAY_16K, decimate_2x, interpolate_2x
from openvad_train.dataset.g711 import alaw_roundtrip, mulaw_roundtrip, require_finite
from openvad_train.dataset.schema import _blake2_64


_FRAME_SAMPLES = 160  # 10 ms at 16 kHz
_P_ALAW = 0.9  # ADR 0021, a placeholder until the gateway log query has run
_BANDS = ("wide", "narrow_leg", "narrow_native")


def eval_seed(sample_id: str) -> int:
    """Seed of the narrowband draws of validation and test, so scoring repeats; own domain, apart from `rank`."""
    return _blake2_64("narrowband-eval", sample_id)


def label_shift_samples(one_way: int = GROUP_DELAY_16K) -> int:
    """Delay of the decimate and interpolate round trip in 16 kHz samples; `one_way` is each filter's delay."""
    return 2 * one_way


def label_shift_frames(one_way: int = GROUP_DELAY_16K) -> int:
    """`label_shift_samples` on the 10 ms grid, halves rounded up. 0 for the current filter."""
    return (label_shift_samples(one_way) + _FRAME_SAMPLES // 2) // _FRAME_SAMPLES


def _through(audio: np.ndarray, law: Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    require_finite(audio)  # the filter would spread one infinity over 133 samples and the int16 cast is undefined
    # An odd length gains one sample in the round trip; the cut restores the input's length.
    narrow = decimate_2x(np.ascontiguousarray(audio, np.float32))
    return interpolate_2x(law(narrow) if law else narrow)[: len(audio)]


def resample_only(audio: np.ndarray) -> np.ndarray:
    """16 to 8 to 16 kHz with the OpenVAD resampler and no G.711: the 8 kHz scoring condition.

    A new float32 array of the input's length, which is not modified. It lags the input by `label_shift_samples()`.
    """
    return _through(audio, None)


def _telephone(audio: np.ndarray, u: float) -> np.ndarray:
    return _through(audio, alaw_roundtrip if u < _P_ALAW else mulaw_roundtrip)


def telephone(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Decimate, G.711, interpolate: A-law with probability 0.9, else mu-law, as `resample_only` otherwise.

    The law takes exactly one draw from `rng`.
    """
    return _telephone(audio, rng.random())


def apply_narrowband(audio: np.ndarray, band: str, rng: np.random.Generator, p_narrow: float = 0.35) -> np.ndarray:
    """The `band` column's train-time step: `wide` takes `telephone` with probability `p_narrow`, `narrow_leg`
    always, `narrow_native` never. An unknown `band` raises `ValueError`.

    Every call takes exactly two draws from `rng`, whatever the band, so a sample's later draws do not depend on
    this decision. Audio left alone is returned as the same object; processed audio is a new float32 array that
    lags the input as `telephone` does.
    """
    if band not in _BANDS:
        raise ValueError(f"unknown band {band!r}, expected one of {_BANDS}")
    if not 0.0 <= p_narrow <= 1.0:
        raise ValueError(f"p_narrow must be in [0, 1], got {p_narrow}")
    decision, law = rng.random(2)
    if band == "narrow_native" or (band == "wide" and decision >= p_narrow):
        return audio
    return _telephone(audio, law)
