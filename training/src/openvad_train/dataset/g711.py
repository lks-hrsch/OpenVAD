# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""G.711 A-law and mu-law round trips as numpy lookup tables (ADR 0021).

The quantiser is ffmpeg's `pcm_alaw` and `pcm_mulaw` bit for bit, so the training data carries the same
quantisation noise as a real gateway built on it: the encoder indexes a table by `(sample + 32768) >> 2`, rounds the
magnitude of a positive sample down and that of a negative sample up, and puts the decision thresholds at
`(m[k] + m[k + 1] + 4) >> 3` between the decoded magnitudes `m`. It is therefore not a nearest-level search, which
differs from it at about 500 of the 65536 int16 values.
"""

import numpy as np


_SCALE = 32768


def _magnitudes(law: str) -> np.ndarray:
    """The 128 decoded magnitudes of a law, ascending (ITU-T G.711 expansion)."""
    i = np.arange(128)
    seg = i >> 4
    if law == "alaw":
        return np.where(seg == 0, ((i & 15) << 4) + 8, (((i & 15) << 4) + 0x108) << np.maximum(seg - 1, 0))
    return ((((i & 15) << 3) + 0x84) << seg) - 0x84


def _roundtrip_table(law: str) -> np.ndarray:
    """The decoded int16 value of every int16 input, indexed by `input + 32768`."""
    magnitude = _magnitudes(law)
    threshold = (magnitude[:-1] + magnitude[1:] + 4) >> 3
    v = np.arange(-_SCALE, _SCALE)
    level = magnitude[np.searchsorted(threshold, np.where(v >= 0, v >> 2, -(v >> 2)), side="right")]
    return np.where(v >= 0, level, -level).astype(np.int16)


_TABLES = {"alaw": _roundtrip_table("alaw"), "mulaw": _roundtrip_table("mulaw")}


def _roundtrip(x8k: np.ndarray, law: str) -> np.ndarray:
    # Clipping first keeps the index inside the table when a filter overshoots full scale. The upper bound is the
    # largest int16, since +1.0 * 32768 is not one. The scale is a power of two, so the product is exact, and
    # `rint` (halves to even) decides the bucket of a value between two int16 steps.
    pcm = np.rint(np.clip(x8k, -1.0, 32767 / _SCALE) * _SCALE).astype(np.int32)
    return (_TABLES[law][pcm + _SCALE] / np.float32(_SCALE)).astype(np.float32)


def alaw_roundtrip(x8k: np.ndarray) -> np.ndarray:
    """A-law encode and decode of float audio in [-1, 1] (values beyond are clipped), as a new float32 array.

    Sample for sample: no delay and no length change. The input is not modified.
    """
    return _roundtrip(x8k, "alaw")


def mulaw_roundtrip(x8k: np.ndarray) -> np.ndarray:
    """mu-law encode and decode, as `alaw_roundtrip`."""
    return _roundtrip(x8k, "mulaw")
