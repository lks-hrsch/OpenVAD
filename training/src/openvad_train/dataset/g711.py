# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""G.711 A-law and mu-law round trips as numpy lookup tables (ADR 0021).

The encoder follows the G.711 rules, written from the standard's published description:

- A-law works on the 13-bit value `v >> 3`, mu-law on the 14-bit value `v >> 2` (floor in both).
- A negative value is coded as the magnitude of its one's complement, so `f(-v - 1) == -f(v)` at every int16 value.
- A-law: eight segments of sixteen steps, the first two with the same step. Mu-law: magnitude clipped to 8158, bias
  33, eight segments of sixteen steps.
- The decoder returns the middle of the interval.

ffmpeg's `pcm_alaw` and `pcm_mulaw` encoders are not G.711: they index their table by `(v + 32768) >> 2`, which
shifts the negative bins 4 LSB toward zero, and they put segment edges at the midpoint of the decoded levels. They
differ from this table at 964 of the 65536 int16 values on each law (for example A-law -13 to -16 gives -24 there
and -8 here, and mu-law -1 to -4 gives -8 there and 0 here). Their decoders agree with this one.
"""

import numpy as np


_SCALE = 32768
_A_SEGMENT_EDGES = 32 * 2 ** np.arange(7)  # 13-bit magnitudes where segments 1 to 7 start
_MU_SEGMENT_EDGES = 64 * 2 ** np.arange(7)  # the same on the biased 14-bit magnitude


def _magnitudes(law: str) -> np.ndarray:
    """The 128 decoded magnitudes of a law at 16 bits, ascending, indexed by `segment * 16 + step`."""
    i = np.arange(128)
    seg = i >> 4
    if law == "alaw":
        return np.where(seg == 0, ((i & 15) << 4) + 8, (((i & 15) << 4) + 0x108) << np.maximum(seg - 1, 0))
    return ((((i & 15) << 3) + 0x84) << seg) - 0x84


def _code_index(v: np.ndarray, law: str) -> np.ndarray:
    """The `segment * 16 + step` of the magnitude code of each int16 value `v`."""
    if law == "alaw":
        s = v >> 3
        m = np.where(s >= 0, s, ~s)
        seg = np.searchsorted(_A_SEGMENT_EDGES, m, side="right")
        return seg * 16 + np.where(seg == 0, m >> 1, (m >> seg) & 15)
    s = v >> 2
    biased = np.minimum(np.where(s >= 0, s, ~s), 8158) + 33
    seg = np.searchsorted(_MU_SEGMENT_EDGES, biased, side="right")
    return seg * 16 + ((biased >> (seg + 1)) & 15)


def _roundtrip_table(law: str) -> np.ndarray:
    """The decoded int16 value of every int16 input, indexed by `input + 32768`."""
    v = np.arange(-_SCALE, _SCALE)
    level = _magnitudes(law)[_code_index(v, law)]
    return np.where(v >= 0, level, -level).astype(np.int16)


_TABLES = {"alaw": _roundtrip_table("alaw"), "mulaw": _roundtrip_table("mulaw")}


def require_finite(x: np.ndarray) -> None:
    """Raise `ValueError` naming the count if `x` holds NaN or infinity, which no int16 conversion defines."""
    bad = x.size - np.count_nonzero(np.isfinite(x))
    if bad:
        raise ValueError(f"audio holds {bad} non-finite samples")


def _roundtrip(x8k: np.ndarray, law: str) -> np.ndarray:
    require_finite(x8k)
    # Clipping first keeps the index inside the table when a filter overshoots full scale. The upper bound is the
    # largest int16, since +1.0 * 32768 is not one. The scale is a power of two, so the product is exact, and
    # `rint` (halves to even) decides the bucket of a value between two int16 steps.
    pcm = np.rint(np.clip(x8k, -1.0, 32767 / _SCALE) * _SCALE).astype(np.int32)
    return (_TABLES[law][pcm + _SCALE] / np.float32(_SCALE)).astype(np.float32)


def alaw_roundtrip(x8k: np.ndarray) -> np.ndarray:
    """A-law encode and decode of float audio in [-1, 1] (finite values beyond are clipped), as a new float32 array.

    Sample for sample: no delay and no length change. The input is not modified. NaN or infinity raises
    `ValueError`.
    """
    return _roundtrip(x8k, "alaw")


def mulaw_roundtrip(x8k: np.ndarray) -> np.ndarray:
    """mu-law encode and decode, as `alaw_roundtrip`."""
    return _roundtrip(x8k, "mulaw")
