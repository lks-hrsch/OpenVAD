# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Speech intervals in milliseconds to one label per 10 ms frame.

Labels: 0 non-speech, 1 speech, 2 masked. Intervals are half-open, may be
unsorted or overlapping, and are clipped to the clip, never rejected.
"""

import numpy as np


FRAME_MS = 10


def apply_gap_rule(words_ms: list[tuple[int, int]], gap_ms: int) -> list[tuple[int, int]]:
    """Merge intervals whose gap is shorter than `gap_ms`; a gap of exactly `gap_ms` stays open."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(w for w in words_ms if w[1] > w[0]):
        if merged and start - merged[-1][1] < gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _covered(intervals: list[tuple[int, int]], shift_ms: int, n_ms: int, limit_ms: int) -> np.ndarray:
    """Per-millisecond union of the shifted intervals, clipped to [0, limit_ms], in `n_ms` slots."""
    if not intervals:
        return np.zeros(n_ms, dtype=bool)
    bounds = np.clip(np.asarray(intervals, dtype=np.int64) + shift_ms, 0, limit_ms)
    bounds = bounds[bounds[:, 1] > bounds[:, 0]]
    diff = np.zeros(n_ms + 1, dtype=np.int64)
    np.add.at(diff, bounds[:, 0], 1)
    np.add.at(diff, bounds[:, 1], -1)
    return np.cumsum(diff[:n_ms]) > 0


def rasterise(
    speech_ms: list[tuple[int, int]],
    mask_ms: list[tuple[int, int]],
    duration_ms: int,
    shift_ms: int = 0,
) -> np.ndarray:
    """One uint8 label per 10 ms frame, `round(duration_ms / 10)` frames.

    A frame is masked if any mask covers any of it, else speech only if speech
    covers strictly more than half of it (a tie is non-speech). `shift_ms`
    moves speech and masks together before clipping to the clip.
    """
    n_frames = round(duration_ms / FRAME_MS)
    n_ms = n_frames * FRAME_MS
    limit_ms = min(duration_ms, n_ms)
    speech = _covered(speech_ms, shift_ms, n_ms, limit_ms).reshape(n_frames, FRAME_MS)
    masked = _covered(mask_ms, shift_ms, n_ms, limit_ms).reshape(n_frames, FRAME_MS)
    labels = (speech.sum(axis=1) * 2 > FRAME_MS).astype(np.uint8)
    labels[masked.any(axis=1)] = 2
    return labels


def speech_ratio(words_ms: list[tuple[int, int]], duration_ms: int) -> float:
    """Share of the clip covered by the word union (G = 0)."""
    if duration_ms <= 0:
        return 0.0
    return float(_covered(words_ms, 0, duration_ms, duration_ms).sum() / duration_ms)
