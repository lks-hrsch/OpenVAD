# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Cut clips too long to be one sample into chunks at pauses. Everything is integer milliseconds."""

from typing import NamedTuple

from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.schema import Timing


class Chunk(NamedTuple):
    start_ms: int
    end_ms: int


def rebase(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """The parts of the intervals inside [start, end), counted from `start`; a word crossing a boundary is cut."""
    return [(max(s, start) - start, min(e, end) - start) for s, e in intervals if min(e, end) > max(s, start)]


def cut_long(timing: Timing, max_speech_s: float = 6.6, gap_ms: int = 200, max_span_s: float = 20.0) -> list[Chunk]:
    """Chunks of at most `max_speech_s` of speech (word union, G = 0) and `max_span_s` in all, cut only at word
    gaps longer than `gap_ms`. A stretch without such a gap stays whole, even over a limit.

    A chunk keeps `gap_ms // 2` of audio around its first and last word, which a cuttable gap always has room for:
    no word tail is lost, and the cut does not coincide with a label edge. The rest of the gap, and the audio before
    the first and after the last word, belongs to no chunk. The caller clips the last chunk to the clip's end.
    `max_span_s` is there because a real SWC article has a gap of over 1000 s with no unaligned token in it.
    """
    runs: list[tuple[int, int, int]] = []  # start, end and speech of the stretches between cuttable gaps
    for s, e in apply_gap_rule(timing.speech_ms, 0):
        if runs and s - runs[-1][1] <= gap_ms:
            runs[-1] = (runs[-1][0], e, runs[-1][2] + e - s)
        else:
            runs.append((s, e, e - s))
    if not runs:
        return []
    margin = gap_ms // 2
    chunks: list[Chunk] = []
    start, end, speech = runs[0]
    for s, e, run_speech in runs[1:]:
        if speech + run_speech > round(max_speech_s * 1000) or e - start + 2 * margin > round(max_span_s * 1000):
            chunks.append(Chunk(max(0, start - margin), end + margin))
            start, speech = s, 0
        end = e
        speech += run_speech
    chunks.append(Chunk(max(0, start - margin), end + margin))
    return chunks
