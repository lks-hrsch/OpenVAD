# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Cut clips too long to be one sample into chunks at pauses. Everything is integer milliseconds."""

from itertools import pairwise
from typing import NamedTuple

from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.schema import Timing


_MIN_LAST_PIECE_MS = 1000


class Chunk(NamedTuple):
    start_ms: int
    end_ms: int


def rebase(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """The parts of the intervals inside [start, end), counted from `start`; a word crossing a boundary is cut."""
    return [(max(s, start) - start, min(e, end) - start) for s, e in intervals if min(e, end) > max(s, start)]


def _speech_ms(words: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in words)


def _pack_words(words: list[tuple[int, int]], max_speech_ms: int) -> list[list[tuple[int, int]]]:
    """Words in order, a new piece where the next word would overflow the budget. A last piece under
    `_MIN_LAST_PIECE_MS` is evened with the one before it, so no sample is left with a scrap of speech.
    """
    pieces: list[list[tuple[int, int]]] = [[]]
    for word in words:
        if pieces[-1] and _speech_ms(pieces[-1]) + word[1] - word[0] > max_speech_ms:
            pieces.append([])
        pieces[-1].append(word)
    if len(pieces) > 1 and _speech_ms(pieces[-1]) < _MIN_LAST_PIECE_MS:
        joined = pieces[-2] + pieces[-1]
        total = _speech_ms(joined)
        cut = min(range(1, len(joined)), key=lambda i: abs(2 * _speech_ms(joined[:i]) - total))
        pieces[-2:] = [joined[:cut], joined[cut:]]
    return pieces


def cut_long(timing: Timing, max_speech_s: float = 6.6, gap_ms: int = 200, max_span_s: float = 20.0) -> list[Chunk]:
    """Chunks of at most `max_speech_s` of speech (word union, G = 0) and `max_span_s` in all. Only a single word
    over the speech limit exceeds it.

    Chunks are first cut at word gaps longer than `gap_ms`, keeping `gap_ms // 2` of audio around their first and
    last word, which such a gap always has room for. The rest of the gap, and the audio before the first and after
    the last word, belongs to no chunk. The caller clips the last chunk to the clip's end.
    A stretch still over the speech limit has no such gap, so it is split between two words where the budget would
    overflow. Both pieces end at that gap's midpoint with no margin, so no word is in two chunks and chunks never
    overlap. Such a cut can sit on a label edge, which is accepted: noise or dither covers the whole sample.
    `max_span_s` is there because a real SWC article has a gap of over 1000 s with no unaligned token in it.
    """
    runs: list[list[tuple[int, int]]] = []  # the words of each stretch between gaps longer than `gap_ms`
    for word in apply_gap_rule(timing.speech_ms, 0):
        if runs and word[0] - runs[-1][-1][1] <= gap_ms:
            runs[-1].append(word)
        else:
            runs.append([word])
    if not runs:
        return []
    max_speech_ms, max_span_ms = round(max_speech_s * 1000), round(max_span_s * 1000)
    margin = gap_ms // 2
    groups = [runs[0]]
    for run in runs[1:]:
        over_speech = _speech_ms(groups[-1]) + _speech_ms(run) > max_speech_ms
        if over_speech or run[-1][1] - groups[-1][0][0] + 2 * margin > max_span_ms:
            groups.append(run)
        else:
            groups[-1] = groups[-1] + run
    chunks: list[Chunk] = []
    for group in groups:
        cuts = [(left[-1][1] + right[0][0]) // 2 for left, right in pairwise(_pack_words(group, max_speech_ms))]
        starts, ends = [max(0, group[0][0] - margin), *cuts], [*cuts, group[-1][1] + margin]
        chunks += [Chunk(s, e) for s, e in zip(starts, ends, strict=True)]
    return chunks


def split_over(timing: Timing, max_s: float = 30.0) -> list[Chunk]:
    """Chunks of at most `max_s` that tile the timeline from 0 to the end of the last word, for sources that are
    kept as they are (no padding, so no audio is left out). Each cut is the midpoint of a gap between two words.

    A cut goes to the longest gap in the second half of the `max_s` after a chunk's start, or in the first half if
    the second has none. If no gap lies within `max_s`, the chunk runs on to the next gap, and with no gap at all
    it stays whole. The first chunk starts at 0 and the last ends at the last word; the caller extends it to the
    clip's end. A timing without speech has no chunks.
    """
    words = apply_gap_rule(timing.speech_ms, 0)
    if not words:
        return []
    max_ms = round(max_s * 1000)
    gaps = [((right[0] - left[1]), (left[1] + right[0]) // 2) for left, right in pairwise(words)]
    chunks: list[Chunk] = []
    start = 0
    while words[-1][1] - start > max_ms:
        within = [(length, cut) for length, cut in gaps if start < cut <= start + max_ms]
        late = [g for g in within if g[1] > start + max_ms // 2] or within
        after = [cut for _, cut in gaps if cut > start + max_ms]
        if not (late or after):
            break
        cut = max(late)[1] if late else after[0]
        chunks.append(Chunk(start, cut))
        start = cut
    return [*chunks, Chunk(start, words[-1][1])]
