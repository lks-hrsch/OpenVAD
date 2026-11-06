# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from itertools import pairwise

import pytest

from openvad_train.dataset.cutting import Chunk, cut_long, rebase
from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.schema import Timing


def timing(speech, mask=()) -> Timing:
    return Timing("c", "word", list(speech), list(mask), None, "t")


def words(n: int, length: int = 500, gap: int = 300, first: int = 1000) -> list[tuple[int, int]]:
    return [(first + i * (length + gap), first + i * (length + gap) + length) for i in range(n)]


def fifteen_second_timing() -> Timing:
    return timing(words(30))


def speech_in(chunk: Chunk, t: Timing) -> int:
    return sum(e - s for s, e in rebase(apply_gap_rule(t.speech_ms, 0), chunk.start_ms, chunk.end_ms))


def test_cut_long_keeps_speech_under_limit():
    t = fifteen_second_timing()
    chunks = cut_long(t)
    assert len(chunks) == 3
    assert all(speech_in(c, t) <= 6600 for c in chunks)


def test_cut_only_at_gaps_over_g():
    t = fifteen_second_timing()
    chunks = cut_long(t)
    for left, right in pairwise(chunks):
        before = max(e for _, e in t.speech_ms if e <= right.start_ms)
        after = min(s for s, _ in t.speech_ms if s >= left.end_ms)
        assert after - before > 200


def test_a_gap_of_exactly_g_is_split_only_by_the_budget_and_one_ms_more_is_a_pause_cut():
    # A budget split shares its midpoint; a cut at a pause leaves the rest of the gap out.
    inside_a_stretch = cut_long(timing(words(30, gap=200)))
    assert len(inside_a_stretch) == 3
    assert all(left.end_ms == right.start_ms for left, right in pairwise(inside_a_stretch))
    at_pauses = cut_long(timing(words(30, gap=201)))
    assert len(at_pauses) == 3
    assert all(left.end_ms + 1 == right.start_ms for left, right in pairwise(at_pauses))


def test_a_word_longer_than_the_limit_stays_one_chunk():
    assert cut_long(timing([(1000, 9000)])) == [Chunk(900, 9100)]


def test_a_run_over_the_limit_is_split_by_the_budget_and_its_neighbours_stay_apart():
    long_run = words(20, gap=100, first=10_000)
    t = timing([(8000, 8500), *long_run, (30_000, 30_500)])
    # 13 words of 500 ms fit the budget, and the 3500 ms left are over the 1 s floor.
    assert [speech_in(c, t) for c in cut_long(t)] == [500, 6500, 3500, 500]


def test_a_budget_cut_sits_at_the_midpoint_of_the_gap_without_a_margin():
    t = timing(words(14, gap=100, first=1000))
    # 14 words overflow by one, whose 500 ms are under the floor, so the cut moves to even the two pieces.
    assert cut_long(t) == [Chunk(900, 1000 + 7 * 600 - 50), Chunk(1000 + 7 * 600 - 50, 1000 + 13 * 600 + 500 + 100)]


def test_a_last_piece_of_a_second_or_more_keeps_the_greedy_cut():
    t = timing(words(15, gap=100))
    assert [speech_in(c, t) for c in cut_long(t)] == [6500, 1000]


def test_a_last_piece_under_a_second_is_evened_with_the_one_before():
    t = timing(words(14, gap=100))
    assert [speech_in(c, t) for c in cut_long(t)] == [3500, 3500]


def test_only_the_last_two_pieces_are_evened():
    t = timing(words(27, gap=100))
    assert [speech_in(c, t) for c in cut_long(t)] == [6500, 3500, 3500]


def test_words_touching_with_no_gap_are_split_between_them():
    t = timing(words(30, gap=0))
    chunks = cut_long(t)
    assert [speech_in(c, t) for c in chunks] == [6500, 6500, 2000]
    assert all(left.end_ms == right.start_ms for left, right in pairwise(chunks))
    # The cut lands on the boundary of two words, so neither is shortened.
    assert all(c.start_ms == 900 or any(s == c.start_ms for s, _ in t.speech_ms) for c in chunks)


@pytest.mark.parametrize("gap", [0, 1, 37, 100, 199, 200])
def test_a_budget_split_puts_every_word_whole_in_exactly_one_chunk(gap):
    t = timing(words(60, length=317, gap=gap))
    chunks = cut_long(t)
    assert all(left.end_ms <= right.start_ms for left, right in pairwise(chunks))
    for s, e in t.speech_ms:
        assert sum(c.start_ms <= s and e <= c.end_ms for c in chunks) == 1
    assert all(speech_in(c, t) <= 6600 for c in chunks)


def test_a_single_word_over_the_limit_inside_a_run_stays_whole():
    t = timing([(1000, 1500), (1600, 9600), (9700, 11_000)])
    assert [speech_in(c, t) for c in cut_long(t)] == [500, 8000, 1300]


def test_chunks_cover_every_word_whole_and_do_not_overlap():
    t = timing(words(100, length=333, gap=250, first=40))
    chunks = cut_long(t)
    assert sum(speech_in(c, t) for c in chunks) == 100 * 333
    assert chunks[0].start_ms == 0  # 40 ms before the first word: the margin stops at the clip start
    assert all(left.end_ms <= right.start_ms for left, right in pairwise(chunks))


def test_chunk_edges_keep_half_of_g_of_the_pause_around_the_speech():
    chunks = cut_long(fifteen_second_timing())
    assert chunks[0] == Chunk(900, 1000 + 12 * 800 + 500 + 100)
    assert chunks[1].start_ms == 1000 + 13 * 800 - 100


def test_a_long_pause_cuts_even_when_the_speech_budget_is_left():
    t = timing([(0, 1000), (26_000, 27_000)])
    chunks = cut_long(t)
    assert [speech_in(c, t) for c in chunks] == [1000, 1000]
    assert all(c.end_ms - c.start_ms <= 20_000 for c in chunks)


def test_a_pause_inside_the_span_limit_does_not_cut():
    assert len(cut_long(timing([(0, 1000), (10_000, 11_000)]))) == 1


def test_a_clip_without_speech_has_no_chunks():
    assert cut_long(timing([], [(0, 5000)])) == []


def test_unsorted_and_overlapping_words_count_once():
    t = timing([(5000, 5500), (1000, 2000), (1500, 2500)])
    chunks = cut_long(t)
    assert chunks == [Chunk(900, 5600)]
    assert speech_in(chunks[0], t) == 2000


def test_rebase_cuts_at_the_bounds_and_counts_from_the_start():
    assert rebase([(0, 100), (150, 250), (400, 500)], 100, 200) == [(50, 100)]
    assert rebase([(90, 110)], 100, 200) == [(0, 10)]
