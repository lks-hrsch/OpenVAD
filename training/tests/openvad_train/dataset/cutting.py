# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from itertools import pairwise

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


def test_a_gap_of_exactly_g_is_not_a_cut_but_one_ms_more_is():
    assert len(cut_long(timing(words(30, gap=200)))) == 1
    assert len(cut_long(timing(words(30, gap=201)))) == 3


def test_a_word_longer_than_the_limit_stays_one_chunk():
    assert cut_long(timing([(1000, 9000)])) == [Chunk(900, 9100)]


def test_a_run_without_a_cuttable_gap_stays_whole_and_its_neighbours_are_split_off():
    long_run = words(20, gap=100, first=10_000)
    t = timing([(8000, 8500), *long_run, (30_000, 30_500)])
    assert [speech_in(c, t) for c in cut_long(t)] == [500, 10_000, 500]


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
