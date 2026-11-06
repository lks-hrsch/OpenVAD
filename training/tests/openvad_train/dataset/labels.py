# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from openvad_train.dataset.labels import apply_gap_rule, rasterise, speech_ratio


def test_gap_shorter_than_g_is_speech():
    assert apply_gap_rule([(100, 300), (450, 600)], 200) == [(100, 600)]


def test_gap_of_exactly_g_is_non_speech():
    assert apply_gap_rule([(100, 300), (500, 600)], 200) == [(100, 300), (500, 600)]


def test_gap_rule_sorts_and_merges_overlaps():
    assert apply_gap_rule([(500, 600), (100, 400), (300, 450)], 0) == [(100, 450), (500, 600)]


def test_gap_rule_drops_empty_intervals_and_accepts_empty_input():
    assert apply_gap_rule([(100, 100), (300, 200)], 200) == []
    assert apply_gap_rule([], 200) == []


def test_rasterise_grid():
    y = rasterise([(100, 300)], [], 500)
    assert y.dtype == np.uint8
    assert len(y) == 50
    assert y[9] == 0
    assert y[10] == 1
    assert y[29] == 1
    assert y[30] == 0


def test_mask_wins_and_any_overlap_masks():
    y = rasterise([(0, 500)], [(25, 26)], 500)
    assert y[2] == 2
    assert y[1] == 1


def test_tie_is_non_speech():
    assert rasterise([(5, 10)], [], 20)[0] == 0
    assert rasterise([(4, 10)], [], 20)[0] == 1


def test_shift_moves_labels():
    assert rasterise([(100, 200)], [], 500, shift_ms=30)[12] == 0
    assert rasterise([(100, 200)], [], 500, shift_ms=30)[13] == 1


def test_shift_moves_masks_too():
    y = rasterise([], [(100, 110)], 500, shift_ms=30)
    assert y[12] == 0
    assert y[13] == 2
    assert y[14] == 0


def test_empty_input_is_all_non_speech():
    y = rasterise([], [], 500)
    assert y.dtype == np.uint8
    assert len(y) == 50
    assert not y.any()


def test_unsorted_and_overlapping_intervals_are_a_union():
    # Overlap must not double count: (0, 6) then (3, 9) covers 9 ms of frame 0.
    assert rasterise([(3, 9), (0, 6)], [], 20)[0] == 1
    # Two 3 ms intervals overlapping fully still cover only 3 ms: a tie, non-speech.
    assert rasterise([(0, 3), (0, 3)], [], 20)[0] == 0


def test_intervals_are_clipped_not_rejected():
    y = rasterise([(-100, 20), (480, 900)], [(-50, -10), (700, 800)], 500)
    assert len(y) == 50
    assert list(y[:3]) == [1, 1, 0]
    assert list(y[47:]) == [0, 1, 1]
    assert 2 not in y


def test_shift_pushes_interval_out_of_range():
    assert not rasterise([(0, 100)], [], 500, shift_ms=-200).any()
    assert not rasterise([(0, 100)], [], 500, shift_ms=900).any()


def test_length_is_rounded_for_non_multiples_of_ten():
    assert len(rasterise([], [], 504)) == 50
    assert len(rasterise([], [], 496)) == 50
    assert len(rasterise([], [], 0)) == 0


def test_interval_past_duration_does_not_cover_the_rounded_up_tail():
    # 496 ms rounds up to 50 frames; ms 496..500 is not audio and must not count.
    assert rasterise([(496, 600)], [], 496)[49] == 0
    assert rasterise([(490, 600)], [], 496)[49] == 1


def test_speech_ratio_uses_the_union():
    assert speech_ratio([(0, 100), (50, 150), (400, 500)], 1000) == pytest.approx(0.25)


def test_speech_ratio_clips_and_handles_empty():
    assert speech_ratio([(-50, 50), (950, 2000)], 1000) == pytest.approx(0.1)
    assert speech_ratio([], 1000) == 0.0
    assert speech_ratio([], 0) == 0.0
