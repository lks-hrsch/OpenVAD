# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses
from collections import Counter

import numpy as np
import pytest

from as_is_support import CALL_SECONDS, make_callhome_tree, noise
from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.sources import callhome


@pytest.fixture
def callhome_tree(tmp_path):
    return make_callhome_tree(tmp_path / "datasets/raw/callhome")


def test_callhome_masks_every_turn_boundary():
    t = callhome.timing_from_turns([("A", 1000, 2000), ("B", 1500, 3000)])
    assert (750, 1250) in t.mask_ms
    assert (1250, 1750) in t.mask_ms
    assert (1750, 2250) in t.mask_ms
    assert (2750, 3250) in t.mask_ms
    assert t.speech_ms == [(1000, 3000)]
    assert t.kind == "turn"


def test_a_turn_that_does_not_run_forward_has_no_speech_and_no_boundary():
    t = callhome.timing_from_turns([("A", 1000, 1000), ("B", 5000, 4000)])
    assert (t.speech_ms, t.mask_ms) == ([], [])


def test_a_mask_does_not_start_before_the_call():
    assert callhome.timing_from_turns([("A", 100, 900)]).mask_ms[0] == (0, 350)


def test_callhome_split_by_call(callhome_tree):
    clips = list(callhome.clips(callhome_tree))
    assert len(clips) == len({c.clip_id for c in clips}) == 14
    # 80/10/10 of 14 calls, rounded: 11, 2 and 1.
    assert Counter(c.split for c in clips) == {"train": 11, "validation": 2, "test": 1}


def test_a_slice_and_the_full_build_give_every_call_the_same_split(callhome_tree):
    full = {c.clip_id: c.split for c in callhome.clips(callhome_tree)}
    german = {c.clip_id: c.split for c in callhome.clips(callhome_tree, {"de"})}
    assert german
    assert german.items() <= full.items()
    assert {c.language for c in callhome.clips(callhome_tree, {"de"})} == {"de"}


def test_callhome_clip_fields_and_the_row_its_audio_is_in(callhome_tree):
    c = next(c for c in callhome.clips(callhome_tree) if c.clip_id == "eng/data-00001-of-00003/2")
    assert (c.source, c.language, c.speaker, c.licence, c.offset_s, c.duration_s, c.row_in_file) == (
        "callhome",
        "en",
        None,
        "CC BY-NC-SA 4.0",
        0.0,
        CALL_SECONDS,
        2,
    )
    assert c.path == "datasets/raw/callhome/eng/data-00001-of-00003.parquet"
    assert {c.language for c in callhome.clips(callhome_tree)} == {"en", "de", "zh"}


def test_timings_come_from_the_turns_of_the_calls_own_row(callhome_tree):
    clips = list(callhome.clips(callhome_tree))
    timings = list(callhome.timings(callhome_tree, clips))
    assert [t.clip_id for t in timings] == [c.clip_id for c in clips]
    assert timings[0].speech_ms == [(0, 2000)]
    assert timings[0].kind == "turn"
    assert timings[0].provenance == "callhome:turns"


def test_the_row_a_clip_names_holds_that_calls_audio(callhome_tree):
    # Audio was seeded by position: the 5th call written is the first of the second English shard.
    c = next(c for c in callhome.clips(callhome_tree) if c.clip_id == "eng/data-00001-of-00003/0")
    expected = noise(CALL_SECONDS, 5)
    row = {"audio_filepath": c.path, "audio_format": "callhome_parquet", "row_in_file": c.row_in_file}
    audio = load_audio(row | {"offset": 0.5, "duration": 1.0}, callhome_tree.parents[2])
    assert audio.shape == (16000,)
    assert np.allclose(audio, expected[8000:24000], atol=1e-4)  # the WAV holds 16-bit samples


def test_a_clip_without_a_row_is_a_value_error(callhome_tree):
    clip = dataclasses.replace(next(iter(callhome.clips(callhome_tree))), row_in_file=None)
    with pytest.raises(ValueError, match="names the row of its call"):
        list(callhome.timings(callhome_tree, [clip]))
