# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from as_is_support import LAUGHTER, make_ami_tree
from openvad_train.dataset.sources import ami


@pytest.fixture
def ami_tree(tmp_path):
    return make_ami_tree(tmp_path / "datasets/raw/ami")


def split_of(raw, meeting):
    (split,) = {c.split for c in ami.clips(raw) if c.clip_id.startswith(meeting)}
    return split


def one_timing(raw, meeting="ES2002a", channel=0):
    clip = next(c for c in ami.clips(raw) if c.clip_id == f"{meeting}.Headset-{channel}")
    (timing,) = ami.timings(raw, [clip])
    return timing


def overlaps_none(intervals, span):
    return all(end <= span[0] or start >= span[1] for start, end in intervals)


def test_ami_vocal_sounds_are_non_speech(ami_tree):
    t = one_timing(ami_tree)
    assert overlaps_none(t.speech_ms, LAUGHTER)


def test_only_spoken_words_are_speech(ami_tree):
    # Punctuation, vocal sounds, disfluency markers, an interval that runs backwards and a word without times are not.
    t = one_timing(ami_tree)
    assert (t.speech_ms, t.mask_ms, t.kind) == ([(1000, 1500), (4000, 4500)], [], "word")


def test_ami_splits_follow_partition(ami_tree):
    assert split_of(ami_tree, "ES2002a") == "train"
    assert split_of(ami_tree, "ES2011a") == "validation"
    assert split_of(ami_tree, "ES2004a") == "test"


def test_a_headset_the_fetch_listed_as_missing_gives_no_clip(ami_tree):
    ids = {c.clip_id for c in ami.clips(ami_tree)}
    assert "ES2002a.Headset-0" in ids
    assert "ES2002a.Headset-1" not in ids


def test_ami_clip_fields(ami_tree):
    c = next(c for c in ami.clips(ami_tree) if c.clip_id == "ES2011a.Headset-1")
    assert (c.source, c.language, c.speaker, c.licence, c.offset_s, c.duration_s, c.transcript) == (
        "ami",
        "en",
        "FEE004",
        "CC BY-NC-SA 2.0 (OpenSLR 16 copy; Edinburgh states CC BY 4.0)",
        0.0,
        6.0,
        None,
    )
    assert c.path == "datasets/raw/ami/audio/ES2011a/ES2011a.Headset-1.wav"


def test_each_headset_is_labelled_by_its_wearer_not_by_the_other_participants(ami_tree):
    first, second = one_timing(ami_tree, "ES2011a", 0), one_timing(ami_tree, "ES2011a", 1)
    assert first.speech_ms == [(1000, 1500), (4000, 4500)]
    assert second.speech_ms == [(11000, 11500), (14000, 14500)]


def test_ami_is_english_only_and_a_subset_keeps_the_splits(ami_tree):
    full = {c.clip_id: c.split for c in ami.clips(ami_tree)}
    assert {c.clip_id: c.split for c in ami.clips(ami_tree, {"en"})} == full
    assert list(ami.clips(ami_tree, {"de"})) == []


def test_ami_is_cut_at_gaps_and_not_forced_aligned():
    assert (ami.CUTTING, ami.MFA_ALIGNED) == ("split_over", False)
