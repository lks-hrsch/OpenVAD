# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from mfa_source_support import make_sc_tree
from openvad_train.dataset.sources import speechcommands


@pytest.fixture
def sc_tree(tmp_path):
    return make_sc_tree(tmp_path)


def test_sc_test_is_testing_list_not_separate_set(sc_tree):
    clips = list(speechcommands.clips(sc_tree))
    assert not any("speech_commands_test_set" in c.path for c in clips)
    assert {c.clip_id for c in clips if c.split == "test"} == {"no/cccccccc_nohash_0", "no/dddddddd_nohash_0"}
    assert {c.clip_id for c in clips if c.split == "validation"} == {"yes/bbbbbbbb_nohash_0"}
    assert {c.clip_id for c in clips if c.split == "train"} == {
        "yes/aaaaaaaa_nohash_0",
        "no/aaaaaaaa_nohash_1",
        "_background_noise_/running_tap",
    }


def test_sc_clip_fields_and_header_duration(sc_tree):
    c = next(c for c in speechcommands.clips(sc_tree) if c.clip_id == "no/cccccccc_nohash_0")
    assert (c.source, c.language, c.speaker, c.transcript, c.licence, c.offset_s, c.duration_s) == (
        "speech-commands",
        "en",
        "cccccccc",
        "no",
        "CC BY 4.0",
        0.0,
        0.75,
    )
    assert c.path == "datasets/raw/speech-commands/speech_commands_v0.02/no/cccccccc_nohash_0.wav"


def test_sc_background_noise_has_no_speaker_and_no_transcript(sc_tree):
    noise = [c for c in speechcommands.clips(sc_tree) if "_background_noise_" in c.clip_id]
    assert [(c.speaker, c.transcript, c.duration_s) for c in noise] == [(None, None, 6.0)]


def test_sc_background_noise_is_non_speech(sc_tree):
    noise = [c for c in speechcommands.clips(sc_tree) if c.transcript is None]
    (t,) = speechcommands.timings(sc_tree, noise)
    assert "_background_noise_" in t.clip_id
    assert (t.speech_ms, t.mask_ms) == ([], [])


def test_sc_refuses_to_label_a_spoken_clip(sc_tree):
    spoken = next(c for c in speechcommands.clips(sc_tree) if c.transcript is not None)
    with pytest.raises(ValueError, match="labelled by MFA"):
        list(speechcommands.timings(sc_tree, [spoken]))


def test_sc_is_english_only(sc_tree):
    assert list(speechcommands.clips(sc_tree, {"de"})) == []
    assert list(speechcommands.clips(sc_tree, {"en"}))


def test_sc_is_an_mfa_source():
    assert speechcommands.MFA_ALIGNED is True
