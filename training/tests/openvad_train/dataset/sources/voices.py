# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict

import numpy as np
import pytest
import soundfile as sf

from as_is_support import (
    DEV_CLEAN_SPEAKERS,
    TRAIN_100_SPEAKERS,
    TRAIN_360_SPEAKERS,
    VOICES_DELAY_SAMPLES,
    VOICES_WORDS_MS,
    make_voices_tree,
    noise,
)
from openvad_train.dataset.sources import voices


@pytest.fixture
def voices_tree(tmp_path):
    return make_voices_tree(tmp_path / "datasets/raw/voices")


def test_voices_speaker_rules(voices_tree):
    clips = list(voices.clips(voices_tree))
    assert not any(c.speaker in DEV_CLEAN_SPEAKERS for c in clips)
    assert {c.split for c in clips if c.speaker in TRAIN_100_SPEAKERS} == {"train"}
    assert {c.speaker for c in clips} == TRAIN_100_SPEAKERS | TRAIN_360_SPEAKERS


def test_the_train_clean_360_speakers_are_split_by_speaker_into_all_three_splits(voices_tree):
    by = defaultdict(set)
    for c in voices.clips(voices_tree):
        by[c.speaker].add(c.split)
    assert all(len(splits) == 1 for splits in by.values())  # also the speaker re-recorded in the devkit's test folder
    assert {next(iter(by[s])) for s in TRAIN_360_SPEAKERS} == {"train", "validation", "test"}


def test_voices_distractors_are_not_used(voices_tree):
    assert not any("distractor" in c.path for c in voices.clips(voices_tree))
    assert len(list(voices.clips(voices_tree))) == len(TRAIN_100_SPEAKERS) + len(TRAIN_360_SPEAKERS) + 1


def test_voices_clip_fields_and_the_whole_recording(voices_tree):
    c = next(c for c in voices.clips(voices_tree) if c.speaker == "0083")
    assert (c.source, c.language, c.licence, c.offset_s, c.duration_s, c.transcript) == (
        "voices",
        "en",
        "CC BY 4.0",
        0.0,
        3.0,
        None,
    )
    assert c.path.startswith("datasets/raw/voices/VOiCES_devkit/distant-16k/speech/train/rm1/musi/sp0083/")


def test_a_slice_and_the_full_build_give_every_clip_the_same_split(voices_tree):
    full = {c.clip_id: c.split for c in voices.clips(voices_tree)}
    assert {c.clip_id: c.split for c in voices.clips(voices_tree, {"en"})} == full
    assert list(voices.clips(voices_tree, {"de"})) == []


def test_a_speaker_from_a_set_without_a_rule_is_refused(voices_tree):
    table = voices_tree / "VOiCES_devkit/references/Lab41-SRI-VOiCES-speaker-gender-dataset.tbl"
    table.write_text(table.read_text().replace("0083 F train-clean-100", "0083 F test-clean"))
    with pytest.raises(ValueError, match="no rule"):
        list(voices.clips(voices_tree))


def test_the_delay_is_the_lag_of_the_recording_behind_its_source(tmp_path):
    source = noise(2.0, seed=1)
    sf.write(tmp_path / "source.wav", source, 16000, subtype="PCM_16")
    for lag in (0, VOICES_DELAY_SAMPLES, 1000, -300):
        delayed = np.roll(source, lag) * 0.3 + noise(2.0, seed=2) * 0.05
        sf.write(tmp_path / "delayed.wav", delayed, 16000, subtype="PCM_16")
        assert voices.measure_voices_delay(tmp_path / "delayed.wav", tmp_path / "source.wav") == lag / 16000


def test_a_lag_beyond_the_window_is_not_found(tmp_path):
    source = noise(2.0, seed=1)
    sf.write(tmp_path / "source.wav", source, 16000, subtype="PCM_16")
    sf.write(tmp_path / "delayed.wav", np.roll(source, 8000), 16000, subtype="PCM_16")  # 0.5 s
    assert abs(voices.measure_voices_delay(tmp_path / "delayed.wav", tmp_path / "source.wav")) < voices.MAX_DELAY_S


def test_recordings_at_different_rates_are_refused(tmp_path):
    sf.write(tmp_path / "a.wav", noise(1.0), 16000)
    sf.write(tmp_path / "b.wav", noise(1.0), 8000)
    with pytest.raises(ValueError, match="Hz"):
        voices.measure_voices_delay(tmp_path / "a.wav", tmp_path / "b.wav")


def test_the_words_of_the_source_alignment_are_shifted_by_the_measured_delay(voices_tree):
    clips = [c for c in voices.clips(voices_tree) if c.speaker == "0083"]
    (t,) = voices.timings(voices_tree, clips)
    shift = VOICES_DELAY_SAMPLES // 16
    assert (t.clip_id, t.kind, t.mask_ms) == (clips[0].clip_id, "word", [])
    assert t.speech_ms == [(VOICES_WORDS_MS[0] + shift, VOICES_WORDS_MS[1] + shift)]


def test_a_word_shifted_past_the_recording_is_cut_to_it(voices_tree):
    grid = voices_tree.parent / "librispeech-alignments/train-clean-100/83/9000/83-9000-0001.TextGrid"
    grid.write_text(
        grid.read_text().replace('xmax = 1.5\n            text = "word"', 'xmax = 2.995\n            text = "word"')
    )
    clip = next(c for c in voices.clips(voices_tree) if c.speaker == "0083")
    (t,) = voices.timings(voices_tree, [clip])
    assert t.speech_ms == [(510, 3000)]


def test_voices_is_not_forced_aligned_and_is_never_cut():
    assert (voices.MFA_ALIGNED, voices.CUTTING) == (False, None)
