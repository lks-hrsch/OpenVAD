# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from mfa_source_support import make_cv_tree
from openvad_train.dataset.sources import commonvoice


@pytest.fixture
def cv_tree(tmp_path):
    return make_cv_tree(tmp_path)


def by_id(raw, languages=None):
    return {c.clip_id: c for c in commonvoice.clips(raw, languages)}


def test_cv_train_is_validated_minus_dev_and_test_speakers(cv_tree):
    clips = [c for c in commonvoice.clips(cv_tree) if c.language == "de"]
    held = {c.speaker for c in clips if c.split != "train"}
    assert held == {"c", "d"}
    assert not held & {c.speaker for c in clips if c.split == "train"}
    assert {c.clip_id for c in clips if c.split == "train"} == {
        "common_voice_de_a1",
        "common_voice_de_a2",
        "common_voice_de_b1",
        "common_voice_de_short",
        "validated_only_clip",
    }


def test_a_validated_clip_of_a_dev_speaker_is_left_out_not_trained_on(cv_tree):
    clips = by_id(cv_tree)
    assert "common_voice_de_c2" not in clips
    assert clips["common_voice_de_c1"].split == "validation"
    assert clips["common_voice_de_d1"].split == "test"


def test_cv_clip_fields_and_duration_from_the_durations_file(cv_tree):
    c = by_id(cv_tree)["common_voice_de_a1"]
    assert (c.source, c.language, c.speaker, c.licence, c.offset_s, c.duration_s) == (
        "common-voice",
        "de",
        "a",
        "CC0-1.0",
        0.0,
        1.5,
    )
    assert c.transcript == "One sentence."
    assert c.path == "datasets/raw/common-voice/de/cv-corpus-27.0-2026-09-11/de/clips/common_voice_de_a1.mp3"
    assert by_id(cv_tree)["common_voice_de_short"].duration_s == 1.25


def test_locales_without_an_mfa_model_are_left_out_and_mandarin_is_zh(cv_tree):
    languages = {c.clip_id: c.language for c in commonvoice.clips(cv_tree)}
    assert languages["common_voice_zh-CN_1"] == "zh"
    assert not any(clip_id in languages for clip_id in ("common_voice_zh-TW_1", "common_voice_kab_1"))


def test_a_locale_that_is_not_fetched_yet_is_absent_not_an_error(cv_tree):
    assert {c.language for c in commonvoice.clips(cv_tree)} == {"de", "zh"}  # no en, fr ... directory exists


def test_only_the_asked_languages_are_read(cv_tree, monkeypatch):
    read = []
    original = commonvoice.read_tsv
    monkeypatch.setattr(commonvoice, "read_tsv", lambda p: read.append(p) or original(p))
    assert {c.language for c in commonvoice.clips(cv_tree, {"zh"})} == {"zh"}
    assert not any("/de/" in str(p) for p in read)


def test_a_validated_clip_without_a_duration_is_refused(cv_tree):
    durations = cv_tree / "de" / "cv-corpus-27.0-2026-09-11" / "de" / "clip_durations.tsv"
    durations.write_text("clip\tduration[ms]\n")
    with pytest.raises(ValueError, match=r"clip_durations\.tsv"):
        list(commonvoice.clips(cv_tree, {"de"}))


def test_cv_is_an_mfa_source():
    assert commonvoice.MFA_ALIGNED is True
