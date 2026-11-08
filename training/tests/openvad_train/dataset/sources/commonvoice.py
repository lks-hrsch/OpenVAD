# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json

import pytest

from mfa_source_support import add_cv_locale, make_cv_tree
from openvad_train.dataset.models import load_models
from openvad_train.dataset.sources import commonvoice
from openvad_train.fetch.commonvoice import CV_LOCALES


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


def test_mandarin_is_zh_cantonese_is_yue_and_taiwan_is_its_own_language(cv_tree):
    languages = {c.clip_id: c.language for c in commonvoice.clips(cv_tree)}
    assert languages["common_voice_zh-CN_1"] == "zh"
    assert languages["common_voice_zh-HK_1"] == "yue"
    assert languages["common_voice_zh-TW_1"] == "zh-TW"
    assert languages["common_voice_kab_1"] == "kab"  # Kabyle, an open row: indexed for the failure rule


def test_a_locale_that_is_not_fetched_yet_is_absent_not_an_error(cv_tree):
    assert {c.language for c in commonvoice.clips(cv_tree)} == {"de", "zh", "yue", "zh-TW", "kab"}  # no en directory


def test_a_locale_with_tsvs_but_neither_marker_nor_manifest_entry_is_skipped_and_reported(cv_tree, caplog):
    report = {}
    with caplog.at_level("WARNING", logger=commonvoice.log.name):
        clips = list(commonvoice.clips(cv_tree, None, report))
    assert not any(c.language == "ca" for c in clips)
    assert report["skipped"] == {"ca": {"reason": "fetch not finished", "clips": 1}}
    assert "ca is skipped" in caplog.text


def test_either_gate_lets_a_locale_in(cv_tree):
    (cv_tree / ".kab-extracted").unlink()
    assert not any(c.language == "kab" for c in commonvoice.clips(cv_tree))
    (cv_tree / "manifest.json").write_text(json.dumps({"locales": {"kab": {}}}))
    assert any(c.language == "kab" for c in commonvoice.clips(cv_tree))


def test_a_locale_whose_language_has_no_row_is_skipped_and_reported(cv_tree, monkeypatch):
    monkeypatch.setitem(commonvoice.CV_LOCALES, "fa", ("id", "sha"))
    add_cv_locale(cv_tree, "fa", {"common_voice_fa_1.mp3": ("p", "text")}, train=["common_voice_fa_1.mp3"])
    (cv_tree / ".fa-extracted").write_text("0")
    report = {}
    assert not any(c.language == "fa" for c in commonvoice.clips(cv_tree, None, report))
    assert report["skipped"]["fa"] == {"reason": "no row in mfa_models.tsv", "clips": 1}


def test_every_locale_of_the_fetcher_has_a_row_so_nothing_is_skipped_for_want_of_one():
    assert {commonvoice.language_of(locale) for locale in CV_LOCALES} <= set(load_models())


def test_only_the_asked_languages_have_their_clips_read_but_every_locale_gives_its_held_out_speakers(
    cv_tree, monkeypatch
):
    read = []
    original = commonvoice.read_tsv
    monkeypatch.setattr(commonvoice, "read_tsv", lambda p: read.append(p) or original(p))
    assert {c.language for c in commonvoice.clips(cv_tree, {"zh"})} == {"zh"}
    de = {p.name for p in read if "/de/" in str(p)}
    assert de == {"dev.tsv", "test.tsv"}


def test_a_speaker_held_out_in_one_locale_is_never_train_in_another_also_when_only_that_one_is_asked_for(cv_tree):
    # d is in the test of de and has a validated clip in zh-TW.
    add_cv_locale(
        cv_tree,
        "zh-TW",
        {"common_voice_zh-TW_1.mp3": ("t", "text"), "common_voice_zh-TW_2.mp3": ("d", "text")},
        train=["common_voice_zh-TW_1.mp3"],
    )
    for asked, left_out in (({"zh-TW"}, 1), (None, 2)):  # all: also the validated clip c2 of a dev speaker
        report = {}
        tw = {c.clip_id: c for c in commonvoice.clips(cv_tree, asked, report) if c.language == "zh-TW"}
        assert set(tw) == {"common_voice_zh-TW_1"}
        assert report["held_out_speaker_clips"] == left_out


def test_official_rows_stay_as_published_and_dev_test_overlaps_are_counted(cv_tree):
    # c is in the dev of de and the test of zh-CN.
    add_cv_locale(cv_tree, "zh-CN", {"common_voice_zh-CN_9.mp3": ("c", "text")}, test=["common_voice_zh-CN_9.mp3"])
    report = {}
    clips = {c.clip_id: c for c in commonvoice.clips(cv_tree, None, report)}
    assert (clips["common_voice_de_c1"].split, clips["common_voice_zh-CN_9"].split) == ("validation", "test")
    assert report["dev_test_speaker_overlaps"] == 1


def test_a_validated_clip_without_a_duration_is_refused(cv_tree):
    durations = cv_tree / "de" / "cv-corpus-27.0-2026-09-11" / "de" / "clip_durations.tsv"
    durations.write_text("clip\tduration[ms]\n")
    with pytest.raises(ValueError, match=r"clip_durations\.tsv"):
        list(commonvoice.clips(cv_tree, {"de"}))


def test_cv_is_an_mfa_source():
    assert commonvoice.MFA_ALIGNED is True
