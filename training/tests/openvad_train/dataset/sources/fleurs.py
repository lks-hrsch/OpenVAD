# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from mfa_source_support import make_fleurs_tree
from openvad_train.dataset.sources import fleurs


@pytest.fixture
def fleurs_tree(tmp_path):
    return make_fleurs_tree(tmp_path)


def test_fleurs_has_no_speakers(fleurs_tree):
    clips = list(fleurs.clips(fleurs_tree))
    assert clips
    assert all(c.speaker is None for c in clips)


def test_fleurs_keeps_official_splits_and_maps_mandarin(fleurs_tree):
    splits = {(c.language, c.split) for c in fleurs.clips(fleurs_tree)}
    assert splits == {(lang, s) for lang in ("de", "zh") for s in ("train", "validation", "test")}


def test_fleurs_fields_from_the_raw_text_and_the_sample_count(fleurs_tree):
    c = next(c for c in fleurs.clips(fleurs_tree) if c.clip_id == "de_de/de_de-dev")
    assert (c.source, c.licence, c.offset_s, c.duration_s, c.transcript) == (
        "fleurs",
        "CC BY 4.0",
        0.0,
        3.0,
        "Raw dev.",
    )
    assert c.path == "datasets/raw/fleurs/de_de/audio/dev/de_de-dev.wav"


def test_fleurs_leaves_out_languages_without_an_mfa_model_and_unasked_ones(fleurs_tree):
    assert not any(c.language in ("hu", "yue") for c in fleurs.clips(fleurs_tree))
    assert {c.language for c in fleurs.clips(fleurs_tree, {"zh"})} == {"zh"}


def test_fleurs_is_an_mfa_source():
    assert fleurs.MFA_ALIGNED is True
