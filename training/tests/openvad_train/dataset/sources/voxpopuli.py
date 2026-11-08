# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict

import pytest

from mfa_source_support import make_vp_tree
from openvad_train.dataset.sources import voxpopuli


@pytest.fixture
def vp_tree(tmp_path):
    return make_vp_tree(tmp_path)


def test_vp_invalid_rows_are_dropped_and_none_is_train_only(vp_tree):
    clips = list(voxpopuli.clips(vp_tree))
    assert not any(c.clip_id == "invalid_row" for c in clips)
    assert {c.split for c in clips if c.speaker is None} == {"train"}
    assert sum(c.speaker is None for c in clips) == 3  # also the one the release calls dev


def test_vp_split_by_speaker(vp_tree):
    by = defaultdict(set)
    for c in voxpopuli.clips(vp_tree):
        if c.speaker:
            by[c.speaker].add(c.split)
    assert all(len(s) == 1 for s in by.values())
    assert {next(iter(s)) for s in by.values()} == {"train", "validation", "test"}


def test_vp_clip_fields(vp_tree):
    c = next(c for c in voxpopuli.clips(vp_tree) if c.clip_id == "de_s3_2")
    assert (c.source, c.language, c.speaker, c.licence, c.offset_s, c.duration_s) == (
        "voxpopuli",
        "de",
        "3",
        "CC0-1.0",
        200.0,
        60.0,
    )
    assert c.transcript == "normed six"
    assert c.path == "datasets/raw/voxpopuli/raw_audios/original/2014/20140224-0900-PLENARY-3_original.ogg"


def test_vp_leaves_out_a_language_without_an_mfa_model_and_one_not_asked_for(vp_tree):
    assert {c.language for c in voxpopuli.clips(vp_tree)} == {"de"}
    assert list(voxpopuli.clips(vp_tree, {"fr"})) == []


def test_vp_is_an_mfa_source():
    assert voxpopuli.MFA_ALIGNED is True
