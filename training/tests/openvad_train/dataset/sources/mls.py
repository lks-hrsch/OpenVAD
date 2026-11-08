# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from mfa_source_support import make_mls_tree
from openvad_train.dataset.sources import mls


@pytest.fixture
def mls_tree(tmp_path):
    return make_mls_tree(tmp_path)


def test_mls_keeps_official_splits(mls_tree):
    by_id = {c.clip_id: c.split for c in mls.clips(mls_tree)}
    assert by_id == {
        "de/10_100_000000": "train",
        "de/11_101_000003": "train",
        "de/20_200_000000": "validation",
        "de/30_300_000007": "test",
    }


def test_mls_clip_fields(mls_tree):
    c = next(c for c in mls.clips(mls_tree) if c.clip_id == "de/11_101_000003")
    assert (c.source, c.language, c.speaker, c.licence, c.offset_s, c.duration_s) == (
        "mls",
        "de",
        "11",
        "CC BY 4.0",
        0,
        3.75,
    )
    assert c.transcript == "text of 11_101_000003"
    assert c.path == "datasets/raw/mls/mls_german/train/audio/11/101/11_101_000003.flac"


def test_mls_leaves_out_a_language_we_do_not_use_and_one_not_asked_for(mls_tree):
    assert {c.language for c in mls.clips(mls_tree)} == {"de"}
    assert list(mls.clips(mls_tree, {"fr"})) == []


def test_a_language_folder_we_have_no_row_for_is_skipped_and_reported(mls_tree):
    report = {}
    assert {c.language for c in mls.clips(mls_tree, None, report)} == {"de"}
    assert report["skipped"] == {"mls_english": {"reason": "no row in mfa_models.tsv", "clips": 1}}


def test_mls_is_an_mfa_source():
    assert mls.MFA_ALIGNED is True
