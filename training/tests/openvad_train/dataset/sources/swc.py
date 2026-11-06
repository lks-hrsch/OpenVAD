# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict

import pytest

from openvad_train.dataset.schema import rank
from openvad_train.dataset.sources import swc
from swc_support import make_swc_tree


@pytest.fixture
def swc_tree(tmp_path):
    return make_swc_tree(tmp_path)


def test_one_clip_per_aligned_article(swc_tree):
    clips = {c.clip_id: c for c in swc.clips(swc_tree)}
    assert sorted(clips) == ["dutch/Eee", "english/Aaa", "english/Bbb", "german/Ccc", "german/Ddd"]
    c = clips["english/Aaa"]
    assert (c.source, c.language, c.licence, c.offset_s, c.duration_s) == ("swc", "en", "CC BY-SA", 0.0, 2.0)
    assert c.path == "datasets/raw/swc/english/Aaa/audio.ogg"


def test_reader_is_in_exactly_one_split(swc_tree):
    by_reader = defaultdict(set)
    for c in swc.clips(swc_tree):
        by_reader[c.speaker].add(c.split)
    assert all(len(s) == 1 for s in by_reader.values())


def test_reader_forms_merge(swc_tree):
    speakers = {c.clip_id: c.speaker for c in swc.clips(swc_tree)}
    assert speakers["english/Aaa"] == speakers["english/Bbb"] == "jcb"
    assert speakers["german/Ddd"] == speakers["dutch/Eee"] == "zed q"
    assert speakers["german/Ccc"] is None


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ("[[User:Zanimum|Nicholas Moreau]]", "zanimum"),
        ("user at projectJcbwikipedianl", "jcb"),
        ("[[:en:User:Alexkillby|Alexkillby]] ([[:en:User talk:Alexkillby|talk]])", "alexkillby"),
        ("[[User:Jcb|Jcb]] and [[User:VR-Land|VR-Land]]", "jcb"),
        ("S_Whistler", "s whistler"),
        ("", None),
    ],
)
def test_reader_name_rules(raw, name):
    assert swc.reader_name({"audio_file_parsed": {"reader": raw}}) == name


def test_article_reader_is_the_fallback():
    assert swc.reader_name({"article_parsed": {"reader": "Blik"}, "audio_file_parsed": {"reader": None}}) == "blik"


def test_split_is_80_10_10_by_rank_and_no_reader_is_train():
    assert swc.split_of(None) == "train"
    shares = defaultdict(int)
    for i in range(2000):
        shares[swc.split_of(f"reader{i}")] += 1
    assert shares["train"] == pytest.approx(1600, abs=100)
    assert shares["validation"] == pytest.approx(200, abs=60)
    assert shares["test"] == pytest.approx(200, abs=60)
    r = next(f"r{i}" for i in range(100) if rank(f"r{i}") % 10 == 9)
    assert swc.split_of(r) == "test"


def test_unaligned_token_masks_the_gap(swc_tree):
    t = next(swc.timings(swc_tree, swc.clips(swc_tree)))
    assert t.clip_id == "english/Aaa"
    assert (300, 450) in t.mask_ms
    assert t.kind == "word"


def test_only_gaps_beside_unaligned_tokens_are_masked(swc_tree):
    t = next(swc.timings(swc_tree, swc.clips(swc_tree)))
    assert t.speech_ms == [(100, 300), (450, 600), (700, 800)]
    # The intro leads, the outro trails and the ignored block counts for nothing; 600 to 700 has no unaligned token.
    assert t.mask_ms == [(0, 100), (300, 450), (800, 2000)]
