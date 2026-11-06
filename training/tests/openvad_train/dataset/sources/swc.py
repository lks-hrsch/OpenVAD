# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import shutil
from collections import defaultdict

import pytest
import soundfile as sf

from openvad_train.dataset.schema import rank
from openvad_train.dataset.sources import swc
from swc_support import add_article, make_swc_tree


@pytest.fixture
def swc_tree(tmp_path):
    return make_swc_tree(tmp_path)


def test_one_clip_per_aligned_single_file_article(swc_tree):
    clips = {c.clip_id: c for c in swc.clips(swc_tree)}
    assert sorted(clips) == ["dutch/Eee", "english/Aaa", "english/Bbb", "german/Ccc", "german/Ddd"]
    c = clips["english/Aaa"]
    assert (c.source, c.language, c.licence, c.offset_s, c.duration_s) == ("swc", "en", "CC BY-SA", 0.0, 2.0)
    assert c.path == "datasets/raw/swc/english/Aaa/audio.ogg"


# (audio file reader, article reader) of real articles that are one person; the forms are the corpus's.
SAME_PERSON = [
    [("[[User:Vikarino|Vikarino]]", "Vikarino"), ("Sarah Oltmanns", "Vikarino")],
    [("[[User:Mnntoino|Mnntoino]]", "[[Benutzer:mnntoino]]"), ("[[User:Mnntoino|Mnntoino]]", "mnntoino")],
    [("", "[[Denis Apel|Benutzer:Denis_Apel]]"), ("[[Denis Apel|Benutzer:Denis Apel]]", None)],
    [("Martin Schlederer (Sprecher)", "Martin Schlederer"), ("Reader: Martin Schlederer", "Martin Schlederer")],
    [("Blanki_Ostfalen", "Blanki Ostfalen"), ("[[User:Blanki Ostfalen|Blanki Ostfalen]]", "Blanki Ostfalen")],
    [("Spoken by:  Hi878  Sound editing by:  London Rook", None), ("Hi878", None)],
    [("SpeakThings:Mellerbeck", None), ("[[User:Mellerbeck|Mellerbeck]]", None)],
]


def test_reader_is_in_exactly_one_split(tmp_path):
    names = {}
    for i, group in enumerate(SAME_PERSON):
        for j, (audio, article) in enumerate(group):
            add_article(tmp_path, "german", f"g{i}_{j}", audio, article_reader=article)
            names[f"german/g{i}_{j}"] = i
    add_article(tmp_path, "german", "alone", "Verlagsgruppe Random House")
    by_person = defaultdict(set)
    for c in swc.clips(tmp_path):
        if c.clip_id in names:
            by_person[names[c.clip_id]].add((c.speaker, c.split))
    assert all(len(pairs) == 1 for pairs in by_person.values()), dict(by_person)
    assert len({next(iter(pairs))[0] for pairs in by_person.values()}) == len(SAME_PERSON)


def test_different_people_stay_apart(tmp_path):
    add_article(tmp_path, "german", "a", "Alice", article_reader="Alice")
    add_article(tmp_path, "german", "b", "Bob")
    assert len({c.speaker for c in swc.clips(tmp_path)}) == 2


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
        ("original uploaderAfrocatzwikipediaen", "afrocatz"),
        ("User at projectTdxiangwikipediaen.", "tdxiang"),
        ("user at projectWouter Druwéwikipedianl<br/>\n(original textnobold=11=Wouter)", "wouter druwé"),
        ("Original uploader was [[:nl:User:Jcb|Jcb]] at [http://nl.wikipedia.org nl.wikipedia]", "jcb"),
        ("[[:en:User:Alexkillby|Alexkillby]] ([[:en:User talk:Alexkillby|talk]])", "alexkillby"),
        ("[[User:Jcb|Jcb]] and [[User:VR-Land|VR-Land]]", "jcb"),
        ("[[w:en:User:CaptainComedy|James Avestruz]]", "captaincomedy"),
        ("Author assumed[[User:Kander|Kander]]", "kander"),
        ("[[Denis Apel|Benutzer:Denis Apel]]", "denis apel"),
        ("[[Die.keimzelle]]", "die.keimzelle"),
        ("[[Benutzer:mnntoino]]", "mnntoino"),
        ("Simon Jäckel [[Benutzer:Ahdar-Ruafo]]", "ahdar-ruafo"),
        ("Spoken by:  Hi878  Sound editing by:  London Rook", "hi878"),
        ("SpeakThings:Mellerbeck", "mellerbeck"),
        (":nl:Paus", "paus"),
        ("Synon~commonswiki", "synon"),
        ("S_Whistler", "s whistler"),
        ("Verlagsgruppe Random House", "verlagsgruppe random house"),
        ("", None),
        (None, None),
    ],
)
def test_reader_name_rules(raw, name):
    assert swc.reader_name(raw) == name


def test_article_readers_take_both_fields():
    info = {"article_parsed": {"reader": "Blik"}, "audio_file_parsed": {"reader": "[[User:Blik|Blik]]"}}
    assert swc.article_readers(info) == {"blik"}
    assert swc.article_readers({"article_parsed": {"reader": "A"}, "audio_file_parsed": {"reader": "B"}}) == {"a", "b"}
    assert swc.article_readers({"article_parsed": {"reader": None}, "audio_file_parsed": None}) == set()


def test_merge_readers_joins_connected_components_to_the_first_name():
    merged = swc.merge_readers([{"b", "c"}, {"c", "d"}, {"x"}, {"a", "d"}])
    assert merged == {"a": "a", "b": "a", "c": "a", "d": "a", "x": "x"}


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


def test_duration_falls_back_to_ffprobe_for_files_libsndfile_cannot_open(swc_tree, monkeypatch):
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")

    def refuse(_):
        raise sf.LibsndfileError(1, "unimplemented format")

    monkeypatch.setattr(swc.sf, "info", refuse)
    assert swc.audio_duration_s(swc_tree / "english/Aaa/audio.ogg") == pytest.approx(2.0, abs=0.05)


def test_never_aligned_clip_edges_are_masked_without_unaligned_tokens(tmp_path):
    add_article(tmp_path, "german", "Edges", None, words=[(100, 300), (400, 500)], boilerplate=False)
    t = next(swc.timings(tmp_path, swc.clips(tmp_path)))
    assert t.speech_ms == [(100, 300), (400, 500)]
    assert t.mask_ms == [(0, 100), (500, 2000)]


def test_a_clip_without_an_aligned_word_is_masked_whole(tmp_path):
    add_article(tmp_path, "german", "Empty", None, words=[None, None])
    t = next(swc.timings(tmp_path, swc.clips(tmp_path)))
    assert (t.speech_ms, t.mask_ms) == ([], [(0, 2000)])
