# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import shutil
from collections import defaultdict
from random import Random

import pytest
import soundfile as sf

from openvad_train.dataset.sources import swc
from swc_support import add_article, make_swc_tree


@pytest.fixture
def swc_tree(tmp_path):
    return make_swc_tree(tmp_path)


def test_one_clip_per_aligned_article(swc_tree):
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


def de(hours: dict[str, float]) -> dict[str, dict[str, float]]:
    return {reader: {"de": h} for reader, h in hours.items()}


def test_split_assignment_follows_aligned_hours_not_reader_count():
    hours = {f"r{i}": 1.0 for i in range(10)} | {"big": 20.0, "bigger": 40.0}
    got = swc.assign_splits(de(hours))
    assert got["bigger"] == "train"  # over 30 h
    placed = {s: sum(min(hours[r], 30.0) for r in hours if got[r] == s) for s in swc.SPLIT_SHARES}
    total = sum(placed.values())
    assert placed["validation"] / total == pytest.approx(0.1, abs=0.04)
    assert placed["test"] / total == pytest.approx(0.1, abs=0.04)


def test_equal_readers_split_8_1_1():
    got = swc.assign_splits(de({f"r{i}": 2.0 for i in range(10)}))
    assert sorted(got.values()) == ["test"] + ["train"] * 8 + ["validation"]


def test_a_small_language_with_one_dominant_reader_still_has_validation_and_test():
    got = swc.assign_splits(de({"jcb": 66.1} | {f"r{i}": 1.0 for i in range(20)}))
    assert got["jcb"] == "train"
    assert {"validation", "test"} <= set(got.values())


def test_assign_splits_is_deterministic_and_independent_of_input_order():
    hours = {f"r{i}": 0.5 + i % 7 for i in range(60)}
    assert swc.assign_splits(de(hours)) == swc.assign_splits(de(dict(reversed(hours.items()))))
    assert swc.assign_splits({}) == {}


def test_readers_in_both_languages_split_8_1_1_in_each(tmp_path):
    for i in range(10):
        add_article(tmp_path, "german", f"d{i}", f"reader{i}")
        add_article(tmp_path, "english", f"e{i}", f"reader{i}")
    by_language = defaultdict(lambda: defaultdict(int))
    for c in swc.clips(tmp_path):
        by_language[c.language][c.split] += 1
    assert by_language["de"] == by_language["en"] == {"train": 8, "validation": 1, "test": 1}


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


def word_timing(tmp_path, words, seconds=30.0):
    add_article(tmp_path, "german", "Gaps", None, words=words, boilerplate=False, seconds=seconds)
    return next(swc.timings(tmp_path, swc.clips(tmp_path)))


def test_an_inner_gap_of_two_seconds_is_masked_and_a_shorter_one_is_not(tmp_path):
    t = word_timing(tmp_path, [(100, 300), (2300, 2500), (4499, 4600)])
    assert t.speech_ms == [(100, 300), (2300, 2500), (4499, 4600)]
    assert t.mask_ms == [(0, 100), (300, 2300), (4600, 30000)]


def test_a_word_of_five_seconds_stays_and_a_longer_one_becomes_a_mask(tmp_path):
    t = word_timing(tmp_path, [(100, 5100), (5200, 10_201), (10_300, 10_400)])
    assert t.speech_ms == [(100, 5100), (10_300, 10_400)]
    # The mask runs over the gaps beside it too, as it does around an unaligned token.
    assert t.mask_ms == [(0, 100), (5100, 10_300), (5200, 10_201), (10_400, 30000)]


def test_an_over_long_first_or_last_word_falls_inside_the_edge_masks(tmp_path):
    t = word_timing(tmp_path, [(100, 6000), (6100, 6200), (6300, 12_000)])
    assert t.speech_ms == [(6100, 6200)]
    assert t.mask_ms == [(0, 6100), (100, 6000), (6200, 30000), (6300, 12_000)]


def test_a_clip_of_only_over_long_words_is_masked_whole(tmp_path):
    t = word_timing(tmp_path, [(100, 6000)])
    assert (t.speech_ms, t.mask_ms) == ([], [(0, 30000)])


def test_a_slice_and_the_full_build_give_every_reader_the_same_split(tmp_path):
    # Hours that differ per language: split on the German hours alone, 10 of these 30 readers would move.
    random = Random(0)
    for i in range(30):
        for language in ("german", "english"):
            words = [(0, round(random.uniform(1, 10) * 1000))]
            add_article(tmp_path, language, f"{language[:2]}{i}", f"reader{i}", words=words, seconds=20.0)
    full = {c.clip_id: (c.speaker, c.split) for c in swc.clips(tmp_path)}
    sliced = {c.clip_id: (c.speaker, c.split) for c in swc.clips(tmp_path, {"de"})}
    assert sliced
    assert {k: v for k, v in full.items() if k.startswith("german/")} == sliced
    assert {c.language for c in swc.clips(tmp_path, {"de"})} == {"de"}


def test_a_multi_part_article_gives_one_clip_per_part_with_rebased_words(tmp_path):
    # Word times run over the concatenation of the parts (1.0 s then 1.5 s); the word 900 to 1300 crosses the boundary.
    words = [(100, 300), (900, 1300), (1500, 1800)]
    add_article(tmp_path, "german", "Multi", "Alice", parts=(1.0, 1.5), words=words, boilerplate=False)
    got = list(swc.clips(tmp_path))
    assert [(c.clip_id, c.path, c.duration_s) for c in got] == [
        ("german/Multi#1", "datasets/raw/swc/german/Multi/audio1.ogg", 1.0),
        ("german/Multi#2", "datasets/raw/swc/german/Multi/audio2.ogg", 1.5),
    ]
    first, second = swc.timings(tmp_path, got)
    assert (first.speech_ms, first.mask_ms) == ([(100, 300), (900, 1000)], [(0, 100)])
    assert (second.speech_ms, second.mask_ms) == ([(0, 300), (500, 800)], [(800, 1500)])


def test_parts_are_ordered_by_number_not_by_name(tmp_path):
    add_article(tmp_path, "german", "Ten", "Alice", parts=(0.2,) * 10)
    assert [c.clip_id.split("#")[1] for c in swc.clips(tmp_path)] == [str(i) for i in range(1, 11)]


def test_an_aligned_article_without_audio_is_an_error(tmp_path):
    add_article(tmp_path, "german", "Lost", "Alice")
    (tmp_path / "german/Lost/audio.ogg").unlink()
    with pytest.raises(FileNotFoundError, match="without audio"):
        list(swc.clips(tmp_path))


def test_a_reader_of_two_languages_gets_one_split(tmp_path):
    # Twelve readers, each in German and English with different aligned hours per language, so placing each
    # language on its own would send some of them to different splits.
    for i in range(12):
        reader = f"reader{i}"
        add_article(tmp_path, "german", f"d{i}", reader, words=[(0, 1000 * (1 + i % 5))], seconds=10.0)
        add_article(tmp_path, "english", f"e{i}", reader, words=[(0, 1000 * (1 + (i * 3) % 7))], seconds=10.0)
    splits = defaultdict(set)
    for c in swc.clips(tmp_path):
        splits[c.speaker].add(c.split)
    assert len(splits) == 12
    assert all(len(s) == 1 for s in splits.values()), dict(splits)


def test_a_bilingual_reader_is_placed_by_its_largest_language_and_counts_in_both():
    # Over 30 h is train; "mixed" is mostly English, so English decides, and its hours count in German too.
    hours = {"mixed": {"de": 1.0, "en": 5.0}, "huge": {"de": 20.0, "en": 20.0}}
    got = swc.assign_splits(hours)
    assert got["huge"] == "train"
    assert got["mixed"] in swc.SPLIT_SHARES


def test_ordinary_readers_keep_each_languages_shares_near_80_10_10():
    hours = {f"de{i}": {"de": 1.0 + i % 4} for i in range(80)}
    hours |= {f"en{i}": {"en": 1.0 + i % 3} for i in range(80)}
    hours |= {f"both{i}": {"de": 1.0 + i % 2, "en": 1.0 + i % 3} for i in range(20)}
    got = swc.assign_splits(hours)
    for language in ("de", "en"):
        per_split = defaultdict(float)
        for reader, by_language in hours.items():
            if language in by_language:
                per_split[got[reader]] += by_language[language]
        total = sum(per_split.values())
        assert per_split["train"] / total == pytest.approx(0.8, abs=0.06)
        assert per_split["validation"] / total == pytest.approx(0.1, abs=0.05)
        assert per_split["test"] / total == pytest.approx(0.1, abs=0.05)


def test_swc_labels_come_from_its_own_timings_not_from_mfa():
    assert swc.MFA_ALIGNED is False
