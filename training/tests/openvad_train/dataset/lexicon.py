# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import sys
import zipfile

import pytest

from openvad_train.dataset import lexicon
from openvad_train.dataset.align import load_models
from openvad_train.dataset.lexicon import EpitranMissingError, ModelMetaError, build_lexicon, model_phones


def fake(word: str) -> list[str]:
    """One phone per letter, except `-` which has none and `q`, which the model does not know."""
    return [c for c in word if c != "-"]


def test_a_lexicon_maps_a_word_to_its_phones_in_the_line_format_of_a_dictionary():
    assert build_lexicon(["ab", "ba", "ab"], "ca", transliterate=fake) == {"ab": "a b", "ba": "b a"}


def test_a_word_with_a_phone_outside_the_inventory_is_left_out_and_counted():
    left_out: list[str] = []
    lex = build_lexicon(["ab", "aq", "qq", "ba"], "ca", phones={"a", "b"}, transliterate=fake, left_out=left_out)
    assert lex == {"ab": "a b", "ba": "b a"}
    assert left_out == ["aq", "qq"]


def test_without_an_inventory_every_word_with_phones_is_kept():
    assert build_lexicon(["aq"], "ca", transliterate=fake) == {"aq": "a q"}


def test_punctuation_the_transliterator_passes_through_is_no_phone_and_the_word_keeps_its_spelling():
    left_out: list[str] = []
    curly = "a\u2019b"  # the typographic apostrophe Common Voice also writes
    lex = build_lexicon(["l'home", "a-b", curly], "ca", phones=set("lhomeab"), transliterate=list, left_out=left_out)
    assert lex == {"l'home": "l h o m e", "a-b": "a b", curly: "a b"}
    assert left_out == []


def test_a_word_with_no_phones_is_skipped_and_not_counted_as_left_out():
    left_out: list[str] = []
    assert build_lexicon(["-", "a"], "ca", phones={"a"}, transliterate=fake, left_out=left_out) == {"a": "a"}
    assert left_out == []


def test_the_inventory_is_read_from_the_meta_of_the_acoustic_model(tmp_path):
    model = tmp_path / "ca_vxc_acoustic16.zip"
    with zipfile.ZipFile(model, "w") as z:
        z.writestr("ca_mfa_acoustic16/meta.json", json.dumps({"phones": ["a", "\u02c8e", "ɲ"], "version": "2.2.17"}))
        z.writestr("ca_mfa_acoustic16/phones.txt", "<eps> 0\n")
    assert model_phones(model) == {"a", "\u02c8e", "ɲ"}


def test_a_zip_without_meta_json_is_a_named_error(tmp_path):
    model = tmp_path / "not_a_model.zip"
    with zipfile.ZipFile(model, "w") as z:
        z.writestr("readme.txt", "no meta")
    with pytest.raises(ModelMetaError, match=r"no meta\.json"):
        model_phones(model)


def test_without_epitran_the_error_names_the_sync_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "epitran", None)  # an import of it raises ImportError, installed or not
    with pytest.raises(EpitranMissingError, match="uv sync --all-packages --group align"):
        build_lexicon(["pare"], "ca")


def test_a_language_without_an_epitran_code_is_an_error():
    with pytest.raises(ValueError, match="no Epitran code for language 'xx'"):
        build_lexicon(["a"], "xx")


def test_every_lexicon_route_language_is_in_the_models_table_with_its_route_in_the_note():
    models = load_models()
    for language in lexicon.EPITRAN_CODES:
        assert "Epitran" in models[language].note


@pytest.mark.filterwarnings("ignore::SyntaxWarning")  # panphon and jamo have invalid escapes, reported once per compile
def test_real_epitran_catalan_against_a_model_inventory():
    """Needs the `align` group: `uv sync --all-packages --group align`, then `uv run pytest <this file>`."""
    pytest.importorskip("epitran", reason="Epitran is in the `align` group: uv sync --all-packages --group align")
    left_out: list[str] = []
    inventory = {"p", "a", "ɾ", "k", "t", "s", "ʃ", "ɔ", "ʎ", "u", "ŋ", "o"}
    lex = build_lexicon(["pare", "casa", "ull", "cançó"], "ca", phones=inventory, left_out=left_out)
    # `e` is the open ɛ in Epitran, which this inventory lacks; the stray cedilla of `cançó` is no phone either.
    assert lex == {"casa": "k a s a", "ull": "u ʎ"}
    assert left_out == ["pare", "cançó"]


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_real_epitran_catalan_apostrophe_and_hyphen_words_get_only_phones():
    pytest.importorskip("epitran", reason="Epitran is in the `align` group: uv sync --all-packages --group align")
    import epitran

    words = ["l'home", "cinquanta-sis"]
    inventory = {p for w in words for p in epitran.Epitran("cat-Latn").trans_list(w)} - {"'", "-"}
    lex = build_lexicon(words, "ca", phones=inventory)
    assert set(lex) == set(words)
    assert all(p in inventory for pron in lex.values() for p in pron.split())
