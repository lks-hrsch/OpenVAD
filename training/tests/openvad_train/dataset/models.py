# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses
from importlib import resources
from pathlib import Path

import pytest

from openvad_train.dataset.models import ModelSpec, indexed_languages, load_models, spec_for


def test_models_table_ships_with_the_package_and_loads():
    packaged = resources.files("openvad_train.dataset").joinpath("mfa_models.tsv")
    assert packaged.is_file()
    models = load_models(Path(str(packaged)))
    assert models == load_models()
    assert {"en", "de", "fr", "es", "it", "nl", "pl", "pt", "ja", "zh", "kk"} <= models.keys()


def test_a_short_row_is_refused(tmp_path):
    table = tmp_path / "models.tsv"
    header = "\t".join(f.name for f in dataclasses.fields(ModelSpec))
    table.write_text(f"{header}\nde\tgerman_mfa\t3.0.0\tgerman_mfa\t3.0.0\t\t\tnote\n")
    with pytest.raises(ValueError, match="fewer cells"):
        load_models(table)


def test_italian_and_dutch_have_no_g2p_and_others_do():
    models = load_models()
    assert models["it"].g2p is None
    assert models["nl"].g2p is None
    assert models["de"].g2p == "german_mfa"
    assert models["de"].g2p_version == "3.0.0"


def test_every_row_says_how_far_it_is_verified_and_open_choices_stay_open():
    models = load_models()
    assert {m.status for m in models.values()} <= {"downloaded", "listed", "open"}
    assert {lang for lang, m in models.items() if m.status == "downloaded"} == {"de"}
    for language in ("cs", "bg", "uk", "ca"):
        assert models[language].status == "open"
        assert not models[language].acoustic
        with pytest.raises(ValueError, match="no MFA model is chosen"):
            spec_for(models, language)
    with pytest.raises(ValueError, match="no row for language 'xx'"):
        spec_for(models, "xx")


def test_indexed_languages_are_the_tables_rows_open_ones_included_narrowed_by_the_request():
    table = set(load_models())
    assert indexed_languages(None) == table
    assert "ca" in table  # open: indexed, and align refuses it until the owner decides
    assert not table & {"fa", "ur", "ar", "mhr", "lzz", "kbd", "ady", "qu"}  # ADR 0023's eight have no model
    assert {"kab", "rw", "be", "yue", "zh-TW"} <= table  # Kabyle is not Kabardian; open rows, so their hours count
    assert all(m.status == "open" for lang, m in load_models().items() if lang in {"kab", "rw", "be", "yue", "zh-TW"})
    assert indexed_languages({"de", "kab", "xx"}) == {"de", "kab"}


def test_every_language_any_source_can_have_on_disk_has_a_row():
    from openvad_train.dataset.sources import commonvoice, fleurs, mls
    from openvad_train.fetch.commonvoice import CV_LOCALES
    from openvad_train.fetch.fleurs import FLEURS_LANGS
    from openvad_train.fetch.voxpopuli import VOXPOPULI_LANGS

    needed = {commonvoice.language_of(locale) for locale in CV_LOCALES}
    needed |= {fleurs._language(d) for d in FLEURS_LANGS} | set(mls.LANGUAGE.values()) | set(VOXPOPULI_LANGS)
    assert needed <= set(load_models())
