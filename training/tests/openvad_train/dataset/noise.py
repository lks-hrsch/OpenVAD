# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import csv

import numpy as np
import pytest
import soundfile as sf

from demand_support import make_demand_tree
from openvad_train.dataset.config import DEMAND_CATEGORIES
from openvad_train.dataset.noise import (
    MUSAN_CATEGORIES,
    NoisePool,
    category_of_demand,
    demand_channels,
    musan_noise_categories,
)
from openvad_train.fetch.musan import musan_licences


@pytest.fixture
def demand_tree(tmp_path):
    return make_demand_tree(tmp_path / "demand", ["DKITCHEN", "DLIVING", "TBUS"])


def test_demand_scene_categories():
    assert category_of_demand("DKITCHEN") == "domestic"
    assert category_of_demand("SPSQUARE") == "street"
    assert category_of_demand("TBUS") == "transport"
    with pytest.raises(ValueError, match="scene"):
        category_of_demand("XYZ")


def test_demand_channels_split_like_librivad():
    assert list(demand_channels("train")) == list(range(1, 13))
    assert list(demand_channels("validation")) == [13, 14]
    assert list(demand_channels("test")) == [15, 16]
    with pytest.raises(ValueError, match="split"):
        demand_channels("dev")


def test_pool_draws_only_split_channels(demand_tree):
    pool = NoisePool(demand_tree)
    draws = [pool.draw("domestic", "test", np.random.default_rng(i)) for i in range(50)]
    assert {d.channel for d in draws} <= {15, 16}
    assert {d.channel for d in draws} == {15, 16}
    assert {d.file.parent.name for d in draws} == {"DKITCHEN", "DLIVING"}
    assert all(0 <= d.offset_n < 1600 and d.file.name == f"ch{d.channel}.wav" for d in draws)


def test_pool_draw_is_a_function_of_rng_and_not_listing_order(tmp_path):
    forward = NoisePool(make_demand_tree(tmp_path / "a", ["DKITCHEN", "DLIVING", "DWASHING"]))
    backward = NoisePool(make_demand_tree(tmp_path / "b", ["DWASHING", "DLIVING", "DKITCHEN"]))
    for i in range(20):
        a = forward.draw("domestic", "train", np.random.default_rng(i))
        b = backward.draw("domestic", "train", np.random.default_rng(i))
        assert (a.file.relative_to(tmp_path / "a"), a.channel, a.offset_n) == (
            b.file.relative_to(tmp_path / "b"),
            b.channel,
            b.offset_n,
        )


def test_pool_rejects_unknown_or_empty_category(demand_tree):
    pool = NoisePool(demand_tree)
    with pytest.raises(ValueError, match="no DEMAND scene"):
        pool.draw("street", "train", np.random.default_rng(0))
    with pytest.raises(ValueError, match="category"):
        pool.draw("music", "train", np.random.default_rng(0))


def test_pool_reads_each_header_once(demand_tree, monkeypatch):
    reads = []
    real_info = sf.info
    monkeypatch.setattr(sf, "info", lambda file: reads.append(file) or real_info(file))
    pool = NoisePool(demand_tree)
    draws = [pool.draw("transport", "test", np.random.default_rng(i)) for i in range(40)]
    assert len(reads) == len({d.file for d in draws}) <= 2
    assert draws[0] == NoisePool(demand_tree).draw("transport", "test", np.random.default_rng(0))  # caching keeps draws


def test_the_musan_table_has_one_row_for_every_noise_file_of_the_licence_table():
    noise_files = {f for f in musan_licences() if f.startswith("noise-")}
    with MUSAN_CATEGORIES.open() as f:
        listed = [row[0] for row in csv.reader(f, delimiter="\t")][1:]
    assert len(listed) == len(set(listed))
    assert set(listed) == noise_files == set(musan_noise_categories())


def test_musan_files_only_join_categories_that_are_drawn_from_files():
    # Babble and SSN are built from reserved speakers, and City is WHAM!, so no MUSAN file can belong to them.
    assert set(musan_noise_categories().values()) <= {*DEMAND_CATEGORIES, "unused"}
    assert any(c != "unused" for c in musan_noise_categories().values())
