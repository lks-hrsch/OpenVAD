# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from openvad_train.dataset.config import FULL, SLICE, dataset_root, work_dir


def test_full_condition_shares_sum_to_one():
    assert sum(FULL.condition_shares.values()) == pytest.approx(1.0)
    assert FULL.condition_shares["clean"] == 0.15
    assert FULL.condition_shares["competing"] == 0.15
    assert FULL.condition_shares["babble"] == pytest.approx(0.7 / 9)
    assert len(FULL.condition_shares) == 11


def test_full_snr_grid():
    assert FULL.snr_grid_db == (-5, 0, 5, 10, 15, 20)


def test_full_codec_shares_sum_to_one():
    assert sum(FULL.codec_shares.values()) == pytest.approx(1.0)
    assert set(FULL.codec_bitrate_kbps) == set(FULL.codec_shares) - {"none"}


def test_full_spec_scalars():
    assert (FULL.p_narrow, FULL.reserved_share, FULL.babble_talkers, FULL.gap_ms) == (0.35, 0.05, 6, 200)
    assert FULL.dither_dbfs == (-90.0, -70.0)


def test_slice_is_swc_german_demand_with_two_codecs():
    assert (SLICE.sources, SLICE.noise, SLICE.languages) == (("swc",), ("demand",), ("de",))
    assert set(SLICE.codec_shares) == {"none", "g722"}
    assert sum(SLICE.codec_shares.values()) == pytest.approx(1.0)
    assert sum(SLICE.condition_shares.values()) == pytest.approx(1.0)


def test_dataset_root_follows_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    assert dataset_root(FULL) == tmp_path / "datasets/processed/openvad-v1"


def test_work_dir_sits_next_to_the_dataset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    assert work_dir(FULL) == tmp_path / "datasets/processed/work"
