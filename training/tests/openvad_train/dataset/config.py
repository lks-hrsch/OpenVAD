# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from openvad_train.dataset.config import FULL, PILOT, SLICE, dataset_root, work_dir


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


def test_slice_is_swc_german_demand_with_every_codec():
    assert (SLICE.sources, SLICE.noise, SLICE.languages) == (("swc",), ("demand",), ("de",))
    assert SLICE.codec_shares == FULL.codec_shares
    assert SLICE.codec_bitrate_kbps == FULL.codec_bitrate_kbps
    assert sum(SLICE.condition_shares.values()) == pytest.approx(1.0)


def test_slice_has_the_full_chain_and_the_transients():
    chain = ("room_p", "handset_p", "agc_p", "clip_p", "suppression_p", "transient_p")
    assert [getattr(SLICE, p) for p in chain] == [getattr(FULL, p) for p in chain]
    assert SLICE.transient_p == 0.3


def test_dataset_root_follows_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    assert dataset_root(FULL) == tmp_path / "datasets/processed/openvad-v1"


def test_work_dir_is_per_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    assert work_dir(FULL) == tmp_path / "datasets/processed/work/openvad-v1"
    assert work_dir(SLICE) == tmp_path / "datasets/processed/work/openvad-slice"


def test_the_pilot_is_common_voice_alone_in_its_own_work_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    assert (PILOT.name, PILOT.sources) == ("openvad-pilot", ("common-voice",))
    assert set(PILOT.languages or ()) >= {"en", "de", "it", "ca"}  # the hand check's four
    assert work_dir(PILOT) == tmp_path / "datasets/processed/work/openvad-pilot"
