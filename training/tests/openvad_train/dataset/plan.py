# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import polars as pl
import pytest

from as_is_support import make_ami_tree, make_callhome_tree, make_librivad_tree, make_voices_tree
from demand_support import make_demand_tree
from mfa_source_support import make_sc_tree
from noise_support import make_musan_tree, make_wham_tree, talker
from openvad_train.dataset import __main__ as cli
from openvad_train.dataset import align, index
from openvad_train.dataset import plan as plan_module
from openvad_train.dataset.channel import RirPool
from openvad_train.dataset.config import DEMAND_CATEGORIES, FULL, SLICE, work_dir
from openvad_train.dataset.cutting import Chunk
from openvad_train.dataset.index import run_index
from openvad_train.dataset.labels import rasterise
from openvad_train.dataset.noise import NoisePool, TalkerPool, demand_channels, musan_noise_categories
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.plan import assign_shards, clip_rows, draw_sample, run_plan
from openvad_train.dataset.schema import Clip, Timing, rank, sample_seed
from openvad_train.fetch.common import data_dir
from rir_support import make_rir_tree
from swc_support import add_article, make_swc_tree


Split = Literal["train", "validation", "test"]
SPEC_COLUMNS = [
    "sample_id",
    "audio_filepath",
    "audio_format",
    "byte_offset",
    "byte_length",
    "offset",
    "duration",
    "row_in_file",
    "labels",
    "band",
    "in_small",
    "rank",
    "source",
    "language",
    "speaker",
    "licence",
    "treatment",
    "speech_ratio",
    "condition",
    "noise_source_ids",
    "noise_offsets",
    "snr_db",
    "tir_db",
    "baseline_snr_db",
    "rir_id",
    "handset",
    "agc",
    "clip_drive_db",
    "suppression",
    "codec",
    "bitrate",
    "codec_delay_ms",
    "transients",
    "gap_ms",
    "dither_dbfs",
    "segments",
    "aligner",
    "seed",
]


def clip(id="c", source="common-voice", split: Split = "train", duration_s=2.0, offset_s=0.0, reserved=False) -> Clip:
    path = f"datasets/raw/{source}/{id}.ogg"
    return Clip(id, source, "de", "s", split, path, offset_s, duration_s, "CC0", None, reserved)


def timing(id="c", speech=((100, 1750),), mask=()) -> Timing:
    return Timing(id, "word", list(speech), list(mask), None, "t")


def words(n: int, length: int = 500, gap: int = 300, first: int = 1000) -> list[tuple[int, int]]:
    return [(first + i * (length + gap), first + i * (length + gap) + length) for i in range(n)]


def rows_of(cfg, n=2000, **clip_args):
    return [draw_sample(clip(f"c{i}", **clip_args), timing(f"c{i}"), cfg) for i in range(n)]


def share(rows, column, value) -> float:
    return sum(r[column] == value for r in rows) / len(rows)


def share_set(rows, column) -> float:
    return sum(r[column] is not None for r in rows) / len(rows)


def test_plan_is_independent_of_other_clips():
    a = draw_sample(clip("x"), timing("x"), SLICE)
    [draw_sample(clip(f"y{i}"), timing(f"y{i}"), SLICE) for i in range(10)]
    assert draw_sample(clip("x"), timing("x"), SLICE) == a


def test_condition_shares_converge():
    rows = rows_of(FULL, 20000)
    assert share(rows, "condition", "clean") == pytest.approx(0.15, abs=0.01)
    assert share(rows, "condition", "competing") == pytest.approx(0.15, abs=0.01)
    assert share(rows, "condition", "office") == pytest.approx(0.7 / 9, abs=0.01)


def test_snr_only_from_grid():
    rows = rows_of(FULL)
    assert {r["snr_db"] for r in rows if r["snr_db"] is not None} == {-5, 0, 5, 10, 15, 20}


def test_clean_padding_is_dither_in_range():
    clean = [r for r in rows_of(FULL) if r["condition"] == "clean"]
    assert clean
    assert all(-90 <= r["dither_dbfs"] <= -70 and r["snr_db"] is None for r in clean)


def test_noisy_rows_have_an_snr_and_no_dither():
    noisy = [r for r in rows_of(SLICE) if r["condition"] != "clean"]
    assert noisy
    assert all(r["snr_db"] is not None and r["dither_dbfs"] is None for r in noisy)


def test_row_has_every_spec_column_and_the_plan_columns():
    row = draw_sample(clip(), timing(), SLICE)
    assert set(row) >= {*SPEC_COLUMNS, "split", "speech_offset_n", "total_n", "speech_ms", "mask_ms", "shard"}


def test_ids_and_seed_come_from_the_sample_id():
    row = draw_sample(clip("x"), timing("x"), SLICE)
    assert row["sample_id"] == "common-voice/x"
    assert row["rank"] == rank("common-voice/x")
    assert row["seed"] == sample_seed(SLICE.global_seed, SLICE.plan_version, "common-voice/x")


def test_another_plan_version_changes_the_draws():
    rows = rows_of(SLICE, 50)
    moved = rows_of(dataclasses.replace(SLICE, plan_version="v2"), 50)
    assert [r["speech_offset_n"] for r in rows] != [r["speech_offset_n"] for r in moved]


def test_clean_foreground_is_a_constructed_row_with_null_pointers():
    row = draw_sample(clip(split="test"), timing(), SLICE)
    assert row["treatment"] == "augmented"
    assert row["audio_format"] == "flac_tar"
    assert row["audio_filepath"] is None
    assert row["byte_offset"] is None
    assert row["byte_length"] is None
    assert row["offset"] == 0.0
    assert row["duration"] == row["total_n"] / 16000
    assert row["source_path"] == "datasets/raw/common-voice/c.ogg"
    assert row["split"] == "test"
    assert row["gap_ms"] == SLICE.gap_ms
    assert row["aligner"] == "t"


def test_padding_reaches_a_third_speech_and_the_speech_fits_at_a_whole_millisecond():
    rows = rows_of(SLICE, 300)
    assert all(r["total_n"] == 5 * 16000 and r["source_n"] == 2 * 16000 for r in rows)
    assert all(0 <= r["speech_offset_n"] <= 3 * 16000 and r["speech_offset_n"] % 16 == 0 for r in rows)
    assert max(r["speech_offset_n"] for r in rows) > 2 * 16000
    assert rows[0]["speech_ratio"] == pytest.approx(0.33)
    assert rows[0]["speech_ms"] == [(100, 1750)]


def test_natural_clip_edges_count_towards_the_padding_and_a_long_clip_is_not_shortened():
    row = draw_sample(clip(duration_s=8.0), timing(), SLICE)
    assert row["total_n"] == row["source_n"] == 8 * 16000
    assert row["speech_offset_n"] == 0


def test_intervals_past_the_clip_end_are_clipped():
    row = draw_sample(clip(duration_s=1.0), timing(speech=[(500, 1200), (1500, 1600)], mask=[(900, 1100)]), SLICE)
    assert row["speech_ms"] == [(500, 1000)]
    assert row["mask_ms"] == [(900, 1000)]


def test_a_chunk_is_padded_to_10_to_20_seconds_and_rebased():
    c = clip("a", "swc", duration_s=100.0, offset_s=1.0)
    t = timing("a", speech=[(50_000, 51_000)], mask=[(49_950, 50_000), (60_000, 61_000)])
    row = draw_sample(c, t, SLICE, chunk=Chunk(49_900, 51_100))
    assert row["sample_id"] == "swc/a@49900"
    assert row["rank"] == rank("swc/a@49900")
    assert row["speech_ms"] == [(100, 1100)]
    assert row["mask_ms"] == [(50, 100)]
    assert row["source_offset_n"] == 16000 + 49_900 * 16
    assert row["source_n"] == 1200 * 16
    assert row["total_n"] == 10 * 16000


def test_a_chunk_total_follows_its_speech_up_to_20_seconds():
    c = clip("a", "swc", duration_s=100.0)
    middle = draw_sample(c, timing("a", speech=[(0, 4950)]), SLICE, chunk=Chunk(0, 5000))
    full = draw_sample(c, timing("a", speech=[(0, 9000)]), SLICE, chunk=Chunk(0, 9000))
    assert middle["total_n"] == 15 * 16000
    assert full["total_n"] == 20 * 16000


def test_a_chunk_total_never_drops_below_its_span():
    c = clip("a", "swc", duration_s=100.0)
    row = draw_sample(c, timing("a", speech=[(0, 30_000)]), SLICE, chunk=Chunk(0, 30_000))
    assert row["total_n"] == row["source_n"] == 30 * 16000
    assert row["speech_offset_n"] == 0


def test_clip_rows_cuts_only_the_sources_that_cut():
    long = timing("a", speech=words(30))
    swc = clip_rows(clip("a", "swc", duration_s=40.0), long, SLICE)
    assert [r["sample_id"] for r in swc] == ["swc/a@900", "swc/a@11300", "swc/a@21700"]
    voice = clip_rows(clip("a", duration_s=40.0), long, SLICE)
    assert [r["sample_id"] for r in voice] == ["common-voice/a"]


def test_chunks_of_one_clip_draw_independently():
    rows = clip_rows(clip("a", "swc", duration_s=40.0), timing("a", speech=words(30)), SLICE)
    assert len({r["seed"] for r in rows}) == 3


def test_last_chunk_stops_at_the_clip_end():
    row = clip_rows(clip("a", "swc", duration_s=2.05), timing("a", speech=[(1000, 2000)]), SLICE)[0]
    assert row["source_offset_n"] == 900 * 16
    assert row["source_n"] == (2050 - 900) * 16


def test_reserved_clips_never_give_rows():
    assert clip_rows(clip(reserved=True), timing(), SLICE) == []
    assert clip_rows(clip("a", "voxpopuli", reserved=True), timing("a"), SLICE) == []


def test_a_fully_masked_or_speechless_sample_gives_no_row():
    assert clip_rows(clip("a", "swc"), timing("a", speech=[]), SLICE) == []
    assert clip_rows(clip("a"), timing("a", speech=[]), SLICE) == []
    masked = timing("a", speech=[(500, 1500)], mask=[(0, 2000)])
    assert clip_rows(clip("a", "swc"), masked, SLICE) == []
    assert clip_rows(clip("a"), masked, SLICE) == []


def test_a_chunk_under_200_ms_of_speech_gives_no_row_and_is_counted():
    # A long pause cuts both words off into chunks of their own.
    short = timing("a", speech=[(1000, 1199), (30_000, 30_200)])
    counts: Counter = Counter()
    rows = clip_rows(clip("a", "swc", duration_s=40.0), short, SLICE, counts=counts)
    assert [r["sample_id"] for r in rows] == ["swc/a@29900"]
    assert counts == {"short_chunks": 1}


def test_a_short_clip_of_a_source_that_does_not_cut_stays():
    assert len(clip_rows(clip("a"), timing("a", speech=[(500, 600)]), SLICE)) == 1


def test_a_fully_masked_chunk_is_counted_apart():
    masked = timing("a", speech=[(1000, 2000), (30_000, 31_000)], mask=[(0, 3000)])
    counts: Counter = Counter()
    rows = clip_rows(clip("a", "swc", duration_s=40.0), masked, SLICE, counts=counts)
    assert len(rows) == 1
    assert counts == {"masked_chunks": 1}


def test_a_masked_chunk_goes_and_its_neighbour_stays():
    t = timing("a", speech=[(1000, 8000), (10_000, 17_000)], mask=[(9000, 18_000)])
    assert [r["sample_id"] for r in clip_rows(clip("a", "swc", duration_s=30.0), t, SLICE)] == ["swc/a@900"]


def test_ablating_g_changes_only_the_gap_column():
    t = timing("a", speech=words(30, gap=150))
    base = clip_rows(clip("a", "swc", duration_s=40.0), t, SLICE)
    other = clip_rows(clip("a", "swc", duration_s=40.0), t, dataclasses.replace(SLICE, gap_ms=100))
    assert len(base) == 3  # no gap is over G, so these are budget splits of one stretch
    assert [r | {"gap_ms": 0} for r in base] == [r | {"gap_ms": 0} for r in other]
    assert other[0]["gap_ms"] == 100


@pytest.fixture
def pool(tmp_path, monkeypatch) -> NoisePool:
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    return NoisePool(make_demand_tree(tmp_path / "datasets/raw/demand"))


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_noise_comes_from_the_split_channels_as_a_relative_path(pool, split):
    rows = [draw_sample(clip(f"c{i}", split=split), timing(f"c{i}"), SLICE, pool) for i in range(200)]
    noisy = [r for r in rows if r["condition"] in DEMAND_CATEGORIES]
    assert noisy
    for r in noisy:
        [path], [offset] = r["noise_source_ids"], r["noise_offsets"]
        assert path.startswith("datasets/raw/demand/")
        assert 0 <= offset < 1600
        assert int(path[-6:-4]) in demand_channels(split)
    assert all(r["noise_source_ids"] == [] for r in rows if r["condition"] == "clean")


@pytest.fixture
def rirs(tmp_path, monkeypatch) -> RirPool:
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    return RirPool(make_rir_tree(tmp_path / "datasets/raw/rirs"))


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_a_room_row_carries_a_rir_of_its_split_as_a_relative_path(rirs, split):
    rows = [draw_sample(clip(f"c{i}", split=split), timing(f"c{i}"), FULL, rirs=rirs) for i in range(300)]
    assert {r["rir_id"] is not None for r in rows if r["room"]} == {True}
    assert all(r["rir_id"] is None for r in rows if not r["room"])
    for r in rows:
        if r["room"]:
            assert r["rir_id"].startswith("datasets/raw/rirs/")
            assert rirs.room_of(data_dir() / r["rir_id"]) in rirs.rooms(split)


def test_rooms_are_disjoint_between_splits_in_the_plan(rirs):
    rooms = {
        split: {
            rirs.room_of(data_dir() / r["rir_id"])
            for i in range(300)
            if (r := draw_sample(clip(f"c{i}", split=split), timing(f"c{i}"), FULL, rirs=rirs))["room"]
        }
        for split in ("train", "validation", "test")
    }
    assert rooms["train"].isdisjoint(rooms["test"])
    assert rooms["train"].isdisjoint(rooms["validation"])
    assert rooms["validation"].isdisjoint(rooms["test"])
    assert all(rooms.values())


def test_the_rir_draw_does_not_move_the_other_draws(rirs):
    with_rirs = draw_sample(clip(), timing(), FULL, rirs=rirs)
    without = draw_sample(clip(), timing(), FULL)
    assert with_rirs | {"rir_id": None} == without


def test_a_row_without_a_room_draws_no_rir(rirs):
    cfg = dataclasses.replace(FULL, room_p=0.0)
    assert {draw_sample(clip(f"c{i}"), timing(f"c{i}"), cfg, rirs=rirs)["rir_id"] for i in range(100)} == {None}


def test_as_is_rows_skip_the_rooms(rirs):
    row = draw_sample(clip("a", "voxpopuli"), timing("a"), FULL, rirs=rirs)
    assert (row["room"], row["rir_id"], row["codec"]) == (False, None, "none")


def test_a_row_with_the_pool_repeats_itself(pool):
    assert draw_sample(clip(), timing(), SLICE, pool) == draw_sample(clip(), timing(), SLICE, pool)


def test_the_pool_does_not_move_the_other_draws(pool):
    with_pool = draw_sample(clip(), timing(), SLICE, pool)
    without = draw_sample(clip(), timing(), SLICE)
    assert with_pool | {"noise_source_ids": [], "noise_offsets": []} == without


def test_chain_probabilities_and_ranges():
    rows = rows_of(FULL, 5000)
    assert share(rows, "room", True) == pytest.approx(0.5, abs=0.03)
    assert share_set(rows, "handset") == pytest.approx(0.7, abs=0.03)
    assert share_set(rows, "agc") == pytest.approx(0.5, abs=0.03)
    assert share_set(rows, "clip_drive_db") == pytest.approx(0.15, abs=0.03)
    assert share_set(rows, "suppression") == pytest.approx(0.3, abs=0.03)
    assert all(r["rir_id"] is None for r in rows)
    for r in rows:
        if h := r["handset"]:
            assert 100 <= h["highpass_hz"] <= 300
            assert 3400 <= h["lowpass_hz"] <= 7000
            assert len(h["peaks"]) == 2
            assert all(
                -6 <= p["gain_db"] <= 6 and h["highpass_hz"] <= p["freq_hz"] <= h["lowpass_hz"] for p in h["peaks"]
            )
        if a := r["agc"]:
            assert -26 <= a["target_dbfs"] <= -16
            assert 5 <= a["attack_ms"] <= 20
            assert 100 <= a["release_ms"] <= 500
        if r["clip_drive_db"] is not None:
            assert 3 <= r["clip_drive_db"] <= 12
    assert {r["suppression"] for r in rows} == {None, "wiener", "spectral_subtraction"}


def test_codec_shares_bitrates_and_bands():
    rows = rows_of(FULL, 5000)
    for codec, expected in FULL.codec_shares.items():
        assert share(rows, "codec", codec) == pytest.approx(expected, abs=0.03)
    assert {r["band"] for r in rows if r["codec"] == "amr_nb"} == {"narrow_leg"}
    assert {r["band"] for r in rows if r["codec"] != "amr_nb"} == {"wide"}
    assert {r["bitrate"] for r in rows if r["codec"] == "none"} == {None}
    assert {r["bitrate"] for r in rows if r["codec"] == "g722"} == {64.0}
    wideband_modes = {6.6, 8.85, 12.65, 14.25, 15.85, 18.25, 19.85, 23.05, 23.85}
    assert {r["bitrate"] for r in rows if r["codec"] == "amr_wb"} == wideband_modes
    assert {r["bitrate"] for r in rows if r["codec"] == "amr_nb"} == {4.75, 5.15, 5.9, 6.7, 7.4, 7.95, 10.2, 12.2}
    assert all(12 <= r["bitrate"] <= 32 for r in rows if r["codec"] == "opus")


def test_an_amr_range_without_a_mode_is_an_error_not_a_uniform_draw():
    cfg = dataclasses.replace(FULL, codec_bitrate_kbps={"amr_nb": (13.0, 14.0)})
    with pytest.raises(ValueError, match="no mode"):
        plan_module._draw_bitrate("amr_nb", cfg, np.random.default_rng(0))


def test_slice_draws_every_codec_the_chain_and_the_transients_but_only_its_conditions():
    rows = rows_of(SLICE, 500)
    assert {r["codec"] for r in rows} == set(FULL.codec_shares)
    assert {r["condition"] for r in rows} == {"clean", *DEMAND_CATEGORIES}
    assert share(rows, "room", True) == pytest.approx(0.5, abs=0.06)
    assert share_set(rows, "handset") == pytest.approx(0.7, abs=0.06)
    assert sum(bool(r["transients"]) for r in rows) / len(rows) == pytest.approx(0.3, abs=0.06)
    assert SLICE.transient_p == FULL.transient_p


def test_transients_share_and_ranges():
    rows = rows_of(FULL, 5000)
    assert sum(bool(r["transients"]) for r in rows) / len(rows) == pytest.approx(0.3, abs=0.03)
    events = [e for r in rows for e in r["transients"]]
    assert {e["kind"] for e in events} == {"click", "dtmf"}
    assert all(-18 <= e["peak_db"] <= 0 and 0 <= e["offset_n"] < 5 * 16000 for e in events)


def test_as_is_row_points_into_raw_and_skips_every_draw():
    c = clip("a", "voxpopuli", duration_s=12.5, offset_s=3.0)
    row = draw_sample(c, timing("a", speech=[(0, 1000)]), SLICE)
    assert row["treatment"] == "as_is"
    assert row["audio_format"] == "file"
    assert row["audio_filepath"] == "datasets/raw/voxpopuli/a.ogg"
    assert row["offset"] == 3.0
    assert row["duration"] == 12.5
    assert row["speech_offset_n"] == 0
    assert row["total_n"] == row["source_n"] == 12.5 * 16000
    assert row["speech_ratio"] == pytest.approx(0.08)
    assert row["condition"] == "as_is"
    assert row["codec"] == "none"
    assert row["band"] == "wide"
    assert row["snr_db"] is None
    assert row["dither_dbfs"] is None
    assert row["noise_source_ids"] == []
    assert row["handset"] is None
    assert row["agc"] is None
    assert row["room"] is False
    assert row["transients"] == []


def test_frame_labels_are_used_as_they_are_so_their_gap_is_zero():
    frames = Timing("a", "frame", [(300, 1000), (1100, 2700)], [], None, "t")
    assert draw_sample(clip("a", "librivad"), frames, SLICE)["gap_ms"] == 0
    assert draw_sample(clip("a", "ami"), timing("a"), SLICE)["gap_ms"] == SLICE.gap_ms


def test_as_is_row_without_speech_stays():
    assert len(clip_rows(clip("a", "speech-commands"), timing("a", speech=[]), SLICE)) == 1


def test_a_voxpopuli_clip_over_30_s_is_cut_at_gaps_into_samples_of_at_most_30_s():
    c = clip("vp", "voxpopuli", duration_s=70.0, offset_s=100.0)
    t = timing("vp", speech=words(80, length=500, gap=300, first=1000))  # words to 65 s, then 5 s of silence
    rows = clip_rows(c, t, SLICE)
    assert len(rows) >= 3
    assert all(r["treatment"] == "as_is" and r["duration"] <= 30.0 for r in rows)
    spans = sorted((r["offset"], r["offset"] + r["duration"]) for r in rows)
    assert spans[0][0] == 100.0
    assert spans[-1][1] == pytest.approx(170.0)
    assert all(a[1] == pytest.approx(b[0]) for a, b in pairwise(spans))
    # Every cut is in a gap: no word crosses a sample's edge.
    for r in rows:
        assert all(s >= 0 and e <= round(r["duration"] * 1000) for s, e in r["speech_ms"])
    assert sum(len(r["speech_ms"]) for r in rows) == 80
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert all(r["source_offset_n"] == round(r["offset"] * 16000) for r in rows)
    assert all(r["source_n"] == round(r["duration"] * 16000) for r in rows)


def test_a_long_silence_in_a_cut_clip_is_cut_too_so_no_sample_passes_30_s():
    c = clip("m", "voxpopuli", duration_s=400.0)
    rows = clip_rows(c, timing("m", speech=[(100_000, 101_000), (250_000, 251_000)]), SLICE)
    assert all(r["duration"] <= 30.0 for r in rows)
    assert sum(r["duration"] for r in rows) == pytest.approx(400.0)
    assert sum(len(r["speech_ms"]) for r in rows) == 2


def test_a_voxpopuli_clip_of_30_s_or_less_stays_one_sample():
    rows = clip_rows(clip("vp", "voxpopuli", duration_s=29.0), timing("vp", speech=words(40)), SLICE)
    assert [(r["offset"], r["duration"]) for r in rows] == [(0.0, 29.0)]


def plan_frame(sources: list[str], n: int = 300) -> pl.DataFrame:
    splits: tuple[Split, ...] = ("train", "validation", "test")
    rows = [
        draw_sample(clip(f"c{i}", source, splits[i % 3]), timing(f"c{i}"), SLICE)
        for source in sources
        for i in range(n)
    ]
    return pl.DataFrame(rows, schema=plan_module.PLAN_SCHEMA)


def test_only_augmented_rows_get_shards_in_rank_order():
    planned = assign_shards(plan_frame(sources=["swc", "voxpopuli"]), bytes_per_hour=63e6)
    assert (planned["shard"].is_null() == (planned["treatment"] == "as_is")).all()
    built = planned.filter(pl.col("shard").is_not_null() & (pl.col("split") == "train"))
    assert built.height > 0
    assert built["rank"].to_list() == sorted(built["rank"].to_list())


def test_shards_are_numbered_per_split_and_rise_with_rank():
    # 5 s samples at 3600 MB per hour are 5 MB each: 4 fit in 20 MB.
    planned = assign_shards(plan_frame(sources=["swc"]), bytes_per_hour=3600e6, target_bytes=20_000_000)
    for split in ("train", "validation", "test"):
        shards = planned.filter(pl.col("split") == split)["shard"].to_list()
        assert shards[:5] == [f"{split}/00000"] * 4 + [f"{split}/00001"]
        assert shards == sorted(shards)
        assert max(Counter(shards).values()) == 4


def test_a_sample_over_the_target_has_its_own_shard_and_numbers_stay_dense():
    planned = assign_shards(plan_frame(sources=["swc"], n=9), bytes_per_hour=3600e6, target_bytes=1)
    for split in ("train", "validation", "test"):
        shards = planned.filter(pl.col("split") == split)["shard"].to_list()
        assert shards == [f"{split}/00000", f"{split}/00001", f"{split}/00002"]


def test_as_is_rows_do_not_fill_a_shard():
    mixed = plan_frame(sources=["swc", "voxpopuli"])
    planned = assign_shards(mixed, bytes_per_hour=3600e6, target_bytes=20_000_000)
    assert max(Counter(planned["shard"].drop_nulls().to_list()).values()) == 4


def test_assign_shards_gives_a_frame_with_the_same_columns():
    frame = plan_frame(sources=["swc"], n=3)
    assert assign_shards(frame, bytes_per_hour=63e6).schema == frame.schema


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    make_swc_tree(tmp_path / "datasets/raw/swc")
    make_demand_tree(tmp_path / "datasets/raw/demand")
    make_rir_tree(tmp_path / "datasets/raw/rirs")
    return tmp_path


CONFIG = dataclasses.replace(SLICE, languages=None)


def plan_of(data) -> pl.DataFrame:
    return asyncio.run(read_parquet(data / "datasets/processed/work/openvad-slice/plan/plan.parquet"))


def test_run_plan_writes_one_row_per_clip_in_rank_order_with_a_manifest(data, capsys):
    run_index(CONFIG)
    run_plan(CONFIG)
    df = plan_of(data)
    assert df.height == 5
    assert df["sample_id"].to_list() == [s for _, s in sorted((rank(s), s) for s in df["sample_id"].to_list())]
    assert df.schema["rank"] == pl.UInt64
    assert df.schema["seed"] == pl.UInt64
    assert df["shard"].null_count() == 0
    assert df["audio_filepath"].null_count() == 5
    assert (df["source_path"].str.starts_with("datasets/raw/swc/")).all()
    manifest = json.loads((data / "datasets/processed/work/openvad-slice/plan/manifest.json").read_text())
    assert manifest["samples"] == 5
    assert manifest["plan_version"] == "v1"
    assert manifest["per_source"]["swc"] == {
        "clips": 5,
        "reserved": 0,
        "without_sample": 0,
        "short_chunks": 0,
        "masked_chunks": 0,
        "samples": 5,
    }
    first = df.row(0, named=True)
    assert f"swc {first['split']} " in capsys.readouterr().out


def test_the_parquet_schema_names_exactly_the_row_columns():
    assert set(plan_module.PLAN_SCHEMA) == set(draw_sample(clip(), timing(), SLICE))


def test_a_row_read_back_feeds_the_labels(data):
    run_index(CONFIG)
    run_plan(CONFIG)
    row = plan_of(data).row(0, named=True)
    assert row["speech_ms"][0] == [100, 300]
    labels = rasterise(row["speech_ms"], row["mask_ms"], row["total_n"], row["speech_offset_n"] // 16)
    assert labels.sum() > 0


def test_batching_does_not_change_the_plan(data, monkeypatch):
    run_index(CONFIG)
    run_plan(CONFIG)
    first = plan_of(data)
    monkeypatch.setattr(plan_module, "_ROWS_PER_FRAME", 2)
    monkeypatch.setattr(plan_module, "_CLIPS_PER_BATCH", 2)
    run_plan(CONFIG)
    assert plan_of(data).equals(first)


def test_timings_out_of_clip_order_are_refused(data, monkeypatch):
    run_index(CONFIG)
    real = plan_module.SOURCES["swc"]
    backwards = SimpleNamespace(MFA_ALIGNED=False, timings=lambda raw, clips: reversed(list(real.timings(raw, clips))))
    monkeypatch.setitem(plan_module.SOURCES, "swc", backwards)
    with pytest.raises(ValueError, match="clip order"):
        run_plan(CONFIG)


def test_a_rerun_gives_the_same_plan(data):
    run_index(CONFIG)
    run_plan(CONFIG)
    first = plan_of(data)
    run_plan(CONFIG)
    assert plan_of(data).equals(first)


def test_noise_rows_on_real_files_name_a_file_in_the_tree(data):
    run_index(CONFIG)
    run_plan(CONFIG)
    for row in plan_of(data).iter_rows(named=True):
        if row["condition"] == "clean":
            assert row["noise_source_ids"] == []
            continue
        [path] = row["noise_source_ids"]
        assert (data / path).is_file()


def test_run_plan_leaves_reserved_clips_out(data):
    run_index(CONFIG)
    path = data / "datasets/processed/work/openvad-slice/index/swc.parquet"
    frame = asyncio.run(read_parquet(path)).with_columns(reserved=pl.col("clip_id") == "english/Aaa")
    asyncio.run(write_parquet({path: frame.lazy()}))
    run_plan(CONFIG)
    assert plan_of(data).height == 4
    assert "swc/english/Aaa@0" not in plan_of(data)["sample_id"].to_list()


def test_limit_keeps_the_lowest_ranks_by_count(data):
    run_index(CONFIG)
    run_plan(CONFIG)
    everything = plan_of(data)["sample_id"].to_list()
    run_plan(CONFIG, limit="2")
    assert plan_of(data)["sample_id"].to_list() == everything[:2]


def test_limit_keeps_the_lowest_ranks_by_hours(data):
    run_index(CONFIG)
    run_plan(CONFIG)
    everything = plan_of(data)
    hours = float(everything["duration"].head(2).sum()) / 3600
    run_plan(CONFIG, limit=f"{hours * 1.01}h")
    assert plan_of(data)["sample_id"].to_list() == everything["sample_id"].to_list()[:2]


def test_max_samples_keeps_the_lowest_ranks_and_redraws_nothing(data):
    run_index(CONFIG)
    run_plan(CONFIG)
    everything = plan_of(data)
    run_plan(dataclasses.replace(CONFIG, max_samples=2))
    capped = plan_of(data)
    assert capped.height == 2
    # Shards are assigned after the cap, so compare the drawn columns only.
    assert capped.drop("shard").equals(everything.head(2).drop("shard"))
    manifest = json.loads((data / "datasets/processed/work/openvad-slice/plan/manifest.json").read_text())
    assert (manifest["samples"], manifest["max_samples"]) == (2, 2)


def test_the_slice_caps_its_samples():
    assert SLICE.max_samples == 500
    assert FULL.max_samples is None


@pytest.mark.parametrize("limit", ["abc", "-1", "0", "-2h", "h", "nanh", "infh"])
def test_a_bad_limit_is_refused_before_any_work(data, limit):
    with pytest.raises(ValueError, match="limit"):
        run_plan(SLICE, limit=limit)


def test_a_limit_selecting_no_row_is_refused_and_writes_nothing(data):
    run_index(CONFIG)
    with pytest.raises(ValueError, match=r"limit '0\.0001h' selects no sample"):
        run_plan(CONFIG, limit="0.0001h")
    assert not (data / "datasets/processed/work/openvad-slice/plan/plan.parquet").exists()


def test_manifest_counts_samples_after_the_limit(data):
    run_index(CONFIG)
    run_plan(CONFIG, limit="2")
    manifest = json.loads((data / "datasets/processed/work/openvad-slice/plan/manifest.json").read_text())
    assert manifest["samples"] == 2
    assert manifest["per_source"]["swc"]["samples"] == 2
    assert manifest["per_source"]["swc"]["clips"] == 5


def test_a_condition_whose_noise_set_the_config_lacks_is_refused_before_any_work(data):
    with pytest.raises(ValueError, match="'city' needs 'wham'"):
        run_plan(dataclasses.replace(FULL, noise=("demand", "musan")))
    with pytest.raises(ValueError, match="'office' needs 'demand'"):
        run_plan(dataclasses.replace(SLICE, noise=("wham",), condition_shares={"clean": 0.5, "office": 0.5}))


def test_a_condition_nobody_draws_is_refused_before_any_work(data):
    with pytest.raises(ValueError, match="no draw for condition 'music'"):
        run_plan(dataclasses.replace(SLICE, condition_shares={"clean": 0.5, "music": 0.5}))


def test_talker_conditions_need_reserved_clips_in_the_index(data):
    cfg = dataclasses.replace(CONFIG, condition_shares={"clean": 0.5, "babble": 0.5})
    run_index(cfg)
    index = data / "datasets/processed/work/openvad-slice/index/swc.parquet"
    asyncio.run(write_parquet({index: asyncio.run(read_parquet(index)).with_columns(reserved=pl.lit(False)).lazy()}))
    with pytest.raises(ValueError, match="no reserved clip"):
        run_plan(cfg)


def test_an_mfa_source_without_word_tables_names_them(data, monkeypatch):
    run_index(CONFIG)
    monkeypatch.setitem(plan_module.SOURCES, "swc", SimpleNamespace(MFA_ALIGNED=True))
    with pytest.raises(FileNotFoundError, match="words/swc"):
        run_plan(CONFIG)


def test_a_source_without_adapter_is_named(data):
    with pytest.raises(ValueError, match="no index adapter for source 'simsamu'"):
        run_plan(dataclasses.replace(SLICE, sources=("simsamu",)))


def test_a_repeated_sample_id_is_refused(data, monkeypatch):
    run_index(CONFIG)
    monkeypatch.setattr(
        plan_module,
        "clip_rows",
        lambda c, t, cfg, pool, counts, rirs, talkers: [draw_sample(c, t, cfg, pool, Chunk(0, 1))] * 2,
    )
    with pytest.raises(ValueError, match="repeated"):
        run_plan(CONFIG)


def test_cli_passes_config_and_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(plan_module, "run_plan", lambda cfg, limit: seen.append((cfg.name, limit)))
    plan_module.main(["--config", "slice", "--limit", "1000h"])
    plan_module.main(["--config", "full"])
    assert seen == [("openvad-slice", "1000h"), ("openvad-v1", None)]


def test_plan_is_a_registered_stage():
    assert cli.STAGES["plan"] is plan_module.main


# ---------------------------------------------------------------------------------------------------- MFA sources


@pytest.fixture
def mfa_data(data, monkeypatch):
    monkeypatch.setitem(index.SOURCES, "fake", SimpleNamespace(MFA_ALIGNED=True, SOURCE="fake"))
    return data


MFA_CONFIG = dataclasses.replace(SLICE, name="mfa-test", sources=("fake",), languages=None)


def fake_clip(clip_id, speaker="x", language="de", hours=1.0, **kw) -> Clip:
    return Clip(
        clip_id, "fake", language, speaker, "train", f"datasets/raw/fake/{clip_id}.wav", 0.0, hours * 3600, "CC0",
        "a b", **kw,
    )  # fmt: skip


def hour_of_speech(clip_id, speech_ms=3_600_000) -> Timing:
    return Timing(clip_id, "word", [(0, speech_ms)], [], {"n_words": 1}, "mfa")


def write_mfa_tables(cfg, source, clips, kept, dropped=(), reserved=(), speech_ms=3_600_000):
    """The index of `clips` and, per language, one words table of the clip IDs in `kept` and one drop table."""
    out = work_dir(cfg) / "index"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({out / f"{source}.parquet": index._frame(clips, set(reserved))}))
    words = work_dir(cfg) / "words" / source
    for language in sorted({c.language for c in clips}):
        folder = words / language
        (folder / "dropped").mkdir(parents=True, exist_ok=True)
        mine = [hour_of_speech(c.clip_id, speech_ms) for c in clips if c.language == language and c.clip_id in kept]
        drops = [
            align.Drop(c.clip_id, "low_loglik", None) for c in clips if c.clip_id in dropped and c.language == language
        ]
        asyncio.run(
            write_parquet(
                {
                    folder / "00000.parquet": align._table(mine),
                    folder / "dropped" / "00000.parquet": align._drop_table(source, language, "00000", drops),
                }
            )
        )


def test_a_speaker_with_25_h_in_each_of_two_languages_keeps_30_h_in_plan(mfa_data):
    clips = [fake_clip(f"{lang}{i}", language=lang) for lang in ("de", "en") for i in range(25)]
    clips += [fake_clip(f"other{i}", speaker="y") for i in range(2)]
    write_mfa_tables(MFA_CONFIG, "fake", clips, kept={c.clip_id for c in clips})
    run_plan(MFA_CONFIG)
    df = asyncio.run(read_parquet(work_dir(MFA_CONFIG) / "plan/plan.parquet"))
    mine = df.filter(pl.col("speaker") == "x")
    assert mine.height == 30
    ranked = sorted((c.clip_id for c in clips if c.speaker == "x"), key=lambda i: (rank(i), i))
    assert set(mine["clip_id"]) == set(ranked[:30])
    assert set(mine["language"]) == {"de", "en"}  # the cap walks both languages together
    assert df.filter(pl.col("speaker") == "y").height == 2
    manifest = json.loads((work_dir(MFA_CONFIG) / "plan/manifest.json").read_text())
    assert manifest["per_source"]["fake"]["over_cap"] == 20


def test_the_cap_counts_audio_seconds_as_align_does_not_speech_seconds(mfa_data):
    # Each 1 h clip holds 30 min of speech: 50 of them are 25 h of speech but 50 h of audio, and align's cap of
    # 30 h of audio would have stopped at 30 clips, so plan must not keep more.
    clips = [fake_clip(f"{lang}{i}", language=lang) for lang in ("de", "en") for i in range(25)]
    write_mfa_tables(MFA_CONFIG, "fake", clips, kept={c.clip_id for c in clips}, speech_ms=1_800_000)
    run_plan(MFA_CONFIG)
    df = asyncio.run(read_parquet(work_dir(MFA_CONFIG) / "plan/plan.parquet"))
    assert df.height == 30
    assert df["speech_ms"].list.first().list.get(1).unique().to_list() == [1_800_000]


def test_clips_without_a_surviving_timing_are_skipped_and_counted_by_reason(mfa_data):
    clips = [
        fake_clip("kept", speaker="a"),
        fake_clip("dropped", speaker="b"),
        fake_clip("over_align_cap", speaker="c"),
        fake_clip("unaligned_language", speaker="d", language="fr"),
        fake_clip("reserved", speaker="r"),
    ]
    write_mfa_tables(MFA_CONFIG, "fake", clips, kept={"kept"}, dropped={"dropped"}, reserved={("train", "de", "r")})
    (work_dir(MFA_CONFIG) / "words/fake/fr").rename(work_dir(MFA_CONFIG) / "words/fake-fr-removed")
    run_plan(MFA_CONFIG)
    df = asyncio.run(read_parquet(work_dir(MFA_CONFIG) / "plan/plan.parquet"))
    assert df["clip_id"].to_list() == ["kept"]
    per = json.loads((work_dir(MFA_CONFIG) / "plan/manifest.json").read_text())["per_source"]["fake"]
    assert (per["clips"], per["samples"], per["reserved"], per["dropped_by_align"], per["not_aligned"]) == (
        5,
        1,
        1,
        1,
        2,
    )
    assert (per["over_cap"], per["without_sample"]) == (0, 0)


def test_a_drop_table_of_an_unfinished_batch_is_not_counted_as_a_drop(mfa_data):
    clips = [fake_clip("kept", speaker="a"), fake_clip("cut_short", speaker="b")]
    write_mfa_tables(MFA_CONFIG, "fake", clips, kept={"kept"})
    unfinished = [align.Drop("cut_short", "low_loglik", None)]  # batch 00001 has a drop table and no words table
    table = {
        work_dir(MFA_CONFIG) / "words/fake/de/dropped/00001.parquet": align._drop_table(
            "fake", "de", "00001", unfinished
        )
    }
    asyncio.run(write_parquet(table))
    run_plan(MFA_CONFIG)
    per = json.loads((work_dir(MFA_CONFIG) / "plan/manifest.json").read_text())["per_source"]["fake"]
    assert (per["dropped_by_align"], per["not_aligned"]) == (0, 1)


def test_a_row_carries_the_timing_of_its_own_clip_whatever_the_language_order(mfa_data):
    clips = [
        fake_clip(f"{lang}{i}", speaker=f"{lang}{i}", language=lang) for i in range(3) for lang in ("en", "de", "en")
    ]
    clips = list({c.clip_id: c for c in clips}.values())
    write_mfa_tables(MFA_CONFIG, "fake", clips, kept={c.clip_id for c in clips})
    run_plan(MFA_CONFIG)
    df = asyncio.run(read_parquet(work_dir(MFA_CONFIG) / "plan/plan.parquet"))
    assert df.height == len(clips)
    assert df["speech_ms"].to_list() == [[[0, 3_600_000]]] * len(clips)
    assert (df["aligner"] == "mfa").all()


def test_background_noise_comes_out_of_plan_as_non_speech_and_has_no_words_row(data):
    make_sc_tree(data / "datasets/raw/speech-commands")
    (data / "datasets/raw/speech-commands/manifest.json").write_text('{"dataset": "speech-commands"}')
    cfg = dataclasses.replace(SLICE, name="sc-test", sources=("speech-commands",), languages=None)
    run_index(cfg)
    fields = [f.name for f in dataclasses.fields(Clip)]
    clips = [
        Clip(**r)
        for r in asyncio.run(read_parquet(work_dir(cfg) / "index/speech-commands.parquet", fields)).iter_rows(
            named=True
        )
    ]
    spoken = [c for c in clips if c.transcript is not None]
    folder = work_dir(cfg) / "words/speech-commands/en"
    folder.mkdir(parents=True)
    timings = [Timing(c.clip_id, "word", [(100, 900)], [], {"n_words": 1}, "mfa") for c in spoken]
    asyncio.run(write_parquet({folder / "00000.parquet": align._table(timings)}))
    run_plan(cfg)
    df = asyncio.run(read_parquet(work_dir(cfg) / "plan/plan.parquet"))
    noise = df.filter(pl.col("clip_id").str.contains("_background_noise_"))
    assert noise.height == 1
    assert noise["speech_ms"].to_list() == [[]]
    assert noise["speech_ratio"].to_list() == [0.0]
    assert noise["aligner"].to_list() == ["speech-commands:background-noise"]
    assert not any(c.reserved for c in clips if c.transcript is None)  # it feeds no babble
    assert df.height == sum(not c.reserved for c in clips)
    assert df.filter(pl.col("clip_id") != noise["clip_id"][0])["speech_ms"].list.len().min() == 1


# ---------------------------------------------------------------------------------------------------- as-is sources


def test_ami_and_callhome_clips_over_30_s_are_cut_at_gaps_like_voxpopuli():
    for source in ("ami", "callhome"):
        c = clip("m", source, duration_s=70.0)
        rows = clip_rows(c, timing("m", speech=words(80, length=500, gap=300, first=1000)), SLICE)
        assert len(rows) >= 3
        assert all(r["treatment"] == "as_is" and r["duration"] <= 30.0 for r in rows)
        assert sum(len(r["speech_ms"]) for r in rows) == 80


def test_a_callhome_sample_points_at_the_row_of_its_call_and_is_narrow_native():
    c = dataclasses.replace(clip("call", "callhome", duration_s=70.0, offset_s=0.0), row_in_file=7)
    rows = clip_rows(c, timing("call", speech=words(80)), SLICE)
    assert len(rows) >= 3
    assert {(r["audio_format"], r["row_in_file"], r["band"], r["treatment"]) for r in rows} == {
        ("callhome_parquet", 7, "narrow_native", "as_is")
    }
    assert [r["offset"] for r in rows] == sorted(r["offset"] for r in rows)
    assert rows[0]["offset"] == 0.0


def test_every_other_as_is_source_is_read_as_a_file_at_the_wide_band():
    for source in ("ami", "voices", "librivad", "voxpopuli", "speech-commands"):
        (row,) = clip_rows(clip("a", source), timing("a"), SLICE)
        assert (row["audio_format"], row["row_in_file"], row["band"]) == ("file", None, "wide")


@pytest.fixture
def as_is_data(data):
    raw = data / "datasets/raw"
    make_ami_tree(raw / "ami")
    make_callhome_tree(raw / "callhome")
    make_voices_tree(raw / "voices")
    make_librivad_tree(raw / "librivad")
    return data


AS_IS_CONFIG = dataclasses.replace(
    SLICE, name="as-is-test", sources=("ami", "callhome", "voices", "librivad"), languages=None, max_samples=None
)


def test_the_four_sources_used_as_they_are_go_through_index_and_plan(as_is_data):
    run_index(AS_IS_CONFIG)
    run_plan(AS_IS_CONFIG)
    df = asyncio.run(read_parquet(work_dir(AS_IS_CONFIG) / "plan/plan.parquet"))
    assert set(df["treatment"]) == {"as_is"}
    assert df["shard"].null_count() == df.height
    by_source = {s: df.filter(pl.col("source") == s) for s in AS_IS_CONFIG.sources}
    assert set(by_source["callhome"]["audio_format"]) == {"callhome_parquet"}
    assert by_source["callhome"]["row_in_file"].null_count() == 0
    assert set(by_source["callhome"]["band"]) == {"narrow_native"}
    for source in ("ami", "voices", "librivad"):
        assert set(by_source[source]["audio_format"]) == {"file"}
        assert by_source[source]["row_in_file"].null_count() == by_source[source].height
        assert set(by_source[source]["band"]) == {"wide"}
    # LibriVAD labels are used as they are, without the gap rule.
    assert set(by_source["librivad"]["gap_ms"]) == {0}
    assert set(by_source["ami"]["gap_ms"]) == {AS_IS_CONFIG.gap_ms}
    assert (by_source["callhome"]["speech_ms"].list.len() > 0).all()


def test_a_slice_of_languages_gives_every_callhome_clip_the_same_split_in_the_index(as_is_data):
    full = AS_IS_CONFIG
    run_index(dataclasses.replace(full, sources=("callhome",)))
    first = asyncio.run(read_parquet(work_dir(full) / "index/callhome.parquet"))
    run_index(dataclasses.replace(full, sources=("callhome",), languages=("de",)))
    german = asyncio.run(read_parquet(work_dir(full) / "index/callhome.parquet"))
    assert set(german["language"]) == {"de"}
    splits = dict(zip(first["clip_id"], first["split"], strict=True))
    assert all(splits[c] == s for c, s in zip(german["clip_id"], german["split"], strict=True))


def test_librivad_and_voices_reserve_no_speaker_and_plan_keeps_every_clip(as_is_data):
    cfg = dataclasses.replace(AS_IS_CONFIG, sources=("voices", "librivad"))
    run_index(cfg)
    run_plan(cfg)
    for source in cfg.sources:
        indexed = asyncio.run(read_parquet(work_dir(cfg) / f"index/{source}.parquet"))
        assert indexed.height > 0
        assert not indexed["reserved"].any()
        assert indexed["speaker"].null_count() == 0
    df = asyncio.run(read_parquet(work_dir(cfg) / "plan/plan.parquet"))
    assert df.filter(pl.col("source") == "voices").height == 34  # the 36 recordings but the 2 of dev-clean
    assert df.filter(pl.col("source") == "librivad").height == 5 * 4 * 3


def test_ami_keeps_the_default_of_reserving_speakers():
    assert getattr(index.SOURCES["ami"], "RESERVES_SPEAKERS", True) is True


def test_the_index_manifest_records_what_the_as_is_adapters_dropped(as_is_data):
    cfg = dataclasses.replace(AS_IS_CONFIG, sources=("librivad",))
    run_index(cfg)
    manifest = json.loads((work_dir(cfg) / "index/manifest.json").read_text())
    assert manifest["per_source"]["librivad"]["adapter"] == {
        "label_length_mismatch": {"test": 4, "train": 4, "validation": 4},
        "label_one_sample_short": {},
    }


@pytest.fixture
def noise_trees(tmp_path, monkeypatch) -> dict[str, Path]:
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    raw = tmp_path / "datasets/raw"
    return {
        "demand": make_demand_tree(raw / "demand"),
        "wham": make_wham_tree(raw / "wham"),
        "musan": make_musan_tree(raw / "musan"),
    }


@pytest.fixture
def full_pool(noise_trees) -> NoisePool:
    return NoisePool(noise_trees["demand"], noise_trees["wham"], noise_trees["musan"])


def reserved_talkers(split_sizes: dict[str, dict[str, int]]) -> TalkerPool:
    """Reserved clips: `split_sizes[split][language]` speakers each, one clip per speaker."""
    clips = [
        talker(f"{split}-{lang}-{i}", lang, split, source="common-voice")
        for split, languages in split_sizes.items()
        for lang, n in languages.items()
        for i in range(n)
    ]
    return TalkerPool(clips, {("common-voice", lang): n for lang, n in SPEAKERS.items()}, FULL.babble_min_speakers)


# Speakers of each language in the source index, reserved or not: kk is under the 50 of `babble_min_speakers`.
SPEAKERS = {"de": 1000, "ja": 1000, "kk": 10, "fr": 1000}
ALL_SPLITS = {s: {"de": 60, "ja": 60, "kk": 3, "fr": 60} for s in ("train", "validation", "test")}
ONLY = {"clean": 0.0, "competing": 0.0, "babble": 0.0, "ssn": 0.0, "city": 0.0} | dict.fromkeys(DEMAND_CATEGORIES, 0.0)


def only(condition: str):
    """`FULL` with every row of one `condition`."""
    return dataclasses.replace(FULL, condition_shares=ONLY | {condition: 1.0})


def rows_with(cfg, n=200, split: Split = "train", language="de", **kwargs):
    return [
        draw_sample(dataclasses.replace(clip(f"c{i}", split=split), language=language), timing(f"c{i}"), cfg, **kwargs)
        for i in range(n)
    ]


def donors(row, pool_clips: dict[str, Clip]) -> list[Clip]:
    return [pool_clips[p] for p in row["noise_source_ids"]]


@pytest.fixture
def talkers() -> TalkerPool:
    return reserved_talkers(ALL_SPLITS)


def by_path(split_sizes: dict) -> dict[str, Clip]:
    return {
        c.path: c
        for split, languages in split_sizes.items()
        for lang, n in languages.items()
        for c in (talker(f"{split}-{lang}-{i}", lang, split, source="common-voice") for i in range(n))
    }


@pytest.mark.parametrize(("split", "folder"), [("train", "tr"), ("validation", "cv"), ("test", "tt")])
def test_city_noise_is_a_wham_file_of_the_rows_split_with_its_channel(full_pool, split, folder):
    rows = rows_with(only("city"), 60, split, pool=full_pool)
    for r in rows:
        [path], [offset] = r["noise_source_ids"], r["noise_offsets"]
        file, _, channel = path.partition("#")
        assert file.startswith(f"datasets/raw/wham/wham_noise/{folder}/")
        assert channel in ("0", "1")
        assert 0 <= offset < 1600


def test_tt_never_reaches_train(full_pool):
    for split in ("train", "validation"):
        rows = rows_with(only("city"), 100, split, pool=full_pool)
        assert not any("/tt/" in p for r in rows for p in r["noise_source_ids"])


def test_musan_files_reach_the_demand_categories_of_their_table_row(full_pool):
    categories = musan_noise_categories()
    seen = set()
    for category in ("nature", "street"):
        for r in rows_with(only(category), 400, pool=full_pool):
            [path] = r["noise_source_ids"]
            if "/musan/" in path:
                seen.add(category)
                assert categories[path.rpartition("/")[2].removesuffix(".wav")] == category
    assert seen == {"nature", "street"}


def test_babble_has_six_different_reserved_talkers_of_the_rows_language_and_split(talkers):
    clips = by_path(ALL_SPLITS)
    for split in ("train", "validation", "test"):
        rows = rows_with(only("babble"), 60, split, talkers=talkers)
        assert {len(r["noise_source_ids"]) for r in rows} == {6}
        for r in rows:
            picked = donors(r, clips)
            assert {c.split for c in picked} == {split}
            assert {c.language for c in picked} == {"de"}
            assert len({c.speaker for c in picked}) == 6
            assert all(c.reserved for c in picked)
            assert len(r["noise_offsets"]) == 6
            assert all(o == 0 for o in r["noise_offsets"])


def test_babble_offsets_start_inside_a_donor_longer_than_the_sample(talkers):
    long = TalkerPool(
        [dataclasses.replace(talker(f"l{i}", source="common-voice"), duration_s=60.0) for i in range(10)],
        {("common-voice", "de"): 1000},
        1,
    )
    rows = rows_with(only("babble"), 50, talkers=long)
    offsets = [o for r in rows for o in r["noise_offsets"]]
    assert all(0 <= o <= (60 * 16000 - r["total_n"]) for r in rows for o in r["noise_offsets"])
    assert max(offsets) > 16000


def test_babble_japanese_takes_any_language_half_the_time(talkers):
    clips = by_path(ALL_SPLITS)
    rows = rows_with(only("babble"), 600, language="ja", talkers=talkers)
    same = [all(c.language == "ja" for c in donors(r, clips)) for r in rows]
    assert sum(same) / len(same) == pytest.approx(0.5, abs=0.05)


def test_a_small_language_borrows_its_babble_from_every_language(talkers):
    clips = by_path(ALL_SPLITS)
    picked = [c for r in rows_with(only("babble"), 60, language="kk", talkers=talkers) for c in donors(r, clips)]
    assert {c.language for c in picked} == {"de", "ja", "kk", "fr"}


def test_ssn_is_fitted_to_one_reserved_talker(talkers):
    clips = by_path(ALL_SPLITS)
    for split in ("train", "test"):
        for r in rows_with(only("ssn"), 30, split, talkers=talkers):
            [c] = donors(r, clips)
            assert (c.split, c.language, c.reserved) == (split, "de", True)


def test_competing_rules(talkers, rirs):
    clips = by_path(ALL_SPLITS)
    rows = rows_with(only("competing"), 300, talkers=talkers, rirs=rirs)
    for r in rows:
        segments = r["segments"]
        assert 1 <= len(segments) <= 3
        covered = np.zeros(r["total_n"], bool)
        for s in segments:
            assert 0 <= s["start_n"] < s["end_n"] <= r["total_n"]
            assert not covered[s["start_n"] : s["end_n"]].any()
            covered[s["start_n"] : s["end_n"]] = True
            assert 1000 <= s["lowpass_hz"] <= 4000
            assert s["rir_id"]
            donor = clips[s["source_path"]]
            assert donor.reserved
            assert (donor.split, donor.language) == ("train", "de")
            assert 0 <= s["source_offset_n"] <= max(0, round(donor.duration_s * 16000) - (s["end_n"] - s["start_n"]))
        assert 0.10 <= covered.mean() <= 0.50
        assert 0 <= r["tir_db"] <= 15
        assert r["snr_db"] is None
        assert -90 <= r["dither_dbfs"] <= -70
        assert r["noise_source_ids"] == []


def test_competing_coverage_and_tir_spread_over_their_ranges(talkers, rirs):
    rows = rows_with(only("competing"), 400, talkers=talkers, rirs=rirs)
    coverage = [sum(s["end_n"] - s["start_n"] for s in r["segments"]) / r["total_n"] for r in rows]
    assert min(coverage) < 0.15
    assert max(coverage) > 0.45
    assert {len(r["segments"]) for r in rows} == {1, 2, 3}
    tir = [r["tir_db"] for r in rows]
    assert min(tir) < 1.5
    assert max(tir) > 13.5


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_every_competing_segment_has_a_rir_of_its_own_split_and_rooms_stay_disjoint(talkers, rirs, split):
    rooms = {
        s: {
            rirs.room_of(data_dir() / seg["rir_id"])
            for r in rows_with(only("competing"), 150, s, talkers=talkers, rirs=rirs)
            for seg in r["segments"]
        }
        for s in ("train", "validation", "test")
    }
    assert rooms[split]
    assert rooms[split] <= set(rirs.rooms(split))
    assert all(rooms[a].isdisjoint(rooms[b]) for a in rooms for b in rooms if a < b)


def test_reserved_talkers_do_not_move_any_other_draw(talkers, rirs, full_pool):
    for cfg in (only("babble"), only("competing"), FULL):
        a = rows_with(cfg, 20, pool=full_pool, rirs=rirs)
        b = rows_with(cfg, 20, pool=full_pool, rirs=rirs, talkers=talkers)
        for x, y in zip(a, b, strict=True):
            talker_columns = {"noise_source_ids", "noise_offsets", "segments", "tir_db"}
            moved = {k for k in x if x[k] != y[k]}
            assert moved <= talker_columns
            if x["condition"] not in ("babble", "ssn", "competing"):
                assert not moved


def test_a_missing_rir_tree_moves_nothing_in_the_talker_draws(talkers, rirs):
    with_rirs = rows_with(only("competing"), 30, talkers=talkers, rirs=rirs)
    without = rows_with(only("competing"), 30, talkers=talkers)
    for x, y in zip(with_rirs, without, strict=True):
        assert [s | {"rir_id": None} for s in x["segments"]] == y["segments"]
        assert {k: v for k, v in x.items() if k not in ("segments", "rir_id")} == {
            k: v for k, v in y.items() if k not in ("segments", "rir_id")
        }


def test_a_pool_without_talkers_names_the_condition_and_leaves_its_donors_empty():
    for condition in ("babble", "ssn", "competing"):
        for r in rows_with(only(condition), 5):
            assert r["noise_source_ids"] == []
            assert r["segments"] == []
            assert r["tir_db"] is None


def test_the_talker_conditions_have_no_snr_unless_they_are_noise():
    competing = rows_with(only("competing"), 5)
    assert all(r["snr_db"] is None for r in competing)
    assert all(r["snr_db"] in FULL.snr_grid_db for r in rows_with(only("babble"), 20))


def test_slice_rows_carry_transients_and_still_plan_and_repeat(pool):
    rows = [draw_sample(clip(f"c{i}"), timing(f"c{i}"), SLICE, pool) for i in range(300)]
    with_transients = [r for r in rows if r["transients"]]
    assert 0.2 < len(with_transients) / len(rows) < 0.4
    assert rows[0] == draw_sample(clip("c0"), timing("c0"), SLICE, pool)


@pytest.fixture
def all_conditions_data(tmp_path, monkeypatch):
    """A SWC tree of many articles by many readers, with DEMAND, WHAM!, MUSAN and RIR trees beside it."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    raw = tmp_path / "datasets/raw"
    raw.mkdir(parents=True)
    (raw / "swc").mkdir()
    (raw / "swc/manifest.json").write_text('{"dataset": "swc"}')
    for i in range(90):
        add_article(raw / "swc", "german", f"Art{i:02d}", f"Reader{i % 30:02d}")
    make_demand_tree(raw / "demand")
    make_wham_tree(raw / "wham")
    make_musan_tree(raw / "musan")
    make_rir_tree(raw / "rirs")
    return tmp_path


FULL_CONDITIONS = dataclasses.replace(
    FULL,
    name="openvad-slice",  # the work directory the helpers above read
    sources=("swc",),
    languages=("de",),
    babble_min_speakers=3,
)


def test_run_plan_builds_every_condition_from_reserved_clips_of_the_index(all_conditions_data):
    cfg = FULL_CONDITIONS
    run_index(cfg)
    run_plan(cfg)
    plan = plan_of(all_conditions_data)
    index = asyncio.run(read_parquet(all_conditions_data / "datasets/processed/work/openvad-slice/index/swc.parquet"))
    reserved = index.filter(pl.col("reserved"))
    assert reserved.height >= 3
    assert set(plan["condition"]) == {"clean", "competing", "babble", "ssn", "city", *DEMAND_CATEGORIES}
    # Reserved speakers are never foreground, and are the only talkers.
    assert not set(plan["speaker"]) & set(reserved["speaker"])
    reserved_paths = set(reserved["path"])
    for r in plan.filter(pl.col("condition").is_in(["babble", "ssn"])).iter_rows(named=True):
        assert set(r["noise_source_ids"]) <= reserved_paths
        assert len(r["noise_source_ids"]) == (6 if r["condition"] == "babble" else 1)
    for r in plan.filter(pl.col("condition") == "competing").iter_rows(named=True):
        assert {s["source_path"] for s in r["segments"]} <= reserved_paths
    for r in plan.filter(pl.col("condition").is_in(["city", *DEMAND_CATEGORIES])).iter_rows(named=True):
        assert (all_conditions_data / r["noise_source_ids"][0].partition("#")[0]).is_file()
    assert plan.filter(pl.col("transients").list.len() > 0).height > 0


def test_a_rerun_of_the_full_condition_plan_is_identical(all_conditions_data):
    run_index(FULL_CONDITIONS)
    run_plan(FULL_CONDITIONS)
    first = plan_of(all_conditions_data)
    run_plan(FULL_CONDITIONS)
    assert plan_of(all_conditions_data).equals(first)


def test_only_reserved_clips_of_the_augmented_sources_become_talkers(as_is_data):
    cfg = dataclasses.replace(
        AS_IS_CONFIG,
        name="openvad-slice",
        sources=("swc", "ami", "librivad"),
        condition_shares={"babble": 0.5, "competing": 0.5},
    )
    run_index(cfg)
    index_dir = as_is_data / "datasets/processed/work/openvad-slice/index"
    # Even a LibriVAD or AMI clip that is flagged reserved is not a talker: the first is not reserved by design, the
    # second is a span of a recording with other speakers in it.
    for name, flagged in (
        ("swc", pl.col("clip_id") == "english/Aaa"),
        ("ami", pl.lit(True)),
        ("librivad", pl.lit(True)),
    ):
        frame = asyncio.run(read_parquet(index_dir / f"{name}.parquet")).with_columns(reserved=flagged)
        asyncio.run(write_parquet({index_dir / f"{name}.parquet": frame.lazy()}))
    run_plan(cfg)
    rows = plan_of(as_is_data).filter(pl.col("source") == "swc")
    donors = {p for ids in rows["noise_source_ids"] for p in ids} | {
        s["source_path"] for ss in rows["segments"] for s in ss
    }
    assert donors
    assert all(p.startswith("datasets/raw/swc/") and p.endswith("english/Aaa/audio.ogg") for p in donors)


def competing_segments_of(total_n: int, seeds: range, talkers, cfg=FULL) -> list[list[dict]]:
    return [
        plan_module._draw_segments(clip("fg"), total_n, cfg, talkers, None, np.random.default_rng([s, 10]))
        for s in seeds
    ]


def test_no_competing_segment_is_under_the_floor_in_20000_full_draws(talkers):
    floor_n = FULL.competing_min_segment_ms * 16
    shortest = min(
        s["end_n"] - s["start_n"] for segs in competing_segments_of(160000, range(20000), talkers) for s in segs
    )
    assert shortest >= floor_n == 3200


def test_the_floor_holds_at_the_shortest_sample_that_fits_a_segment_and_coverage_stays_in_range(talkers):
    # 0.4 s: the 50% cap holds exactly one segment of the 200 ms floor. A shorter sample has no competing talker.
    total_n = 6400
    assert plan_module._holds_a_segment(total_n, FULL)
    assert not plan_module._holds_a_segment(total_n - 2, FULL)
    for total in (total_n, 6401, 9000, 16000, 48000):
        for segs in competing_segments_of(total, range(1500), talkers):
            lengths = [s["end_n"] - s["start_n"] for s in segs]
            assert min(lengths) >= 3200
            assert 1 <= len(segs) <= 3
            assert 0.10 * total - 1 <= sum(lengths) <= 0.50 * total
            ordered = sorted((s["start_n"], s["end_n"]) for s in segs)
            assert ordered[0][0] >= 0
            assert ordered[-1][1] <= total
            assert all(a[1] <= b[0] for a, b in pairwise(ordered))


def test_a_sample_too_short_for_a_segment_is_planned_clean_with_its_dither(talkers):
    short = dataclasses.replace(clip("short", duration_s=0.1), language="de")
    row = draw_sample(short, timing("short", speech=((0, 100),)), only("competing"), talkers=talkers)
    assert row["total_n"] < 6400
    assert (row["condition"], row["segments"], row["tir_db"], row["snr_db"]) == ("clean", [], None, None)
    assert -90 <= row["dither_dbfs"] <= -70


def source_clips(source: str, n: int = 80, language: str = "de") -> list[Clip]:
    return [talker(f"{source}-{i}", language, "train", source=source) for i in range(n)]


def test_adding_a_source_leaves_the_talker_rows_of_another_source_unchanged():
    x, y = source_clips("swc"), source_clips("common-voice")
    counts = {("swc", "de"): 1000, ("common-voice", "de"): 1000}
    alone, together = TalkerPool(x, counts, 50), TalkerPool(x + y, counts, 50)
    rirs_free = None
    for condition in ("babble", "ssn", "competing"):
        cfg = only(condition)
        for i in range(100):
            fg = dataclasses.replace(clip(f"s{i}", source="swc"), language="de")
            a = draw_sample(fg, timing(f"s{i}"), cfg, talkers=alone, rirs=rirs_free)
            b = draw_sample(fg, timing(f"s{i}"), cfg, talkers=together, rirs=rirs_free)
            assert a == b
            assert a["condition"] == condition


def test_a_row_of_the_added_source_draws_from_its_own_reserved_speakers():
    x, y = source_clips("swc"), source_clips("common-voice")
    pool = TalkerPool(x + y, {("swc", "de"): 1000, ("common-voice", "de"): 1000}, 50)
    for i in range(30):
        row = draw_sample(clip(f"cv{i}", source="common-voice"), timing(f"cv{i}"), only("babble"), talkers=pool)
        assert all(p.startswith("datasets/raw/common-voice/") for p in row["noise_source_ids"])


def test_fleurs_takes_its_talkers_from_the_declared_donor_source():
    donors = source_clips("common-voice")
    pool = TalkerPool(donors, {("common-voice", "de"): 1000}, 50)
    assert FULL.talker_donors == {"fleurs": "common-voice"}
    for condition in ("babble", "ssn"):
        for i in range(20):
            row = draw_sample(clip(f"f{i}", source="fleurs"), timing(f"f{i}"), only(condition), talkers=pool)
            assert row["noise_source_ids"]
            assert all(p.startswith("datasets/raw/common-voice/") for p in row["noise_source_ids"])
    competing = draw_sample(clip("f0", source="fleurs"), timing("f0"), only("competing"), talkers=pool)
    assert {s["source_path"].split("/")[2] for s in competing["segments"]} == {"common-voice"}


def test_a_config_with_fleurs_talkers_and_no_donor_source_is_refused_before_any_source_is_planned(data):
    cfg = dataclasses.replace(FULL, name="no-donor", sources=("fleurs", "mls"), languages=None)
    with pytest.raises(ValueError, match="talkers of fleurs come from common-voice"):
        run_plan(cfg)
    clean = dataclasses.replace(cfg, condition_shares={"clean": 1.0})
    assert plan_module._needs_talkers(clean) == []  # no talker condition, no donor needed


BABBLE_HALF = {"clean": 0.5, "babble": 0.5}


def write_index(cfg, source: str, clips: list[Clip], reserved: set[str]) -> None:
    out = work_dir(cfg) / "index"
    out.mkdir(parents=True, exist_ok=True)
    frame = index._frame(clips, {(c.split, c.language, str(c.speaker)) for c in clips if c.clip_id in reserved})
    asyncio.run(write_parquet({out / f"{source}.parquet": frame}))


def test_the_talker_pool_counts_the_speakers_of_a_language_in_the_index_reserved_or_not(data):
    cfg = dataclasses.replace(
        CONFIG, name="speakers-test", sources=("mls",), babble_min_speakers=50, condition_shares=BABBLE_HALF
    )
    clips = [talker(f"kk{i}", "kk", "train") for i in range(10)] + [talker(f"de{i}", "de", "train") for i in range(60)]
    clips = [dataclasses.replace(c, reserved=False) for c in clips]
    write_index(cfg, "mls", clips, {"kk0", "de0", "de1", "de2"})
    pool = plan_module._talker_pool(cfg)
    # kk: 10 speakers in the source, under 50: borrows from every language. de: 60, over 50: stays, then fills.
    kk = Counter(c.language for i in range(100) for c in pool.draw("mls", "kk", "train", 4, np.random.default_rng(i)))
    assert kk["de"] > 0
    de = [pool.draw("mls", "de", "train", 3, np.random.default_rng(i)) for i in range(100)]
    assert all({c.language for c in picked} == {"de"} and len(picked) == 3 for picked in de)


def test_a_split_with_rows_and_no_reserved_clip_to_draw_from_is_refused_before_planning(data):
    cfg = dataclasses.replace(CONFIG, name="split-test", sources=("mls",), condition_shares=BABBLE_HALF)
    clips = [talker(f"t{i}", "de", "train") for i in range(6)] + [talker(f"v{i}", "de", "validation") for i in range(6)]
    write_index(cfg, "mls", [dataclasses.replace(c, reserved=False) for c in clips], {"t0", "t1"})
    with pytest.raises(ValueError, match="rows in validation, where mls has no reserved clip"):
        plan_module._talker_pool(cfg)


def test_a_competing_share_without_rooms_is_refused_before_any_work(data):
    with pytest.raises(ValueError, match="needs room_p above 0"):
        run_plan(dataclasses.replace(FULL, name="no-rooms", room_p=0.0))
    no_competing = dataclasses.replace(CONFIG, room_p=0.0, condition_shares={"clean": 0.5, "office": 0.5})
    run_index(no_competing)
    run_plan(no_competing)  # the other conditions need no RIR
