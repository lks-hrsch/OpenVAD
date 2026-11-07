# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
import json
from collections import Counter
from types import SimpleNamespace
from typing import Literal

import polars as pl
import pytest

from demand_support import make_demand_tree
from openvad_train.dataset import __main__ as cli
from openvad_train.dataset import plan as plan_module
from openvad_train.dataset.config import DEMAND_CATEGORIES, FULL, SLICE
from openvad_train.dataset.cutting import Chunk
from openvad_train.dataset.index import run_index
from openvad_train.dataset.labels import rasterise
from openvad_train.dataset.noise import NoisePool, demand_channels
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.plan import assign_shards, clip_rows, draw_sample, run_plan
from openvad_train.dataset.schema import Clip, Timing, rank, sample_seed
from swc_support import make_swc_tree


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


def test_slice_draws_only_its_codecs_and_conditions():
    rows = rows_of(SLICE, 500)
    assert {r["codec"] for r in rows} == {"none", "g722"}
    assert {r["condition"] for r in rows} == {"clean", *DEMAND_CATEGORIES}


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
    backwards = SimpleNamespace(timings=lambda raw, clips: reversed(list(real.timings(raw, clips))))
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


def test_full_refuses_conditions_without_a_draw_before_any_work(data):
    with pytest.raises(NotImplementedError, match="babble"):
        run_plan(FULL)


def test_an_mfa_source_without_word_tables_names_them(data, monkeypatch):
    run_index(CONFIG)
    monkeypatch.setitem(plan_module.SOURCES, "swc", SimpleNamespace())
    with pytest.raises(FileNotFoundError, match="words/swc"):
        run_plan(CONFIG)


def test_a_source_without_adapter_is_named(data):
    with pytest.raises(ValueError, match="no index adapter for source 'ami'"):
        run_plan(dataclasses.replace(SLICE, sources=("ami",)))


def test_a_repeated_sample_id_is_refused(data, monkeypatch):
    run_index(CONFIG)
    monkeypatch.setattr(
        plan_module, "clip_rows", lambda c, t, cfg, pool, counts: [draw_sample(c, t, cfg, pool, Chunk(0, 1))] * 2
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
