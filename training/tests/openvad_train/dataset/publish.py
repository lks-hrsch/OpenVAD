# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

import polars as pl
import pytest
from datasets import load_dataset

from build_support import free_space, make_build  # noqa: F401 (free_space is a fixture)
from openvad_train.dataset import parquet_io
from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.config import BuildConfig, dataset_root, work_dir
from openvad_train.dataset.construct import PlanMismatch, run_construct
from openvad_train.dataset.labels import rasterise
from openvad_train.dataset.parquet_io import read_parquet
from openvad_train.dataset.plan import PLAN_SCHEMA
from openvad_train.dataset.publish import PLAN_ONLY, _git_state, run_publish, set_small
from openvad_train.design_resampler import coefficients
from openvad_train.fetch.common import file_hash, read_manifest


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch, free_space) -> Path:  # noqa: F811
    """The data root of every test here, so no test can reach the share by forgetting a fixture."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    return tmp_path


def build(tmp_path: Path, **kwargs) -> BuildConfig:
    cfg = make_build(tmp_path, **kwargs)
    run_construct(cfg, workers=1)
    run_publish(cfg)
    return cfg


@pytest.fixture
def built(data) -> BuildConfig:
    return build(data)


@pytest.fixture
def built_with_as_is(data) -> BuildConfig:
    return build(data, as_is=True)


def frame(cfg: BuildConfig, name: str) -> pl.DataFrame:
    return asyncio.run(read_parquet(dataset_root(cfg) / f"{name}.parquet"))


def test_load_dataset_gives_splits_with_pointers(built, data):
    ds = load_dataset(str(dataset_root(built)), cache_dir=str(data / "hf-cache"))
    assert set(ds) == {"train", "validation", "test"}
    row = ds["train"][0]
    assert not row["audio_filepath"].startswith("/")
    assert len(row["labels"]) == round(len(load_audio(row)) / 160)


def test_every_published_pointer_decodes_to_its_labels(built_with_as_is):
    for split in ("train", "validation", "test"):
        for row in frame(built_with_as_is, split).iter_rows(named=True):
            assert round(len(load_audio(row)) / 160) == len(row["labels"]), row["sample_id"]


def test_as_is_points_into_raw_and_copies_nothing(built_with_as_is):
    cfg = built_with_as_is
    train = frame(cfg, "train")
    as_is = train.filter(pl.col("treatment") == "as_is")
    assert as_is.height == 2
    assert as_is["audio_filepath"].str.starts_with("datasets/raw/").all()
    members: set[str] = set()
    for path in (dataset_root(cfg) / "data").rglob("*.tar"):
        with tarfile.open(path) as tar:
            members |= {m.name.removesuffix(".flac") for m in tar.getmembers()}
    assert members.isdisjoint(set(as_is["sample_id"]))
    assert not list((dataset_root(cfg) / "data").rglob("a0*"))


def test_as_is_labels_are_the_plan_intervals_at_g_from_the_start(built_with_as_is):
    row = frame(built_with_as_is, "train").filter(pl.col("sample_id") == "ami/a0").row(0, named=True)
    assert row["labels"] == rasterise([(500, 1500), (2000, 3000)], [], 4 * 16000).tolist()


def test_frame_labels_keep_a_gap_under_g_after_publish(built_with_as_is):
    row = frame(built_with_as_is, "train").filter(pl.col("sample_id") == "librivad/l0").row(0, named=True)
    assert row["gap_ms"] == 0
    assert row["labels"][100:110] == [0] * 10
    assert row["labels"] == rasterise([(300, 1000), (1100, 2700)], [], 3 * 16000).tolist()


def test_plan_only_columns_are_dropped_and_dither_stays(built_with_as_is):
    expected = [c for c in PLAN_SCHEMA if c not in PLAN_ONLY]
    for split in ("train", "validation", "test"):
        assert frame(built_with_as_is, split).columns == expected
    assert "dither_dbfs" in expected
    assert frame(built_with_as_is, "train").filter(pl.col("sample_id") == "swc/c0")["dither_dbfs"].item() == -80.0


def test_splits_hold_their_own_samples_in_rank_order(built_with_as_is):
    train = frame(built_with_as_is, "train")
    assert sorted(train["sample_id"]) == ["ami/a0", "librivad/l0", "swc/c0", "swc/c1", "swc/c2"]
    assert train["rank"].is_sorted()
    assert frame(built_with_as_is, "validation")["sample_id"].sort().to_list() == ["ami/a1", "swc/c3"]


def test_index_has_each_sample_once(built_with_as_is):
    index = frame(built_with_as_is, "index")
    assert index["sample_id"].n_unique() == index.height == 9
    assert index.columns == ["sample_id", "split", "in_small", "source", "treatment", "audio_filepath"]


def test_in_small_null_until_set(built):
    assert frame(built, "index")["in_small"].null_count() == 5
    assert frame(built, "train")["in_small"].null_count() == 3


def test_small_is_a_subset_and_leaves_out_librivad(built_with_as_is):
    set_small(dataset_root(built_with_as_is), hours=1.0)
    train = frame(built_with_as_is, "train")
    small = train.filter(pl.col("in_small"))
    assert 0 < small.height < train.height
    assert not small["source"].eq("librivad").any()
    assert small.height == 4


def test_small_cut_keeps_lowest_ranks(built):
    train = frame(built, "train").sort("rank")
    # Room for the two lowest ranks, not the third.
    hours = (float(train["duration"].head(2).sum()) + 0.01) / 3600
    set_small(dataset_root(built), hours=hours)
    train = frame(built, "train")
    assert max(train.filter(pl.col("in_small"))["rank"]) < min(train.filter(~pl.col("in_small"))["rank"])
    assert train["in_small"].sum() == 2


def test_small_in_the_index_agrees_with_train_and_leaves_other_splits_null(built):
    set_small(dataset_root(built), hours=1.0)
    index, train = frame(built, "index"), frame(built, "train")
    assert index.filter(pl.col("split") == "train").sort("sample_id")["in_small"].to_list() == (
        train.sort("sample_id")["in_small"].to_list()
    )
    assert index.filter(pl.col("split") != "train")["in_small"].null_count() == 2


def test_a_second_small_replaces_the_first(built):
    root = dataset_root(built)
    set_small(root, hours=1.0)
    train = frame(built, "train").sort("rank")
    set_small(root, hours=(float(train["duration"].head(1).sum()) + 0.01) / 3600)
    assert frame(built, "train")["in_small"].sum() == 1
    assert frame(built, "index")["in_small"].sum() == 1


def test_small_that_fails_while_writing_changes_neither_file(built, monkeypatch):
    root = dataset_root(built)
    before = {n: (root / f"{n}.parquet").read_bytes() for n in ("train", "index")}

    def broken(_queries):
        raise OSError("disk full")

    monkeypatch.setattr(parquet_io.pl, "collect_all_async", broken)
    with pytest.raises(OSError, match="disk full"):
        set_small(root, hours=1.0)
    assert {n: (root / f"{n}.parquet").read_bytes() for n in ("train", "index")} == before
    assert not list(root.glob("*.tmp"))


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf")])
def test_small_refuses_a_bad_size(built, hours):
    with pytest.raises(ValueError, match="hours"):
        set_small(dataset_root(built), hours=hours)


def test_a_complete_build_is_marked_complete(built):
    manifest = read_manifest(dataset_root(built))
    assert manifest is not None
    assert (manifest["state"], manifest["missing_shards"], manifest["orphan_tars"]) == ("complete", [], [])
    assert manifest["samples"] == {"train": 3, "validation": 1, "test": 1}


def test_partial_build_is_marked_interim_and_publishes_only_finished_shards(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    (work_dir(cfg) / "fragments/train/00001.parquet").unlink()
    run_publish(cfg)
    manifest = read_manifest(dataset_root(cfg))
    assert manifest is not None
    assert manifest["state"] == "interim"
    assert manifest["missing_shards"] == ["train/00001"]
    # The tar is still there: an orphan, reported and not published.
    assert manifest["orphan_tars"] == ["train/00001"]
    assert (dataset_root(cfg) / "data/train/00001.tar").exists()
    assert sorted(frame(cfg, "train")["sample_id"]) == ["swc/c0", "swc/c1"]
    assert "swc/c2" not in frame(cfg, "index")["sample_id"].to_list()


def test_working_files_are_not_fragments(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    fragments = work_dir(cfg) / "fragments"
    (fragments / "train/00002.parquet.tmp").write_bytes(b"half a file")
    (fragments / "train/00002.failures.jsonl").write_text("{}\n")
    run_publish(cfg)
    manifest = read_manifest(dataset_root(cfg))
    assert manifest is not None
    assert (manifest["state"], manifest["samples"]["train"]) == ("complete", 3)


def test_manifest_records_provenance(built_with_as_is, data):
    m = read_manifest(dataset_root(built_with_as_is))
    assert m is not None
    assert {"plan_version", "global_seed", "commit", "ffmpeg", "resampler_sha256", "delays", "licences"} <= set(m)
    # Computed from the design taps, not from the bindings the manifest reads them through.
    assert m["resampler_sha256"] == hashlib.sha256(coefficients().astype("<f4").tobytes()).hexdigest()
    assert m["ffmpeg"]["version"]
    assert "delays_ms" in m["delays"]
    assert m["licences"] == {"ami": ["CC0"], "librivad": ["CC0"], "swc": ["CC0"]}
    assert set(m["raw_sources"]) == {"ami", "demand", "librivad", "swc"}
    for name, digest in m["raw_sources"].items():
        assert digest == file_hash(data / "datasets/raw" / name / "manifest.json", "sha256")


def test_the_manifest_says_whether_the_working_tree_was_dirty(built):
    m = read_manifest(dataset_root(built))
    assert m is not None
    assert isinstance(m["commit_dirty"], bool)
    assert len(m["commit"]) == 40


def test_git_state_names_head_and_sees_every_change_except_ignored_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["config", "core.excludesFile", "/dev/null"]):
        subprocess.run([*git, *args], cwd=repo, check=True)
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "a.txt").write_text("a")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], cwd=repo, check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "ignored.txt").write_text("x")
    assert _git_state(repo) == (head, False)
    (repo / "a.txt").write_text("changed")
    assert _git_state(repo) == (head, True)
    subprocess.run([*git, "checkout", "-q", "a.txt"], cwd=repo, check=True)
    (repo / "new.txt").write_text("untracked")
    assert _git_state(repo) == (head, True)


def test_git_state_without_a_repository_is_none(tmp_path):
    assert _git_state(tmp_path) == (None, None)


def test_a_missing_raw_manifest_fails_before_any_file_is_written(data):
    cfg = make_build(data, as_is=True)
    run_construct(cfg, workers=1)
    (data / "datasets/raw/ami/manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="ami"):
        run_publish(cfg)
    assert not (dataset_root(cfg) / "train.parquet").exists()


def test_readme_names_the_three_files_and_no_data_directory(built):
    readme = (dataset_root(built) / "README.md").read_text()
    for split in ("train", "validation", "test"):
        assert f"path: {split}.parquet" in readme
    assert "config_name: default" in readme
    assert "data/" not in readme.split("---")[1]


def test_publish_with_no_rows_refuses_and_leaves_a_good_set_alone(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    run_publish(cfg)
    root = dataset_root(cfg)
    before = {p.name: p.read_bytes() for p in root.glob("*.*")}
    shutil.rmtree(work_dir(cfg) / "fragments")
    with pytest.raises(ValueError, match="nothing to publish"):
        run_publish(cfg)
    assert {p.name: p.read_bytes() for p in root.glob("*.*")} == before


def test_fragments_of_an_older_plan_are_refused_before_anything_is_written(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    run_publish(cfg)
    root = dataset_root(cfg)
    before = {p.name: p.read_bytes() for p in root.glob("*.*")}
    plan_path = work_dir(cfg) / "plan/plan.parquet"
    asyncio.run(parquet_io.write_parquet({plan_path: asyncio.run(read_parquet(plan_path)).head(4).lazy()}))
    with pytest.raises(PlanMismatch, match="plan_sha256"):
        run_publish(cfg)
    assert {p.name: p.read_bytes() for p in root.glob("*.*")} == before


def test_a_plan_of_another_version_is_refused(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    with pytest.raises(PlanMismatch, match="plan_version"):
        run_publish(dataclasses.replace(cfg, plan_version="v2"))
    assert not (dataset_root(cfg) / "train.parquet").exists()


def test_a_fragment_outside_the_plan_is_refused(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    fragments = work_dir(cfg) / "fragments"
    shutil.copy(fragments / "train/00001.parquet", fragments / "train/00007.parquet")
    with pytest.raises(PlanMismatch, match="train/00007"):
        run_publish(cfg)
    assert not (dataset_root(cfg) / "train.parquet").exists()


def test_shards_without_a_fragments_manifest_are_refused(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    (work_dir(cfg) / "fragments/manifest.json").unlink()
    with pytest.raises(PlanMismatch, match="no manifest"):
        run_publish(cfg)


def test_publish_refuses_without_a_plan(data):
    cfg = make_build(data)
    (work_dir(cfg) / "plan/manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="plan stage"):
        run_publish(cfg)


def test_a_rerun_gives_the_same_files(built):
    root = dataset_root(built)
    before = {n: (root / f"{n}.parquet").read_bytes() for n in ("train", "validation", "test", "index")}
    run_publish(built)
    assert {n: (root / f"{n}.parquet").read_bytes() for n in before} == before
