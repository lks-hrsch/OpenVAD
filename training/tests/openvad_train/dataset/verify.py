# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from build_support import free_space, make_build  # noqa: F401 (free_space is a fixture)
from openvad_train.dataset import __main__ as cli
from openvad_train.dataset import parquet_io, publish, verify
from openvad_train.dataset.config import BuildConfig, dataset_root, work_dir
from openvad_train.dataset.construct import run_construct
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.publish import run_publish, set_small
from openvad_train.dataset.verify import VerifyError, run_verify
from openvad_train.fetch.common import read_manifest


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch, free_space) -> Path:  # noqa: F811
    """The data root of every test here, so no test can reach the share by forgetting a fixture."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def built(data) -> BuildConfig:
    cfg = make_build(data, as_is=True)
    run_construct(cfg, workers=1)
    run_publish(cfg)
    return cfg


def edit(cfg: BuildConfig, name: str, change) -> None:
    """Rewrite one published Parquet file through `change(frame) -> frame`."""
    path = dataset_root(cfg) / f"{name}.parquet"
    asyncio.run(write_parquet({path: change(asyncio.run(read_parquet(path))).lazy()}))


def test_a_good_build_passes_and_reports_shares_and_speech_ratios(built):
    report = run_verify(built)
    assert report.state == "complete"
    assert report.samples == {"train": 5, "validation": 2, "test": 2}
    assert (report.tars_hashed, report.constructed_decoded, report.as_is_decoded) == (4, 5, 4)
    groups = {(g["source"], g["split"]): g for g in report.groups}
    assert set(groups) == {
        ("swc", "train"),
        ("swc", "validation"),
        ("swc", "test"),
        ("ami", "train"),
        ("ami", "validation"),
        ("librivad", "train"),
        ("librivad", "test"),
    }
    train = groups["swc", "train"]
    assert train["samples"] == 3
    assert train["condition_shares"] == pytest.approx({"clean": 2 / 3, "office": 1 / 3})
    assert 0 < train["speech_ratio"] < 1
    assert 0 < train["label_speech_ratio"] < 1
    assert groups["ami", "train"]["speech_ratio"] == pytest.approx(2000 / 4000)
    # Neither AMI here nor LibriVAD has all-null speakers except LibriVAD.
    assert report.speakers_unchecked == ["librivad"]


def test_a_build_missing_shards_verifies_and_says_interim(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    (work_dir(cfg) / "fragments/test/00000.parquet").unlink()
    run_publish(cfg)
    assert run_verify(cfg).state == "interim"


def test_verify_before_publish_is_refused(data):
    cfg = make_build(data)
    with pytest.raises(VerifyError, match="run publish"):
        run_verify(cfg)


def test_broken_pointer_fails_verify(built, data):
    (data / "datasets/raw/ami/a1.wav").unlink()
    with pytest.raises(VerifyError, match=r"ami/a1.wav"):
        run_verify(built)


def test_a_missing_tar_fails_verify(built):
    (dataset_root(built) / "data/validation/00000.tar").unlink()
    with pytest.raises(VerifyError, match=r"validation/00000.tar"):
        run_verify(built)


def test_a_tar_without_its_sidecar_fails_verify(built):
    (dataset_root(built) / "data/test/00000.tar.sha256").unlink()
    with pytest.raises(VerifyError, match="sidecar"):
        run_verify(built)


def test_verify_catches_label_mismatch(built):
    edit(
        built,
        "train",
        lambda f: f.with_columns(
            labels=pl.when(pl.col("sample_id") == "swc/c0")
            .then(pl.col("labels").list.head(5))
            .otherwise(pl.col("labels"))
        ),
    )
    with pytest.raises(VerifyError, match=r"swc/c0: .* labels"):
        run_verify(built)


def test_a_label_mismatch_on_an_as_is_row_fails_verify(built):
    edit(
        built,
        "validation",
        lambda f: f.with_columns(
            labels=pl.when(pl.col("sample_id") == "ami/a1")
            .then(pl.col("labels").list.head(5))
            .otherwise(pl.col("labels"))
        ),
    )
    with pytest.raises(VerifyError, match=r"ami/a1: .* labels"):
        run_verify(built)


def test_a_label_outside_zero_to_two_fails_verify(built):
    edit(built, "test", lambda f: f.with_columns(labels=pl.col("labels").list.eval(pl.element() + 3)))
    with pytest.raises(VerifyError, match="not 0, 1 or 2"):
        run_verify(built)


def test_an_index_that_disagrees_fails_verify(built):
    edit(
        built,
        "index",
        lambda f: f.with_columns(
            source=pl.when(pl.col("sample_id") == "swc/c0").then(pl.lit("mls")).otherwise(pl.col("source"))
        ),
    )
    with pytest.raises(VerifyError, match=r"index.parquet and the split files disagree"):
        run_verify(built)


def test_an_index_with_a_sample_twice_fails_verify(built):
    edit(built, "index", lambda f: pl.concat([f, f.head(1)]))
    with pytest.raises(VerifyError, match="repeats a sample id"):
        run_verify(built)


def test_a_sample_in_two_splits_fails_verify(built):
    edit(
        built,
        "test",
        lambda f: pl.concat([f, asyncio.run(read_parquet(dataset_root(built) / "validation.parquet")).head(1)]),
    )
    with pytest.raises(VerifyError, match="more than one split"):
        run_verify(built)


def test_a_speaker_in_two_splits_fails_verify(built):
    edit(
        built,
        "test",
        lambda f: f.with_columns(
            speaker=pl.when(pl.col("source") == "swc").then(pl.lit("t1")).otherwise(pl.col("speaker"))
        ),
    )
    with pytest.raises(VerifyError, match="speaker 't1' is in 2 splits"):
        run_verify(built)


def test_a_raw_file_that_is_not_16_khz_fails_verify(built, data):
    sf.write(data / "datasets/raw/ami/a0.wav", np.zeros(8000, np.float32), 8000, subtype="PCM_16")
    with pytest.raises(VerifyError, match="ami/a0: raw file is 8000 Hz"):
        run_verify(built)


@pytest.mark.parametrize("source", ["ami", "librivad"])
def test_every_as_is_source_is_sampled_not_only_the_set_as_a_whole(built, data, monkeypatch, source):
    monkeypatch.setattr(verify, "_AS_IS_SAMPLE", 1)
    assert run_verify(built).as_is_decoded == 2
    for path in (data / "datasets/raw" / source).glob("*.wav"):
        sf.write(path, np.zeros(8000, np.float32), 8000, subtype="PCM_16")
    with pytest.raises(VerifyError, match=f"{source}/.*: raw file is 8000 Hz"):
        run_verify(built)


def test_changed_raw_data_makes_the_set_stale(built, data):
    (data / "datasets/raw/swc/manifest.json").write_text('{"dataset": "swc", "changed": true}')
    with pytest.raises(VerifyError, match="swc: the raw manifest changed"):
        run_verify(built)


def test_problems_are_all_listed_and_long_lists_are_cut(built, data, monkeypatch):
    monkeypatch.setattr(verify, "_MAX_REPORTED", 1)
    (data / "datasets/raw/ami/a1.wav").unlink()
    (data / "datasets/raw/ami/a0.wav").unlink()
    with pytest.raises(VerifyError, match=r"(?s)^2 problems.*and 1 more"):
        run_verify(built)


def test_a_flipped_byte_is_found_by_the_hash_and_a_rerun_of_construct_repairs_it(built):
    root = dataset_root(built)
    tar = root / "data/train/00001.tar"
    other = root / "data/train/00000.tar"
    raw = bytearray(tar.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    tar.write_bytes(bytes(raw))
    before = other.stat().st_mtime_ns
    fragment = work_dir(built) / "fragments/train/00001.parquet"
    # Same size, same names: construct takes the shard for finished, which is why verify must undo that.
    run_construct(built, workers=1)
    assert tar.read_bytes() == bytes(raw)

    with pytest.raises(VerifyError, match=r"train/00001.tar: sha256 differs"):
        run_verify(built)
    assert not fragment.exists()

    run_construct(built, workers=1)
    assert tar.read_bytes() != bytes(raw)
    assert other.stat().st_mtime_ns == before
    run_publish(built)
    assert run_verify(built).tars_hashed == 4


def test_a_set_small_killed_between_its_renames_is_found_and_repaired_by_a_rerun(built, monkeypatch):
    root = dataset_root(built)
    replaced = []
    real = parquet_io.os.replace

    def killed_after_the_first(src, dst):
        if replaced:
            raise KeyboardInterrupt
        replaced.append(dst)
        real(src, dst)

    monkeypatch.setattr(parquet_io.os, "replace", killed_after_the_first)
    with pytest.raises(KeyboardInterrupt):
        set_small(root, hours=1.0)
    monkeypatch.setattr(parquet_io.os, "replace", real)
    with pytest.raises(VerifyError, match=r"index.parquet and the split files disagree"):
        run_verify(built)
    set_small(root, hours=1.0)
    run_verify(built)


def test_a_publish_killed_before_its_manifest_leaves_no_complete_set(built):
    (work_dir(built) / "fragments/test/00000.parquet").unlink()

    def killed(root, info):
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(publish, "write_manifest", killed)
        with pytest.raises(KeyboardInterrupt):
            run_publish(built)
    with pytest.raises(VerifyError, match="run publish"):
        run_verify(built)


def test_a_manifest_that_counts_other_samples_than_the_files_fails_verify(built):
    edit(built, "test", lambda f: f.head(1))
    edit(built, "index", lambda f: f.filter(pl.col("sample_id") != "librivad/l1"))
    with pytest.raises(VerifyError, match=r"manifest.json says test has 2 samples, the file has 1"):
        run_verify(built)


def test_a_set_with_no_train_sample_fails_verify(built):
    edit(built, "train", lambda f: f.head(0))
    with pytest.raises(VerifyError, match=r"train.parquet has no samples"):
        run_verify(built)


def test_a_build_without_a_test_shard_passes_and_says_so_in_its_manifest(data):
    cfg = make_build(data)
    run_construct(cfg, workers=1)
    (work_dir(cfg) / "fragments/test/00000.parquet").unlink()
    run_publish(cfg)
    report = run_verify(cfg)
    assert (report.state, report.samples["test"]) == ("interim", 0)
    assert read_manifest(dataset_root(cfg))["missing_shards"] == ["test/00000"]  # pyright: ignore[reportOptionalSubscript]


def test_the_command_prints_the_report(built, monkeypatch, capsys):
    monkeypatch.setattr(verify, "SLICE", built)
    cli.main(["verify", "--config", "slice"])
    out = capsys.readouterr().out
    assert "verified: state complete" in out
    assert "swc train: 3 samples" in out
    assert "no speaker IDs): librivad" in out


def test_the_command_exits_with_the_problems(built, data, monkeypatch):
    monkeypatch.setattr(verify, "SLICE", built)
    (data / "datasets/raw/ami/a1.wav").unlink()
    with pytest.raises(SystemExit, match="1 problems"):
        cli.main(["verify", "--config", "slice"])
