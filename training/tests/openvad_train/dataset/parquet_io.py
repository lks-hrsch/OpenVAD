# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio

import polars as pl
import pytest

from openvad_train.dataset.parquet_io import read_parquet, write_parquet


def test_parquet_round_trip_keeps_labels(tmp_path):
    p = tmp_path / "a.parquet"
    df = pl.DataFrame(
        {"sample_id": ["a"], "labels": [[0, 1, 2]]}, schema={"sample_id": pl.String, "labels": pl.List(pl.UInt8)}
    )
    asyncio.run(write_parquet({p: df.lazy()}))
    assert asyncio.run(read_parquet(p)).equals(df)


def test_read_selects_columns(tmp_path):
    p = tmp_path / "a.parquet"
    asyncio.run(write_parquet({p: pl.LazyFrame({"a": [1], "b": [2]})}))
    assert asyncio.run(read_parquet(p, ["b"])).columns == ["b"]


def test_several_files_in_one_call(tmp_path):
    frames = {tmp_path / "a.parquet": pl.LazyFrame({"a": [1]}), tmp_path / "b.parquet": pl.LazyFrame({"a": [2]})}
    asyncio.run(write_parquet(frames))
    assert [asyncio.run(read_parquet(p))["a"][0] for p in frames] == [1, 2]
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.parquet", "b.parquet"]


def test_failed_write_leaves_no_file(tmp_path):
    p = tmp_path / "a.parquet"
    bad = pl.LazyFrame({"a": [1]}).select(pl.col("missing"))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        asyncio.run(write_parquet({p: bad}))
    assert not p.exists()


def test_one_failed_frame_leaves_no_file_for_the_others(tmp_path):
    good, bad = tmp_path / "good.parquet", tmp_path / "bad.parquet"
    frames = {good: pl.LazyFrame({"a": [1]}), bad: pl.LazyFrame({"a": [1]}).select(pl.col("missing"))}
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        asyncio.run(write_parquet(frames))
    assert list(tmp_path.iterdir()) == []


def test_write_replaces_an_existing_file_only_on_success(tmp_path):
    p = tmp_path / "a.parquet"
    asyncio.run(write_parquet({p: pl.LazyFrame({"a": [1]})}))
    bad = pl.LazyFrame({"a": [2]}).select(pl.col("missing"))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        asyncio.run(write_parquet({p: bad}))
    assert asyncio.run(read_parquet(p))["a"].to_list() == [1]
    asyncio.run(write_parquet({p: pl.LazyFrame({"a": [2]})}))
    assert asyncio.run(read_parquet(p))["a"].to_list() == [2]
