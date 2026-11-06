# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Literal

import polars as pl
import pytest

from openvad_train.dataset import index
from openvad_train.dataset.config import SLICE
from openvad_train.dataset.index import ForbiddenSource, apply_cap_rank, guard_path, reserve_speakers, run_index
from openvad_train.dataset.parquet_io import read_parquet
from openvad_train.dataset.schema import Clip, rank
from swc_support import make_swc_tree


Split = Literal["train", "validation", "test"]


def clip(id="c", speaker: str | None = "s", language="en", split: Split = "train", source="x", duration_s=1.0) -> Clip:
    return Clip(id, source, language, speaker, split, f"datasets/raw/{source}/{id}.flac", 0.0, duration_s, "CC0", None)


@pytest.mark.parametrize(
    "path",
    [
        "raw/ava-speech/a.flac",
        "raw/simsamu/a.m4a",
        "raw/fearless-steps/a.wav",
        "raw/librispeech/LibriSpeech/test-clean/a.flac",
        "raw/librispeech/LibriSpeech/train-clean-100/a.flac",
        "raw/picovoice/a.wav",
    ],
)
def test_guard_rejects_benchmark_paths(path):
    with pytest.raises(ForbiddenSource):
        guard_path(path, "test")


def test_guard_keeps_librivad_evaluation_out_of_train():
    guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "test")
    guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "validation")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "train")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "train")


def test_guard_keeps_each_librivad_evaluation_in_its_own_split():
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "validation")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "test")


def test_guard_rejects_evaluation_folders_outside_librivad():
    with pytest.raises(ForbiddenSource):
        guard_path("raw/voices/LibriSpeech/test-clean/a.wav", "test")


def test_guard_accepts_ordinary_training_paths():
    guard_path("datasets/raw/swc/german/A/audio.ogg", "train")
    guard_path("datasets/raw/librivad/generated/LibriSpeech/train-clean-100/a.wav", "train")


def test_reserve_at_least_one_per_language():
    clips = [clip(speaker=f"s{i}", language="kk") for i in range(6)]
    assert len(reserve_speakers(clips)) == 1


def test_reserve_five_percent_per_split_and_language():
    clips = [clip(speaker=f"s{i}", language="en") for i in range(100)]
    clips += [clip(speaker=f"t{i}", language="de") for i in range(40)]
    clips += [clip(speaker=f"v{i}", split="test") for i in range(3)]
    reserved = reserve_speakers(clips)
    assert len([r for r in reserved if r[:2] == ("train", "en")]) == 5
    assert len([r for r in reserved if r[:2] == ("train", "de")]) == 2
    assert len([r for r in reserved if r[:2] == ("test", "en")]) == 1


def test_reserve_ignores_clips_without_a_speaker():
    assert reserve_speakers([clip(speaker=None) for _ in range(5)]) == set()


def test_reserve_is_stable_across_clip_order():
    clips = [clip(id=f"c{i}", speaker=f"s{i}") for i in range(60)]
    assert reserve_speakers(clips) == reserve_speakers(reversed(clips))


def test_cap_keeps_rank_prefix_under_30_h():
    clips = [clip(id=f"c{i}", speaker="s", duration_s=3600.0) for i in range(40)]
    kept = list(apply_cap_rank(clips))
    assert sum(c.duration_s for c in kept) <= 30 * 3600
    assert {c.clip_id for c in kept} == {c.clip_id for c in sorted(clips, key=lambda c: rank(c.clip_id))[:30]}


def test_cap_stops_at_the_first_clip_that_would_exceed_it():
    clips = [clip(id=f"c{i}", duration_s=d) for i, d in enumerate([2.0, 2.0, 0.5, 0.5, 0.5])]
    by_rank = sorted(clips, key=lambda c: rank(c.clip_id))
    kept = list(apply_cap_rank(clips, hours=3.0 / 3600))
    expected = []
    total = 0.0
    for c in by_rank:
        total += c.duration_s
        if total > 3.0:
            break
        expected.append(c.clip_id)
    assert {c.clip_id for c in kept} == set(expected)


def test_cap_counts_each_speaker_and_source_on_its_own():
    clips = [clip(id=f"a{i}", speaker="s", source="x", duration_s=3600.0) for i in range(3)]
    clips += [clip(id=f"b{i}", speaker="s", source="y", duration_s=3600.0) for i in range(3)]
    clips += [clip(id=f"c{i}", speaker="t", source="x", duration_s=3600.0) for i in range(3)]
    assert len(list(apply_cap_rank(clips, hours=2.0))) == 6


def test_cap_leaves_clips_without_a_speaker_alone():
    clips = [clip(id=f"c{i}", speaker=None, duration_s=3600.0) for i in range(40)]
    assert list(apply_cap_rank(clips)) == clips


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    make_swc_tree(tmp_path / "datasets/raw/swc")
    return tmp_path


def _read(data_dir, source="swc") -> pl.DataFrame:
    return asyncio.run(read_parquet(data_dir / "datasets/processed/work/index" / f"{source}.parquet"))


def test_run_index_writes_one_parquet_per_source(data_dir):
    run_index(dataclasses.replace(SLICE, languages=None))
    df = _read(data_dir)
    assert df.height == 5
    assert df.schema["rank"] == pl.UInt64
    assert df.schema["speaker"] == pl.String
    row = df.filter(pl.col("clip_id") == "english/Aaa").row(0, named=True)
    assert row["rank"] == rank("english/Aaa")
    assert row["path"] == "datasets/raw/swc/english/Aaa/audio.ogg"
    assert not (data_dir / "datasets/processed/work/index/swc.parquet.tmp").exists()


def test_run_index_keeps_the_configured_languages(data_dir):
    run_index(SLICE)
    assert set(_read(data_dir)["language"]) == {"de"}


def test_run_index_reserves_a_speaker_per_split_and_language(data_dir):
    run_index(dataclasses.replace(SLICE, languages=None))
    df = _read(data_dir)
    assert df.filter(pl.col("speaker").is_null())["reserved"].to_list() == [False]
    named = df.filter(pl.col("speaker").is_not_null())
    # Every (split, language) holds its own lowest salted rank, and equal readers agree.
    assert named.group_by("split", "language", "speaker").agg(pl.col("reserved").n_unique())["reserved"].max() == 1
    assert named.group_by("split", "language").agg(pl.col("reserved").any())["reserved"].all()


def test_run_index_rejects_a_forbidden_path(data_dir, monkeypatch):
    bad = SimpleNamespace(
        clips=lambda _raw: iter([dataclasses.replace(clip(language="de"), path="raw/ava-speech/a.flac")])
    )
    monkeypatch.setitem(index.SOURCES, "swc", bad)
    with pytest.raises(ForbiddenSource):
        run_index(SLICE)


def test_run_index_caps_only_sources_without_mfa(data_dir, monkeypatch):
    many = [
        clip(id=f"c{i}", speaker="s", language="de", duration_s=3600.0, source=n)
        for n in ("swc", "mfa")
        for i in range(40)
    ]
    swc_like = SimpleNamespace(clips=lambda _raw: iter(many[:40]), timings=lambda *_: iter(()))
    mfa_like = SimpleNamespace(clips=lambda _raw: iter(many[40:]))
    monkeypatch.setitem(index.SOURCES, "swc", swc_like)
    monkeypatch.setitem(index.SOURCES, "mfa", mfa_like)
    run_index(dataclasses.replace(SLICE, sources=("swc", "mfa"), languages=None))
    assert _read(data_dir, "swc").height == 30
    assert _read(data_dir, "mfa").height == 40


def test_run_index_names_a_source_without_adapter(data_dir):
    with pytest.raises(ValueError, match="no index adapter for source 'ami'"):
        run_index(dataclasses.replace(SLICE, sources=("ami",)))
