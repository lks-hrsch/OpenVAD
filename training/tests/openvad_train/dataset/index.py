# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
import hashlib
import json
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import polars as pl
import pytest

from openvad_train.dataset import index
from openvad_train.dataset.config import SLICE
from openvad_train.dataset.index import ForbiddenSource, apply_cap_rank, guard_path, reserve_speakers, run_index
from openvad_train.dataset.parquet_io import read_parquet
from openvad_train.dataset.schema import Clip, Timing, rank
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


def test_reserve_nothing_from_a_group_with_a_single_speaker():
    assert reserve_speakers([clip(speaker="only"), clip(id="d", speaker="only")]) == set()
    assert len(reserve_speakers([clip(speaker="a"), clip(speaker="b")])) == 1


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


# rank order of these ids: c2, c4, c1, c3, c0
def test_cap_stops_at_the_first_clip_that_would_exceed_it_even_if_a_smaller_one_follows():
    durations = {"c2": 2.0, "c4": 2.0, "c1": 0.5, "c3": 0.5, "c0": 0.5}
    clips = [clip(id=i, duration_s=d) for i, d in durations.items()]
    assert {c.clip_id for c in apply_cap_rank(clips, hours=3.0 / 3600)} == {"c2"}


def test_cap_counts_speech_seconds_when_given():
    clips = [clip(id=f"c{i}", duration_s=3600.0) for i in range(5)]
    speech = {"c2": 1.0, "c4": 2.0, "c1": 1.5, "c3": 0.5, "c0": 0.5}
    # c2 and c4 make 3.0 s; c1 would make 4.5 s, so it and everything after it in rank order go.
    assert {c.clip_id for c in apply_cap_rank(clips, hours=3.0 / 3600, speech_s=speech)} == {"c2", "c4"}


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
    for other in ("mfa", "other"):
        (tmp_path / "datasets/raw" / other).mkdir()
        (tmp_path / "datasets/raw" / other / "manifest.json").write_text("{}")
    return tmp_path


def _read(data_dir, source="swc") -> pl.DataFrame:
    return asyncio.run(read_parquet(data_dir / "datasets/processed/work/openvad-slice/index" / f"{source}.parquet"))


def test_run_index_writes_one_parquet_per_source(data_dir):
    run_index(dataclasses.replace(SLICE, languages=None))
    df = _read(data_dir)
    assert df.height == 5
    assert df.schema["clip_rank"] == pl.UInt64
    assert df.schema["speaker"] == pl.String
    row = df.filter(pl.col("clip_id") == "english/Aaa").row(0, named=True)
    assert row["clip_rank"] == rank("english/Aaa")
    assert row["path"] == "datasets/raw/swc/english/Aaa/audio.ogg"
    assert not (data_dir / "datasets/processed/work/openvad-slice/index/swc.parquet.tmp").exists()


def test_run_index_keeps_the_configured_languages(data_dir):
    run_index(SLICE)
    assert set(_read(data_dir)["language"]) == {"de"}


def test_run_index_reserves_nothing_for_speakerless_clips_or_single_speaker_groups(data_dir):
    run_index(dataclasses.replace(SLICE, languages=None))
    assert not _read(data_dir)["reserved"].any()


def test_run_index_rejects_a_forbidden_path(data_dir, monkeypatch):
    bad = SimpleNamespace(
        MFA_ALIGNED=True,
        clips=lambda _raw, _languages: iter([dataclasses.replace(clip(language="de"), path="raw/ava-speech/a.flac")]),
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
    swc_like = SimpleNamespace(
        MFA_ALIGNED=False,
        clips=lambda _raw, _languages: iter(many[:40]),
        timings=lambda _raw, clips: (Timing(c.clip_id, "word", [(0, 3_600_000)], [], None, "t") for c in clips),
    )
    mfa_like = SimpleNamespace(MFA_ALIGNED=True, clips=lambda _raw, _languages: iter(many[40:]))
    monkeypatch.setitem(index.SOURCES, "swc", swc_like)
    monkeypatch.setitem(index.SOURCES, "mfa", mfa_like)
    run_index(dataclasses.replace(SLICE, sources=("swc", "mfa"), languages=None))
    assert _read(data_dir, "swc").height == 30
    assert _read(data_dir, "mfa").height == 40


def test_run_index_names_a_source_without_adapter(data_dir):
    with pytest.raises(ValueError, match="no index adapter for source 'ami'"):
        run_index(dataclasses.replace(SLICE, sources=("ami",)))


def test_run_index_asks_the_adapter_for_the_configured_languages_only(data_dir, monkeypatch):
    asked = []
    monkeypatch.setitem(
        index.SOURCES,
        "swc",
        SimpleNamespace(MFA_ALIGNED=True, clips=lambda _raw, languages: asked.append(languages) or iter(())),
    )
    run_index(SLICE)
    run_index(dataclasses.replace(SLICE, languages=None))
    assert asked == [{"de"}, None]


def test_run_index_counts_speech_seconds_and_drops_clips_without_any(caplog):
    hour_ms = 3_600_000
    clips = [clip(id=f"c{i}", speaker="s", language="de", duration_s=1.0) for i in range(5)]
    # Rank order c2, c4, c1, c3, c0: 12 h, 24 h, none (dropped), then 34 h passes the 30 h cap.
    speech = {
        "c2": [(0, 12 * hour_ms)],
        "c4": [(0, 12 * hour_ms)],
        "c1": [],
        "c3": [(0, 10 * hour_ms)],
        "c0": [(0, hour_ms)],
    }
    fake = SimpleNamespace(
        MFA_ALIGNED=False,
        clips=lambda _raw, _languages: iter(clips),
        timings=lambda _raw, cs: (Timing(c.clip_id, "word", speech[c.clip_id], [], None, "t") for c in cs),
    )
    with caplog.at_level("INFO", logger=index.log.name):
        kept, speech_s, dropped = index._index_source("swc", cast(ModuleType, fake), None)
    assert {c.clip_id for c in kept} == {"c2", "c4"}
    assert dropped == {"no_aligned_word": 1, "over_cap": 2}
    assert speech_s is not None
    assert speech_s["c2"] == 12 * 3600
    assert "dropped {'no_aligned_word': 1, 'over_cap': 2}" in caplog.text


def test_run_index_reserves_per_source_and_for_all_languages_of_a_split(data_dir, monkeypatch):
    def source(name):
        clips = [
            clip(id=f"{lang}{i}", speaker=f"s{i}", language=lang, source=name)
            for lang in ("en", "de")
            for i in range(20)
        ]
        return SimpleNamespace(MFA_ALIGNED=True, clips=lambda _raw, _languages: iter(clips))

    monkeypatch.setitem(index.SOURCES, "swc", source("swc"))
    monkeypatch.setitem(index.SOURCES, "other", source("other"))
    run_index(dataclasses.replace(SLICE, sources=("swc", "other"), languages=None))
    for name in ("swc", "other"):
        df = _read(data_dir, name)
        picked = reserve_speakers(
            [clip(speaker=r["speaker"], language=r["language"]) for r in df.iter_rows(named=True)]
        )
        speakers = {speaker for _, _, speaker in picked}
        assert 1 <= len(speakers) <= 2
        # Both languages of a reserved speaker are reserved, and nobody else is.
        assert set(df.filter(pl.col("reserved"))["speaker"]) == speakers
        assert df.filter(pl.col("speaker").is_in(speakers))["reserved"].all()


def test_run_index_writes_a_stage_manifest(data_dir):
    run_index(dataclasses.replace(SLICE, languages=None))
    m = json.loads((data_dir / "datasets/processed/work/openvad-slice/index/manifest.json").read_text())
    swc_m = m["per_source"]["swc"]
    assert (m["stage"], m["config"], m["sources"], m["languages"]) == ("index", "openvad-slice", ["swc"], None)
    assert swc_m["clips"] == 5
    assert sum(v["clips"] for v in swc_m["by_split"].values()) == 5
    assert set(swc_m["by_language"]) == {"en", "de", "nl"}
    assert swc_m["dropped"] == {"no_aligned_word": 0, "over_cap": 0}
    # Five articles of 2.0 s with 600 ms of aligned words each.
    assert swc_m["hours"] == pytest.approx(5 * 2.0 / 3600, abs=1e-3)
    assert swc_m["speech_hours"] == pytest.approx(5 * 0.6 / 3600, abs=1e-3)
    raw_manifest = data_dir / "datasets/raw/swc/manifest.json"
    assert swc_m["raw_manifest_sha256"] == hashlib.sha256(raw_manifest.read_bytes()).hexdigest()


def test_the_stage_runs_a_named_configuration_and_requires_one(data_dir):
    index.main(["--config", "slice"])
    assert set(_read(data_dir)["language"]) == {"de"}
    with pytest.raises(SystemExit):
        index.main([])


def test_every_adapter_is_registered_under_its_raw_folder_and_says_whether_mfa_labels_it():
    assert set(index.SOURCES) == {"swc", "common-voice", "mls", "fleurs", "voxpopuli", "speech-commands"}
    for name, module in index.SOURCES.items():
        assert name == module.SOURCE
        assert isinstance(module.MFA_ALIGNED, bool)
        assert callable(module.clips)
        assert hasattr(module, "timings") == (not module.MFA_ALIGNED or name == "speech-commands")
