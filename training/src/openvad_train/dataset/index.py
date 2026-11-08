# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `index`: what clips exist, which split each is in, and which speakers are capped or reserved."""

import argparse
import asyncio
import dataclasses
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from math import floor
from pathlib import PurePosixPath
from types import ModuleType

import polars as pl

from openvad_train.dataset.config import CONFIGS, SPEAKER_CAP_HOURS, BuildConfig, work_dir
from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.parquet_io import write_parquet
from openvad_train.dataset.schema import Clip, Timing, rank
from openvad_train.dataset.sources import (
    ami,
    callhome,
    commonvoice,
    fleurs,
    librivad,
    mls,
    speechcommands,
    swc,
    voices,
    voxpopuli,
)
from openvad_train.fetch.common import MANIFEST, file_hash, raw_dir, write_manifest


log = logging.getLogger(__name__)


class ForbiddenSource(Exception):  # noqa: N818  (the name the plan and its tests use)
    """A clip that ADR 0024 keeps out of the pipeline, or out of the split it was assigned."""


# Directories under datasets/raw whose audio never enters the pipeline: benchmarks and LibriSpeech itself.
# LibriVAD small is for screening apart from OpenVAD Small (ADR 0017, ADR 0023), so it is not read here either.
_FORBIDDEN_RAW = {"ava-speech", "fearless-steps", "picovoice", "simsamu", "librispeech", "librivad-small"}
_LIBRIVAD = {"librivad", "librivad-val"}
# LibriSpeech evaluation folders reach the set through LibriVAD only, and only in their own split; its training
# folder only in train.
_SPLIT_FOLDERS = {"dev-clean": "validation", "test-clean": "test", "train-clean-100": "train"}


def guard_path(path: str, split: str) -> None:
    """Refuse a path ADR 0024 keeps out of `split`. It fails closed: a path that is not under `raw/<source>`, or
    that has a `..` or more than one `raw` segment, cannot be judged and is refused, and names are compared without
    regard to case.
    """
    parts = tuple(p.casefold() for p in PurePosixPath(path).parts)
    if ".." in parts:
        raise ForbiddenSource(f"{path}: a path with a .. segment cannot be checked (ADR 0024)")
    if parts.count("raw") != 1 or parts.index("raw") + 1 >= len(parts):
        raise ForbiddenSource(f"{path}: not a path under raw/<source>, so it cannot be checked (ADR 0024)")
    source = parts[parts.index("raw") + 1]
    if source in _FORBIDDEN_RAW:
        raise ForbiddenSource(f"{path}: {source} never enters the pipeline (ADR 0024)")
    if source == "librivad-val" and split != "validation":
        raise ForbiddenSource(f"{path}: LibriVAD val is validation only, not {split} (ADR 0024)")
    for part in parts:
        if part in _SPLIT_FOLDERS and source not in _LIBRIVAD:
            raise ForbiddenSource(f"{path}: {part} is read through LibriVAD only (ADR 0024)")
        if part in _SPLIT_FOLDERS and split != _SPLIT_FOLDERS[part]:
            raise ForbiddenSource(f"{path}: {part} belongs to {_SPLIT_FOLDERS[part]}, not {split} (ADR 0024)")


def reserve_speakers(clips: Iterable[Clip], share: float = 0.05) -> set[tuple[str, str, str]]:
    """`share` of the speakers of each (split, language), at least one, lowest salted rank first.

    A clip without a speaker (FLEURS) never reserves anything, and a group with a single speaker reserves none,
    because reserving it would leave no foreground speaker. The caller passes one source's clips.
    """
    speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in clips:
        if c.speaker is not None:
            speakers[c.split, c.language].add(c.speaker)
    return {
        (split, language, speaker)
        for (split, language), names in speakers.items()
        if len(names) > 1
        # The salt keeps the reserved choice independent of a split drawn from rank(speaker) itself.
        for speaker in sorted(names, key=lambda s: (rank(f"reserved:{s}"), s))[: max(1, floor(len(names) * share))]
    }


def apply_cap_rank(
    clips: Iterable[Clip], hours: float = SPEAKER_CAP_HOURS, speech_s: Mapping[str, float] | None = None
) -> Iterator[Clip]:
    """At most `hours` per speaker and source: a speaker's clips in rank order, stopping before the first that would
    exceed the cap, even if a smaller one follows. A clip counts `speech_s[clip_id]` seconds when that is given
    (SWC counts aligned speech), else its `duration_s`. Clips without a speaker are not capped; the input order is kept.
    """
    clips = list(clips)
    by_speaker: dict[tuple[str, str], list[Clip]] = defaultdict(list)
    for c in clips:
        if c.speaker is not None:
            by_speaker[c.source, c.speaker].append(c)
    kept: set[str] = set()
    for group in by_speaker.values():
        total = 0.0
        for c in sorted(group, key=lambda c: (rank(c.clip_id), c.clip_id)):
            total += c.duration_s if speech_s is None else speech_s[c.clip_id]
            if total > hours * 3600:
                break
            kept.add(c.clip_id)
    return (c for c in clips if c.speaker is None or c.clip_id in kept)


# An adapter has `clips(raw, languages)` and the constant `MFA_ALIGNED`. One that is not forced-aligned by MFA also has
# `timings(raw, clips)` and is capped here. An MFA adapter may have `timings` too, for its clips without a transcript
# (Speech Commands background noise), whose labels are fixed. An adapter with `REPORTS = True` takes a third argument,
# a dict it fills with what it left out and why (duplicates, held-out speakers, skipped languages), which the index
# manifest records under `adapter`. An adapter with `RESERVES_SPEAKERS = False` (LibriVAD and VOiCES, whose spec rows
# keep every file) has no reserved speakers: `run_index` skips `reserve_speakers` for it, and its speaker IDs stay.
SOURCES: dict[str, ModuleType] = {
    m.SOURCE: m for m in (swc, commonvoice, mls, fleurs, voxpopuli, speechcommands, ami, callhome, voices, librivad)
}

_SCHEMA = {
    "clip_id": pl.String,
    "source": pl.String,
    "language": pl.String,
    "speaker": pl.String,
    "split": pl.String,
    "path": pl.String,
    "offset_s": pl.Float64,
    "duration_s": pl.Float64,
    "licence": pl.String,
    "transcript": pl.String,
    "reserved": pl.Boolean,
    "row_in_file": pl.Int64,
    "clip_rank": pl.UInt64,
}


def _frame(clips: list[Clip], reserved: set[tuple[str, str, str]]) -> pl.LazyFrame:
    # A speaker reserved in one language of a split is reserved for all its clips there, so it is never foreground.
    reserved_in_split = {(split, speaker) for split, _, speaker in reserved}
    rows = [
        dataclasses.asdict(c) | {"reserved": (c.split, c.speaker) in reserved_in_split, "clip_rank": rank(c.clip_id)}
        for c in clips
    ]
    return pl.DataFrame(rows, schema=_SCHEMA).lazy()


def _speech_s(timings: Iterable[Timing]) -> dict[str, float]:
    return {t.clip_id: sum(e - s for s, e in apply_gap_rule(t.speech_ms, 0)) / 1000 for t in timings}


def _index_source(
    name: str, module: ModuleType, languages: tuple[str, ...] | None
) -> tuple[list[Clip], dict[str, float] | None, dict[str, int], dict]:
    """The clips of one source that survive its rules, their speech seconds (non-MFA sources), how many
    were dropped for which reason, and the adapter's own report of what it left out.
    """
    raw = raw_dir(name)
    report: dict = {}
    wanted = None if languages is None else set(languages)
    clips = list(module.clips(raw, wanted, report) if getattr(module, "REPORTS", False) else module.clips(raw, wanted))
    for c in clips:
        guard_path(c.path, c.split)
    speech_s = None
    dropped: dict[str, int] = {}
    if not module.MFA_ALIGNED:
        # MFA sources get their cap in `align`, after the drop rules. Here a clip without speech yields no labels
        # and would only use up budget, and the cap counts speech seconds.
        speech_s = _speech_s(module.timings(raw, clips))
        with_speech = [c for c in clips if speech_s[c.clip_id] > 0]
        capped = list(apply_cap_rank(with_speech, speech_s=speech_s))
        dropped = {"no_aligned_word": len(clips) - len(with_speech), "over_cap": len(with_speech) - len(capped)}
        clips = capped
    log.info("%s: %d clips, dropped %s, adapter report %s", name, len(clips), dropped, report)
    return clips, speech_s, dropped, report


def _totals(clips: list[Clip], speech_s: dict[str, float] | None) -> dict:
    totals = {
        "clips": len(clips),
        "speakers": len({c.speaker for c in clips if c.speaker is not None}),
        "hours": round(sum(c.duration_s for c in clips) / 3600, 3),
    }
    if speech_s is not None:
        totals["speech_hours"] = round(sum(speech_s[c.clip_id] for c in clips) / 3600, 3)
    return totals


def _raw_manifest_hash(name: str) -> str | None:
    """The fetch writes the raw manifest last, so a source that is still being fetched has none."""
    path = raw_dir(name) / MANIFEST
    if not path.is_file():
        log.warning("%s: no %s yet, so the raw manifest hash is null", name, MANIFEST)
        return None
    return file_hash(path, "sha256")


def _source_manifest(
    name: str, clips: list[Clip], speech_s: dict[str, float] | None, dropped: dict[str, int], report: dict
) -> dict:
    by_split = {split: [c for c in clips if c.split == split] for split in sorted({c.split for c in clips})}
    by_language = {lang: [c for c in clips if c.language == lang] for lang in sorted({c.language for c in clips})}
    return {
        **_totals(clips, speech_s),
        "by_split": {split: _totals(cs, speech_s) for split, cs in by_split.items()},
        "by_language": {lang: _totals(cs, speech_s) for lang, cs in by_language.items()},
        "dropped": dropped,
        "adapter": report,
        "raw_manifest_sha256": _raw_manifest_hash(name),
    }


def run_index(cfg: BuildConfig) -> None:
    """Write `work/<name>/index/<source>.parquet` for every source of `cfg`, and the stage's `manifest.json`."""
    indexed = {}
    for name in cfg.sources:
        if name not in SOURCES:
            raise ValueError(f"no index adapter for source {name!r}; have {sorted(SOURCES)}")
        indexed[name] = _index_source(name, SOURCES[name], cfg.languages)
    per_source = {name: _source_manifest(name, *parts) for name, parts in indexed.items()}  # before any file exists
    out = work_dir(cfg) / "index"
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        out / f"{name}.parquet": _frame(
            parts[0],
            reserve_speakers(parts[0], cfg.reserved_share)
            if getattr(SOURCES[name], "RESERVES_SPEAKERS", True)
            else set(),
        )
        for name, parts in indexed.items()
    }
    asyncio.run(write_parquet(frames))
    write_manifest(
        out,
        {
            "stage": "index",
            "config": cfg.name,
            "sources": list(cfg.sources),
            "languages": None if cfg.languages is None else list(cfg.languages),
            "per_source": per_source,
        },
    )


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset index", description=__doc__)
    parser.add_argument("--config", choices=tuple(CONFIGS), required=True)
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_index(CONFIGS[ns.config])
