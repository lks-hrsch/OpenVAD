# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `index`: what clips exist, which split each is in, and which speakers are capped or reserved."""

import argparse
import asyncio
import dataclasses
from collections import defaultdict
from collections.abc import Iterable, Iterator
from math import floor
from pathlib import PurePosixPath
from types import ModuleType

import polars as pl

from openvad_train.dataset.config import FULL, SLICE, BuildConfig, work_dir
from openvad_train.dataset.parquet_io import write_parquet
from openvad_train.dataset.schema import Clip, rank
from openvad_train.dataset.sources import swc
from openvad_train.fetch.common import raw_dir


class ForbiddenSource(Exception):  # noqa: N818  (the name the plan and its tests use)
    """A clip that ADR 0024 keeps out of the pipeline, or out of the split it was assigned."""


# Directories under datasets/raw whose audio never enters the pipeline: benchmarks and LibriSpeech itself.
_FORBIDDEN_RAW = {"ava-speech", "fearless-steps", "picovoice", "simsamu", "librispeech"}
# LibriSpeech evaluation folders reach the set through LibriVAD only, and only in their own split.
_EVALUATION_FOLDERS = {"dev-clean": "validation", "test-clean": "test"}


def guard_path(path: str, split: str) -> None:
    parts = PurePosixPath(path).parts
    source = parts[parts.index("raw") + 1] if "raw" in parts and parts.index("raw") + 1 < len(parts) else None
    if source in _FORBIDDEN_RAW:
        raise ForbiddenSource(f"{path}: {source} never enters the pipeline (ADR 0024)")
    if source == "librivad-val" and split != "validation":
        raise ForbiddenSource(f"{path}: LibriVAD val is validation only, not {split} (ADR 0024)")
    for part in parts:
        if part in _EVALUATION_FOLDERS and not (source or "").startswith("librivad"):
            raise ForbiddenSource(f"{path}: {part} is read through LibriVAD only (ADR 0024)")
        if part in _EVALUATION_FOLDERS and split != _EVALUATION_FOLDERS[part]:
            raise ForbiddenSource(f"{path}: {part} belongs to {_EVALUATION_FOLDERS[part]}, not {split} (ADR 0024)")


def reserve_speakers(clips: Iterable[Clip], share: float = 0.05) -> set[tuple[str, str, str]]:
    """`share` of the speakers of each (split, language), at least one, lowest salted rank first.

    A clip without a speaker (FLEURS) never reserves anything.
    """
    speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in clips:
        if c.speaker is not None:
            speakers[c.split, c.language].add(c.speaker)
    return {
        (split, language, speaker)
        for (split, language), names in speakers.items()
        # The salt keeps the reserved choice independent of a split drawn from rank(speaker) itself.
        for speaker in sorted(names, key=lambda s: (rank(f"reserved:{s}"), s))[: max(1, floor(len(names) * share))]
    }


def apply_cap_rank(clips: Iterable[Clip], hours: float = 30.0) -> Iterator[Clip]:
    """At most `hours` per speaker and source: a speaker's clips in rank order, stopping before the first that would
    exceed the cap. Clips without a speaker are not capped. The input order is kept.
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
            total += c.duration_s
            if total > hours * 3600:
                break
            kept.add(c.clip_id)
    return (c for c in clips if c.speaker is None or c.clip_id in kept)


# An adapter has `clips(raw)`; one that is not forced-aligned by MFA also has `timings(raw, clips)` and is capped here.
SOURCES: dict[str, ModuleType] = {"swc": swc}

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
    "rank": pl.UInt64,
}


def _frame(clips: list[Clip], reserved: set[tuple[str, str, str]]) -> pl.LazyFrame:
    rows = [
        dataclasses.asdict(c) | {"reserved": (c.split, c.language, c.speaker) in reserved, "rank": rank(c.clip_id)}
        for c in clips
    ]
    return pl.DataFrame(rows, schema=_SCHEMA).lazy()


def run_index(cfg: BuildConfig) -> None:
    """Write `work/index/<source>.parquet` for every source of `cfg`."""
    per_source: dict[str, list[Clip]] = {}
    for name in cfg.sources:
        if name not in SOURCES:
            raise ValueError(f"no index adapter for source {name!r}; have {sorted(SOURCES)}")
        module = SOURCES[name]
        clips = [c for c in module.clips(raw_dir(name)) if cfg.languages is None or c.language in cfg.languages]
        for c in clips:
            guard_path(c.path, c.split)
        # MFA sources get their cap in `align`, after the drop rules.
        per_source[name] = list(apply_cap_rank(clips)) if hasattr(module, "timings") else clips
    # Reservation sees every source of a language together, so a speaker is reserved once, not per source.
    reserved = reserve_speakers((c for clips in per_source.values() for c in clips), cfg.reserved_share)
    out = work_dir(cfg) / "index"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({out / f"{name}.parquet": _frame(clips, reserved) for name, clips in per_source.items()}))


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset index", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    ns = parser.parse_args(args)
    run_index({"slice": SLICE, "full": FULL}[ns.config])
