# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `index`: what clips exist, which split each is in, and which speakers are capped or reserved."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from math import floor
from pathlib import PurePosixPath

from openvad_train.dataset.schema import Clip, rank


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
