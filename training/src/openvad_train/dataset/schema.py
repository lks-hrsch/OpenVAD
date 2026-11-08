# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Identifiers and the row types shared by every dataset stage."""

import hashlib
from dataclasses import dataclass
from typing import Literal


def _blake2_64(*parts: str) -> int:
    # NUL cannot occur in an id, so ("a", "bc") and ("ab", "c") hash differently.
    return int.from_bytes(hashlib.blake2b("\0".join(parts).encode(), digest_size=8).digest(), "big")


def rank(sample_id: str) -> int:
    """Stable random order of a sample; Small is a cut-off on it."""
    return _blake2_64(sample_id)


def sample_seed(global_seed: int, plan_version: str, sample_id: str) -> int:
    """Seed of every draw for one sample, so adding a source never moves another sample's draws."""
    return _blake2_64(str(global_seed), plan_version, sample_id)


Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class Clip:
    clip_id: str
    source: str
    language: str
    speaker: str | None
    split: Split
    path: str
    offset_s: float
    duration_s: float
    licence: str
    transcript: str | None
    reserved: bool = False


@dataclass(frozen=True)
class Timing:
    clip_id: str
    kind: Literal["word", "turn", "frame"]
    speech_ms: list[tuple[int, int]]
    mask_ms: list[tuple[int, int]]
    metrics: dict | None
    provenance: str
