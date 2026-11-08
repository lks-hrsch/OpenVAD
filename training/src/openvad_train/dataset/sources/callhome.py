# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""CallHome telephone calls from the TalkBank mirror: one clip per call, labelled by its speaker turns.

Adapter contract: `MFA_ALIGNED`, `BAND`, `clips(raw, languages)` and `timings(raw, clips)`. `raw` is
`datasets/raw/callhome`. The mirror embeds the audio of a call in a row of its Parquet shard, so a clip points at the
shard and its `row_in_file` (`audio_format = callhome_parquet`). The audio is 8 kHz telephone speech held in a 16 kHz
file and is used unchanged (`BAND`).
"""

import asyncio
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.parquet_io import read_parquet
from openvad_train.dataset.schema import Clip, Split, Timing, rank
from openvad_train.dataset.sources.swc import SPLIT_SHARES


SOURCE = "callhome"
CUTTING = "split_over"
MFA_ALIGNED = False
BAND = "narrow_native"
RAW_PREFIX = Path("datasets/raw/callhome")
LICENCE = "CC BY-NC-SA 4.0"
LANGUAGES = {"eng": "en", "deu": "de", "jpn": "ja", "spa": "es", "zho": "zh"}
# ADR 0022: every turn boundary of either side is masked this far to both sides.
MASK_MS = 250


def timing_from_turns(turns: Iterable[tuple[str, int, int]], clip_id: str = "") -> Timing:
    """`turns` are (side, start_ms, end_ms). The turns of the two sides overlap and are used as given: the speech is
    their union, with no gap rule, and every boundary of every turn is masked by `MASK_MS` (ADR 0022). A turn that
    does not run forward (the mirror has some of zero length) has no speech and no boundary.
    """
    spans = [(start, end) for _, start, end in turns if end > start]
    mask = sorted((max(0, boundary - MASK_MS), boundary + MASK_MS) for start, end in spans for boundary in (start, end))
    return Timing(clip_id, "turn", apply_gap_rule(spans, 0), mask, None, "callhome:turns")


@lru_cache(maxsize=64)
def _shard(path: Path) -> list[dict]:
    """The turns of every call of one shard, without its audio."""
    columns = ["timestamps_start", "timestamps_end", "speakers"]
    return asyncio.run(read_parquet(path, columns)).to_dicts()


def _call_timing(call: dict, clip_id: str) -> Timing:
    turns = zip(call["speakers"], call["timestamps_start"], call["timestamps_end"], strict=True)
    return timing_from_turns(((side, round(s * 1000), round(e * 1000)) for side, s, e in turns), clip_id)


def _split_by_rank(call_ids: list[str]) -> dict[str, Split]:
    """The first 80% of the IDs in rank order are train, the next 10% validation, the rest test."""
    ordered = sorted(call_ids, key=lambda c: (rank(c), c))
    train_end = round(len(ordered) * SPLIT_SHARES["train"])
    validation_end = round(len(ordered) * (SPLIT_SHARES["train"] + SPLIT_SHARES["validation"]))
    return {
        call_id: "train" if i < train_end else "validation" if i < validation_end else "test"
        for i, call_id in enumerate(ordered)
    }


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    """The calls are split 80/10/10 by call (ADR 0023) by the rank of their ID over all calls on disk, whatever
    `languages` asks for, so a slice and the full build give a call the same split. A call has no speaker ID that
    holds across calls, so the speaker is None. The clip is as long as the last turn, where the mirror cuts the audio.
    """
    calls = {
        f"{lang}/{shard.stem}/{row}": (lang, shard, row, call)
        for lang in LANGUAGES
        for shard in sorted((raw / lang).glob("*.parquet"))
        for row, call in enumerate(_shard(shard))
    }
    splits = _split_by_rank(list(calls))
    for call_id, (lang, shard, row, call) in calls.items():
        if languages is not None and LANGUAGES[lang] not in languages:
            continue
        yield Clip(
            clip_id=call_id,
            source=SOURCE,
            language=LANGUAGES[lang],
            speaker=None,
            split=splits[call_id],
            path=(RAW_PREFIX / lang / shard.name).as_posix(),
            offset_s=0.0,
            duration_s=round(max(call["timestamps_end"]), 3),
            licence=LICENCE,
            transcript=None,
            row_in_file=row,
        )


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    for clip in clips:
        assert clip.row_in_file is not None, clip.clip_id
        call = _shard(raw / Path(clip.path).relative_to(RAW_PREFIX))[clip.row_in_file]
        yield _call_timing(call, clip.clip_id)
