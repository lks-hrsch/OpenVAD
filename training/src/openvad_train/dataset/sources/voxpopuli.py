# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""VoxPopuli transcribed data: segments of the session audio, `invalid` rows dropped, our own split by speaker.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)`. `raw` is `datasets/raw/voxpopuli`.
"""

import csv
import gzip
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip
from openvad_train.dataset.sources import count_lines
from openvad_train.dataset.sources.swc import assign_splits
from openvad_train.fetch.voxpopuli import VOXPOPULI_LANGS


SOURCE = "voxpopuli"
CUTTING = "split_over"
MFA_ALIGNED = True
REPORTS = True
RAW_PREFIX = Path("datasets/raw/voxpopuli")
LICENCE = "CC0-1.0"
csv.field_size_limit(1 << 30)  # the `vad` column of a long segment


def _rows(table: Path) -> Iterator[dict[str, str]]:
    with gzip.open(table, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if row["split"] != "invalid":  # the text does not match the audio
                yield row


def _unique(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """One row per `id_`. The release lists a paragraph both in a part session file and in the whole-day file, with
    the same text and speaker; the row with the smallest start offset wins (ties: the path), which is the part file
    and avoids seeking hours into a whole-day Ogg. Returns the rows and how many were removed.
    """
    best: dict[str, dict[str, str]] = {}
    total = 0
    for r in rows:
        total += 1
        key = lambda row: (float(row["start_time"]), row["session_id"])  # noqa: E731
        if r["id_"] not in best or key(r) < key(best[r["id_"]]):
            best[r["id_"]] = r
    return list(best.values()), total - len(best)


def clips(raw: Path, languages: set[str] | None = None, report: dict | None = None) -> Iterator[Clip]:
    """The official splits share speakers, so a speaker goes to train, validation or test as a whole: 80/10/10 by
    hours (the SWC rule, which gives a speaker heard in several languages one split). A row without a speaker ID
    cannot be split and is train only (ADR 0023). The hours of every language on disk count, so the split does not
    depend on the languages asked for.
    The text is `normed_text`, which writes numbers out but drops letters outside the language's alphabet in about
    0.3 to 1.6% of rows.
    """
    wanted = indexed_languages(languages)
    table = indexed_languages(None)
    tables = {
        language: path
        for language in VOXPOPULI_LANGS
        if (path := raw / "annotations" / f"asr_{language}.tsv.gz").is_file()
    }
    if report is not None:
        report["skipped"] = {
            language: {"reason": "no row in mfa_models.tsv", "clips": count_lines(path) - 1}
            for language, path in tables.items()
            if language not in table
        }
    # Hours of every table on disk, whatever `languages` asks for or the models table lists, so a speaker's split
    # does not move with the request.
    rows: dict[str, list[dict[str, str]]] = {}
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for language, path in tables.items():
        language_rows, removed = _unique(_rows(path))
        if report is not None:
            report["duplicate_ids_removed"] = report.get("duplicate_ids_removed", 0) + removed
        if language in wanted:
            rows[language] = language_rows
        for r in language_rows:
            if r["speaker_id"] != "None":
                hours[r["speaker_id"]][language] += (float(r["end_time"]) - float(r["start_time"])) / 3600
    splits = assign_splits(hours)
    for language, language_rows in rows.items():
        for r in language_rows:
            speaker = None if r["speaker_id"] == "None" else r["speaker_id"]
            start, end = float(r["start_time"]), float(r["end_time"])
            session = r["session_id"]
            yield Clip(
                clip_id=r["id_"],
                source=SOURCE,
                language=language,
                speaker=speaker,
                split="train" if speaker is None else splits[speaker],
                path=(RAW_PREFIX / "raw_audios/original" / session[:4] / f"{session}_original.ogg").as_posix(),
                offset_s=start,
                duration_s=round(end - start, 3),
                licence=LICENCE,
                transcript=r["normed_text"],
            )
