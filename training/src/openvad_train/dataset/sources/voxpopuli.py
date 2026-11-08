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
from collections.abc import Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip
from openvad_train.dataset.sources.swc import assign_splits
from openvad_train.fetch.voxpopuli import VOXPOPULI_LANGS


SOURCE = "voxpopuli"
MFA_ALIGNED = True
RAW_PREFIX = Path("datasets/raw/voxpopuli")
LICENCE = "CC0-1.0"


def _rows(table: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(1 << 30)  # the `vad` column of a long segment
    with gzip.open(table, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if row["split"] != "invalid":  # the text does not match the audio
                yield row


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    """The official splits share speakers, so a speaker goes to train, validation or test as a whole: 80/10/10 by
    hours (the SWC rule, which gives a speaker heard in several languages one split). A row without a speaker ID
    cannot be split and is train only (ADR 0023). A speaker's split depends on the languages asked for, as for SWC.
    The text is `normed_text`, with numbers written out.
    """
    wanted = indexed_languages(languages)
    rows: dict[str, list[dict[str, str]]] = {}
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for language in VOXPOPULI_LANGS:
        table = raw / "annotations" / f"asr_{language}.tsv.gz"
        if language in wanted and table.is_file():
            rows[language] = list(_rows(table))
            for r in rows[language]:
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
