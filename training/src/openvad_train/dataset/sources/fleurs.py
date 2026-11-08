# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FLEURS: the official splits, without speaker IDs, labelled by MFA.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)`. `raw` is `datasets/raw/fleurs`.
"""

import csv
from collections.abc import Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip, Split
from openvad_train.fetch.fleurs import FLEURS_LANGS


SOURCE = "fleurs"
MFA_ALIGNED = True
RAW_PREFIX = Path("datasets/raw/fleurs")
LICENCE = "CC BY 4.0"
SPLITS: dict[str, Split] = {"train": "train", "dev": "validation", "test": "test"}
# A FLEURS directory is `<language>_<region>`; Mandarin is `cmn_hans_cn`.
LANGUAGE_OF_PREFIX = {"cmn": "zh"}


def _language(directory: str) -> str:
    prefix = directory.split("_", 1)[0]
    return LANGUAGE_OF_PREFIX.get(prefix, prefix)


def _split_clips(raw: Path, directory: str, language: str, official: str) -> Iterator[Clip]:
    """Columns of `<split>.tsv`: id, file, raw text, normalised text, characters, sample count at 16 kHz, gender. The
    normalised text has numbers written out, which MFA can align and a digit would turn into `<unk>`.
    """
    with (raw / directory / f"{official}.tsv").open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            _, name, _, text, _, samples, _ = row
            yield Clip(
                clip_id=f"{directory}/{Path(name).stem}",
                source=SOURCE,
                language=language,
                speaker=None,
                split=SPLITS[official],
                path=(RAW_PREFIX / directory / "audio" / official / name).as_posix(),
                offset_s=0.0,
                duration_s=int(samples) / 16000,
                licence=LICENCE,
                transcript=text,
            )


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    wanted = indexed_languages(languages)
    for directory in FLEURS_LANGS:
        if _language(directory) in wanted and (raw / directory).is_dir():
            for official in SPLITS:
                yield from _split_clips(raw, directory, _language(directory), official)
