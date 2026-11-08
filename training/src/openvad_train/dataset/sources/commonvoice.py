# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Common Voice Scripted Speech 27.0: validated clips with the official dev and test, labelled by MFA.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)`. `raw` is `datasets/raw/common-voice`.
"""

from collections.abc import Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip, Split
from openvad_train.fetch.commonvoice import CV_LOCALES, read_tsv


SOURCE = "common-voice"
MFA_ALIGNED = True
RAW_PREFIX = Path("datasets/raw/common-voice")
LICENCE = "CC0-1.0"
# A locale is its own language code except these; zh-HK and zh-TW are not the mainland Mandarin of `mandarin_china_mfa`.
LANGUAGE = {"zh-CN": "zh", "sv-SE": "sv"}
HELD_OUT: dict[str, Split] = {"dev": "validation", "test": "test"}


def _locale_dir(raw: Path, locale: str) -> Path | None:
    """The folder with the TSVs and `clips/`, or None while the locale is not fetched (the fetch is resumable)."""
    found = sorted((raw / locale).glob("**/validated.tsv"))
    if len(found) > 1:
        raise RuntimeError(
            f"Common Voice {locale}: expected one validated.tsv under {raw / locale}, found {len(found)}"
        )
    return found[0].parent if found else None


def _locale_clips(raw: Path, locale: str, language: str, folder: Path) -> Iterator[Clip]:
    """Dev and test as the release has them. Train is every validated clip of a speaker in neither (ADR 0023), also
    one that is in no `train.tsv`, which is the pool rule of `split_stats`.

    `duration_s` is the length in `clip_durations.tsv`, which comes from the MP3 header: 5.6 to 7.5 ms too long on
    real files without a Xing header. That is fine for the index and the 30 h cap; `align` labels on the decoded
    length. Nothing is decoded here.
    """
    durations = {r["clip"]: int(r["duration[ms]"]) / 1000 for r in read_tsv(folder / "clip_durations.tsv")}
    held: dict[str, Split] = {}  # clip path to split
    held_speakers: set[str] = set()
    for official, split in HELD_OUT.items():
        for row in read_tsv(folder / f"{official}.tsv"):
            held.setdefault(row["path"], split)
            held_speakers.add(row["client_id"])
    prefix = RAW_PREFIX / folder.relative_to(raw) / "clips"
    for row in read_tsv(folder / "validated.tsv"):
        name = row["path"]
        if name in held:
            split = held[name]
        elif row["client_id"] in held_speakers:
            continue
        else:
            split = "train"
        if name not in durations:
            raise ValueError(f"Common Voice {locale}: {name} is not in clip_durations.tsv")
        yield Clip(
            clip_id=Path(name).stem,
            source=SOURCE,
            language=language,
            speaker=row["client_id"],
            split=split,
            path=(prefix / name).as_posix(),
            offset_s=0.0,
            duration_s=durations[name],
            licence=LICENCE,
            transcript=row["sentence"],
        )


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    wanted = indexed_languages(languages)
    for locale in CV_LOCALES:
        language = LANGUAGE.get(locale, locale)
        if language in wanted and (folder := _locale_dir(raw, locale)) is not None:
            yield from _locale_clips(raw, locale, language, folder)
