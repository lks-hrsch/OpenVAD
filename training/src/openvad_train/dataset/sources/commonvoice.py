# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Common Voice Scripted Speech 27.0: validated clips with the official dev and test, labelled by MFA.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)`. `raw` is `datasets/raw/common-voice`.
"""

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip, Split
from openvad_train.fetch.commonvoice import CV_LOCALES, read_tsv


log = logging.getLogger(__name__)

SOURCE = "common-voice"
MFA_ALIGNED = True
REPORTS = True
RAW_PREFIX = Path("datasets/raw/common-voice")
LICENCE = "CC0-1.0"
# A locale is its language code without the region, except these: Cantonese (`zh-HK` and `yue`) and Taiwan Mandarin are
# not the mainland Mandarin of `mandarin_china_mfa`.
LANGUAGE = {"zh-CN": "zh", "zh-HK": "yue", "zh-TW": "zh-TW"}
HELD_OUT: dict[str, Split] = {"dev": "validation", "test": "test"}


def language_of(locale: str) -> str:
    return LANGUAGE.get(locale, locale.split("-")[0])


def _complete(raw: Path, locale: str) -> bool:
    """The fetch writes `.<locale>-extracted` when a locale's clips are out and the raw manifest at the very end. A
    locale with neither is still being fetched, and its missing MP3s would be dropped for good by `align`.
    """
    if (raw / f".{locale}-extracted").exists():
        return True
    manifest = raw / "manifest.json"
    return manifest.is_file() and locale in json.loads(manifest.read_text()).get("locales", {})


def _locale_dir(raw: Path, locale: str) -> Path | None:
    """The folder with the TSVs and `clips/`, or None while the locale is not fetched (the fetch is resumable)."""
    found = sorted((raw / locale).glob("*/*/validated.tsv"))  # the fixed layout; `**` would walk `clips/` over SMB
    if len(found) > 1:
        raise RuntimeError(
            f"Common Voice {locale}: expected one validated.tsv under {raw / locale}, found {len(found)}"
        )
    return found[0].parent if found else None


def _locale_clips(
    raw: Path, locale: str, language: str, folder: Path, held_speakers: set[str], counts: Counter
) -> Iterator[Clip]:
    """Dev and test as the release has them. Train is every validated clip of a speaker who is in the dev or test
    of no locale (ADR 0023), also one that is in no `train.tsv`: the pool rule of `split_stats`, over all locales
    because a `client_id` is global. Clips left out for that are counted in `counts["held_out_speaker_clips"]`.

    `duration_s` is the length in `clip_durations.tsv`, which comes from the MP3 header: 5.6 to 7.5 ms too long on
    real files without a Xing header. That is fine for the index and the 30 h cap; `align` labels on the decoded
    length. Nothing is decoded here.
    """
    durations = {r["clip"]: int(r["duration[ms]"]) / 1000 for r in read_tsv(folder / "clip_durations.tsv")}
    official: dict[str, Split] = {}  # clip path to split
    for official_split, split in HELD_OUT.items():
        for row in read_tsv(folder / f"{official_split}.tsv"):
            official.setdefault(row["path"], split)
    prefix = RAW_PREFIX / folder.relative_to(raw) / "clips"
    for row in read_tsv(folder / "validated.tsv"):
        name = row["path"]
        if name in official:
            split = official[name]
        elif row["client_id"] in held_speakers:
            counts["held_out_speaker_clips"] += 1
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


def _held_out(folders: Iterable[Path]) -> tuple[set[str], int]:
    """The speakers of the official dev and test of every given locale, and how many are in a dev and a test."""
    dev = {r["client_id"] for folder in folders for r in read_tsv(folder / "dev.tsv")}
    test = {r["client_id"] for folder in folders for r in read_tsv(folder / "test.tsv")}
    return dev | test, len(dev & test)


def _count(folder: Path) -> int:
    with (folder / "validated.tsv").open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def clips(raw: Path, languages: set[str] | None = None, report: dict | None = None) -> Iterator[Clip]:
    """Locales on disk that are skipped, because their language has no row in `mfa_models.tsv` or because their
    fetch is not finished, are recorded in `report["skipped"]` with their validated clip counts.

    The held-out speakers come from the dev and test of every finished locale, whatever `languages` asks for, so a
    speaker is never train in one language and dev or test in another. Official dev and test rows stay as published;
    speakers who are dev in one locale and test in another are counted in `report["dev_test_speaker_overlaps"]`.
    """
    wanted = indexed_languages(languages)
    table = indexed_languages(None)
    report = {} if report is None else report
    skipped = report.setdefault("skipped", {})
    finished: dict[str, Path] = {}
    for locale in CV_LOCALES:
        if (folder := _locale_dir(raw, locale)) is None:
            continue
        if language_of(locale) not in table or not _complete(raw, locale):
            reason = "no row in mfa_models.tsv" if language_of(locale) not in table else "fetch not finished"
            skipped[locale] = {"reason": reason, "clips": _count(folder)}
            log.warning("Common Voice %s is skipped: %s", locale, reason)
        else:
            finished[locale] = folder
    held_speakers, overlaps = _held_out(finished.values())
    report["dev_test_speaker_overlaps"] = overlaps
    counts: Counter = Counter()
    for locale, folder in finished.items():
        if language_of(locale) in wanted:
            yield from _locale_clips(raw, locale, language_of(locale), folder, held_speakers, counts)
    report["held_out_speaker_clips"] = counts["held_out_speaker_clips"]
