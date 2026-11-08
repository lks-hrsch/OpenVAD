# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Multilingual LibriSpeech: the official speaker-disjoint splits, labelled by MFA.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)`. `raw` is `datasets/raw/mls`.
"""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip, Split
from openvad_train.dataset.sources import count_lines


SOURCE = "mls"
CUTTING = "cut_long"
MFA_ALIGNED = True
REPORTS = True
RAW_PREFIX = Path("datasets/raw/mls")
LICENCE = "CC BY 4.0"
LANGUAGE = {
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
}
SPLITS: dict[str, Split] = {"train": "train", "dev": "validation", "test": "test"}


def _speakers(raw: Path, name: str, official: str) -> set[str]:
    lines = (raw / f"mls_{name}" / official / "transcripts.txt").read_text().splitlines()
    return {line.split("_", 1)[0] for line in lines}


def _split_clips(
    raw: Path, name: str, language: str, official: str, held_speakers: set[str], counts: Counter
) -> Iterator[Clip]:
    """One clip per line of `transcripts.txt` (IDs are `<speaker>_<book>_<segment>`). A train clip of a speaker who
    is in the dev or test of any language is left out and counted: reader ids are global, and a reader can record in
    several languages.

    `duration_s` is the segment's `end - start` in `segments.txt`, which agrees with the FLAC header to 4 ms on 150
    sampled files; reading 1.6 M headers off the share would take hours. `align` labels on the decoded length.
    """
    folder = raw / f"mls_{name}" / official
    segments = {}
    for line in (folder / "segments.txt").read_text().splitlines():
        segment, _, start, end = line.split("\t")
        segments[segment] = round(float(end) - float(start), 2)
    for line in (folder / "transcripts.txt").read_text().splitlines():
        segment, text = line.split("\t", 1)
        speaker, book, _ = segment.split("_")
        if official == "train" and speaker in held_speakers:
            counts["held_out_speaker_clips"] += 1
            continue
        yield Clip(
            clip_id=f"{language}/{segment}",
            source=SOURCE,
            language=language,
            speaker=speaker,
            split=SPLITS[official],
            path=(RAW_PREFIX / f"mls_{name}" / official / "audio" / speaker / book / f"{segment}.flac").as_posix(),
            offset_s=0.0,
            duration_s=segments[segment],
            licence=LICENCE,
            transcript=text,
        )


def clips(raw: Path, languages: set[str] | None = None, report: dict | None = None) -> Iterator[Clip]:
    """A language folder on disk whose language has no row in `mfa_models.tsv` is recorded in `report["skipped"]`."""
    wanted = indexed_languages(languages)
    table = indexed_languages(None)
    skipped = report.setdefault("skipped", {}) if report is not None else {}
    for folder in sorted(raw.glob("mls_*")):
        if LANGUAGE.get(folder.name.removeprefix("mls_")) not in table:
            skipped[folder.name] = {
                "reason": "no row in mfa_models.tsv",
                "clips": sum(count_lines(f) for f in folder.glob("*/transcripts.txt")),
            }
    present = {name: language for name, language in LANGUAGE.items() if (raw / f"mls_{name}").is_dir()}
    # From every language on disk, whatever `languages` asks for, so a split does not depend on the request.
    dev = {s for name in present for s in _speakers(raw, name, "dev")}
    test = {s for name in present for s in _speakers(raw, name, "test")}
    if report is not None:
        report["dev_test_speaker_overlaps"] = len(dev & test)
    counts: Counter = Counter()
    for name, language in present.items():
        if language in wanted:
            for official in SPLITS:
                yield from _split_clips(raw, name, language, official, dev | test, counts)
    if report is not None:
        report["held_out_speaker_clips"] = counts["held_out_speaker_clips"]
