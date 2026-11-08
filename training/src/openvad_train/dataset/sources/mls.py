# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Multilingual LibriSpeech: the official speaker-disjoint splits, labelled by MFA.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)`. `raw` is `datasets/raw/mls`.
"""

from collections.abc import Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip, Split


SOURCE = "mls"
MFA_ALIGNED = True
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


def _split_clips(raw: Path, name: str, language: str, official: str) -> Iterator[Clip]:
    """One clip per line of `transcripts.txt` (IDs are `<speaker>_<book>_<segment>`).

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


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    wanted = indexed_languages(languages)
    for name, language in LANGUAGE.items():
        if language in wanted and (raw / f"mls_{name}").is_dir():
            for official in SPLITS:
                yield from _split_clips(raw, name, language, official)
