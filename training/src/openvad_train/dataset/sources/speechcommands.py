# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Google Speech Commands v0.02: one-second words labelled by MFA, and the background noise, which is non-speech.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)` and `timings(raw, clips)` for the clips without a transcript.
`raw` is `datasets/raw/speech-commands`. Only the main archive is read: its `testing_list.txt` is the test split and
the separate 4,890-clip test set is never used.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

from openvad_train.dataset.models import indexed_languages
from openvad_train.dataset.schema import Clip, Split, Timing
from openvad_train.dataset.sources import wav_durations


SOURCE = "speech-commands"
CUTTING = None
MFA_ALIGNED = True
ARCHIVE = "speech_commands_v0.02"
RAW_PREFIX = Path("datasets/raw/speech-commands") / ARCHIVE
LICENCE = "CC BY 4.0"
LANGUAGE = "en"
LISTS: dict[str, Split] = {"validation_list.txt": "validation", "testing_list.txt": "test"}
NOISE_FOLDER = "_background_noise_"


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    """A word clip is named `<word>/<speaker hash>_nohash_<n>.wav` and has the word as its transcript. The lists
    hold validation and test; the rest is train. A background noise file has no speaker and no transcript, and is
    train; its labels come from `timings`.
    """
    if LANGUAGE not in indexed_languages(languages) or not (root := raw / ARCHIVE).is_dir():
        return
    named: dict[str, Split] = {}
    for listing, split in LISTS.items():
        named |= dict.fromkeys((root / listing).read_text().split(), split)
    files = sorted(root.glob("*/*.wav"))
    for file, duration in zip(files, wav_durations(files), strict=True):
        word = file.parent.name
        relative = f"{word}/{file.name}"
        noise = word == NOISE_FOLDER
        yield Clip(
            clip_id=f"{word}/{file.stem}",
            source=SOURCE,
            language=LANGUAGE,
            speaker=None if noise else file.name.split("_nohash_", 1)[0],
            split="train" if noise else named.get(relative, "train"),
            path=(RAW_PREFIX / relative).as_posix(),
            offset_s=0.0,
            duration_s=duration,
            licence=LICENCE,
            transcript=None if noise else word,
        )


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    """Background noise is non-speech by definition: no speech and no mask."""
    for clip in clips:
        if clip.transcript is not None:
            raise ValueError(f"{clip.clip_id}: a spoken clip is labelled by MFA, not by this adapter")
        yield Timing(clip.clip_id, "frame", [], [], None, f"{SOURCE}:background-noise")
