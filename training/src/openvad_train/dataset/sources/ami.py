# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""AMI Meeting Corpus, individual headsets: one clip per participant of a meeting, labelled by the manual words.

Adapter contract: `MFA_ALIGNED`, `clips(raw, languages)` and `timings(raw, clips)`. `raw` is `datasets/raw/ami`. The
audio is read in place; a long meeting is cut into samples of at most 30 s by `plan` (`CUTTING`).
"""

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.schema import Clip, Split, Timing
from openvad_train.dataset.sources import wav_durations
from openvad_train.fetch.ami import AMI_PARTITION


SOURCE = "ami"
CUTTING = "split_over"
MFA_ALIGNED = False
RAW_PREFIX = Path("datasets/raw/ami")
# The stricter of the two statements is recorded, as for DEMAND (data/README.md).
LICENCE = "CC BY-NC-SA 2.0 (OpenSLR 16 copy; Edinburgh states CC BY 4.0)"
LANGUAGE = "en"
# The full-corpus-ASR partition names its splits train, dev and test.
_SPLITS: dict[str, Split] = {"train": "train", "dev": "validation", "test": "test"}
_SPLIT_OF_SESSION: dict[str, Split] = {
    session: _SPLITS[name] for name, sessions in AMI_PARTITION.items() for session in sessions
}
_HEADSET = ".Headset-"


@cache
def _participants(meetings_xml: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """(meeting, headset channel) to (the participant's `global_name`, the letter that names their words file)."""
    out = {}
    for meeting in ET.parse(meetings_xml).getroot().iter("meeting"):
        for speaker in meeting.findall("speaker"):
            out[meeting.attrib["observation"], int(speaker.attrib["channel"])] = (
                speaker.attrib["global_name"],
                speaker.attrib["nxt_agent"],
            )
    return out


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    """One clip per headset file of a meeting in the partition, split by the official partition (dev is
    validation), which is speaker-disjoint. The speaker is the participant, named across meetings.
    """
    if languages is not None and LANGUAGE not in languages:
        return
    participants = _participants(raw / "annotations/corpusResources/meetings.xml")
    found = []
    for (meeting, channel), (name, _) in sorted(participants.items()):
        wav = raw / "audio" / meeting / f"{meeting}{_HEADSET}{channel}.wav"
        if meeting[:6] in _SPLIT_OF_SESSION and wav.is_file():  # a headset the fetch listed as missing has no file
            found.append((meeting, channel, name, wav))
    for (meeting, channel, name, wav), duration in zip(found, wav_durations([f[3] for f in found]), strict=True):
        yield Clip(
            clip_id=f"{meeting}{_HEADSET}{channel}",
            source=SOURCE,
            language=LANGUAGE,
            speaker=name,
            split=_SPLIT_OF_SESSION[meeting[:6]],
            path=(RAW_PREFIX / wav.relative_to(raw)).as_posix(),
            offset_s=0.0,
            duration_s=duration,
            licence=LICENCE,
            transcript=None,
        )


def _words(file: Path) -> list[tuple[int, int]]:
    """The spoken words of one participant. Punctuation, vocal sounds (laughter, coughs), disfluency markers and
    gaps are other elements or carry `punc`, and so are not speech; a word without times (4 in the corpus) or whose
    interval does not run forward (11) has none to label and is dropped.
    """
    words = (w.attrib for w in ET.parse(file).getroot().iter("w") if w.get("punc") != "true")
    intervals = (
        (round(float(w["starttime"]) * 1000), round(float(w["endtime"]) * 1000))
        for w in words
        if "starttime" in w and "endtime" in w
    )
    return [(s, e) for s, e in intervals if e > s]


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    """The words of the wearer of the headset; what the other participants say into it is non-speech."""
    participants = _participants(raw / "annotations/corpusResources/meetings.xml")
    for clip in clips:
        meeting, channel = clip.clip_id.split(_HEADSET)
        _, agent = participants[meeting, int(channel)]
        speech = apply_gap_rule(_words(raw / "annotations/words" / f"{meeting}.{agent}.words.xml"), 0)
        yield Timing(clip.clip_id, "word", speech, [], None, "ami:manual-words")
