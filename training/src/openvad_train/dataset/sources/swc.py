# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Spoken Wikipedia Corpora 2.0: one clip per article, word timings from `aligned.swc`.

Adapter contract: `clips(raw)` and, for sources that are not forced-aligned by MFA,
`timings(raw, clips)`. `raw` is `datasets/raw/swc`.
"""

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import soundfile as sf

from openvad_train.dataset.schema import Clip, Timing, rank
from openvad_train.fetch.swc import swc_licences


SOURCE = "swc"
LANGUAGES = {"english": "en", "german": "de", "dutch": "nl"}
# Pointers are relative to $OPENVAD_DATA_DIR; `raw` itself is only the place to read from.
RAW_PREFIX = Path("datasets/raw/swc")

_USER_LINK = re.compile(r"\[\[:?(?:[a-z]{2}:)?user:([^|\]]+)", re.IGNORECASE)
_MANGLED_USER = re.compile(r"user at project(.+)wikipedia[a-z]{2}$", re.IGNORECASE)


def reader_name(info: dict) -> str | None:
    """The reader of an article's audio, with wiki-link forms and case merged.

    `[[User:Jcb|Jcb]]`, `Jcb`, `jcb` and `user at projectJcbwikipedianl` are one reader; with several
    links the first one counts, because one article gets one speaker.
    """
    raw = (info.get("audio_file_parsed") or {}).get("reader") or (info.get("article_parsed") or {}).get("reader")
    if not raw:
        return None
    name = m.group(1) if (m := _USER_LINK.search(raw) or _MANGLED_USER.search(raw)) else raw
    return " ".join(name.replace("_", " ").split()).casefold() or None


def split_of(reader: str | None) -> Literal["train", "validation", "test"]:
    """80/10/10 by reader, so every reader is in exactly one split; no reader means train, uncapped."""
    if reader is None:
        return "train"
    return ("train", "validation", "test")[max(0, rank(reader) % 10 - 7)]


def audio_duration_s(audio: Path) -> float:
    """Length from the file header. Some articles are Ogg with a Skeleton stream, which libsndfile cannot open."""
    try:
        return sf.info(audio).duration
    except sf.LibsndfileError:
        if shutil.which("ffprobe") is None:
            raise
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return float(out)


def clips(raw: Path) -> Iterator[Clip]:
    licences = swc_licences(raw)
    for language_dir, language in LANGUAGES.items():
        for article in sorted((raw / language_dir).glob("*/aligned.swc")):
            folder = article.parent
            info = json.loads((folder / "info.json").read_text())
            reader = reader_name(info)
            audio = folder / "audio.ogg"
            if not audio.exists():
                # shortcut: 135 aligned articles are split over audio1.ogg, audio2.ogg ... and their word times
                # run over the concatenation, so they need one joined file first; skipped until that is built.
                continue
            key = f"{language_dir}/{folder.name}"
            yield Clip(
                clip_id=key,
                source=SOURCE,
                language=language,
                speaker=reader,
                split=split_of(reader),
                path=(RAW_PREFIX / key / "audio.ogg").as_posix(),
                offset_s=0.0,
                # The header's length, not the last word's end: the mask after the last word needs it.
                duration_s=audio_duration_s(audio),
                licence=licences.get(key, "unknown"),
                transcript=None,
            )


def _span(n: ET.Element) -> tuple[int, int] | None:
    start, end = n.get("start"), n.get("end")
    return None if start is None or end is None else (int(start), int(end))


def _tokens(doc: ET.Element) -> list[ET.Element]:
    """Spoken tokens in time order; `ignored` parts (images, tables) are not read aloud.

    The boilerplate in `extra` is spoken before and after the article, but the file lists it in one block.
    shortcut: each `extra` sentence goes before the body when its first aligned word does, and a sentence
    with no aligned word goes first only if it opens the block; revisit if a corpus audit shows otherwise.
    """
    ignored = {t for e in doc.iter("ignored") for t in e.iter("t")}

    def first_start(tokens: Iterable[ET.Element]) -> int | None:
        return min((s[0] for t in tokens for n in t.findall("n") if (s := _span(n))), default=None)

    body = [t for part in doc if part.tag not in ("ignored", "extra") for t in part.iter("t") if t not in ignored]
    body_start = first_start(body)
    before: list[ET.Element] = []
    after: list[ET.Element] = []
    for extra in doc.findall("extra"):
        for i, sentence in enumerate(extra.findall("s")):
            tokens = list(sentence.iter("t"))
            start = first_start(tokens)
            leads = i == 0 if start is None or body_start is None else start < body_start
            (before if leads else after).extend(tokens)
    return before + body + after


def _word_timings(doc: ET.Element, duration_ms: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Aligned words, and the gaps between two aligned words (or the clip edges) with an unaligned token inside."""
    words: list[tuple[int, int]] = []
    masks: list[tuple[int, int]] = []
    last_end = 0
    unaligned = False
    for token in _tokens(doc):
        for n in token.findall("n"):  # a token without `n` is punctuation
            span = _span(n)
            if span is None:
                unaligned = True
                continue
            start, end = span
            if unaligned and start > last_end:
                masks.append((last_end, start))
            unaligned = False
            words.append((start, end))
            last_end = max(last_end, end)
    if unaligned and duration_ms > last_end:
        masks.append((last_end, duration_ms))
    return words, masks


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    for clip in clips:
        doc = ET.parse(raw / Path(clip.path).relative_to(RAW_PREFIX).parent / "aligned.swc").getroot().find("d")
        assert doc is not None
        speech, mask = _word_timings(doc, round(clip.duration_s * 1000))
        yield Timing(clip.clip_id, "word", speech, mask, None, "swc:aligned.swc")
