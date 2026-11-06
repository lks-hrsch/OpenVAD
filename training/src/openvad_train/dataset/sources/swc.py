# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Spoken Wikipedia Corpora 2.0: one clip per article, word timings from `aligned.swc`.

Adapter contract: `clips(raw, languages)` (language codes, None for all; it reads nothing of the others) and,
for sources that are not forced-aligned by MFA, `timings(raw, clips)`. `raw` is `datasets/raw/swc`.
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
from openvad_train.fetch.swc import audio_licence


SOURCE = "swc"
LANGUAGES = {"english": "en", "german": "de", "dutch": "nl"}
# Pointers are relative to $OPENVAD_DATA_DIR; `raw` itself is only the place to read from.
RAW_PREFIX = Path("datasets/raw/swc")

# Wiki prefixes of a reader field: `:nl:`, and a user namespace with optional language and project (`w:en:User:`).
_COLON_PREFIX = re.compile(r"^:(?:[a-z]{1,3}:)*", re.IGNORECASE)
_USER_NAMESPACE = re.compile(r"^(?:[a-z]{1,3}:)*(?:user|benutzer|gebruiker|speakthings):", re.IGNORECASE)
_HOME_WIKI = re.compile(r"~\w+wiki$", re.IGNORECASE)
_WIKI_LINK = re.compile(r"\[\[([^\]]+)\]\]")
_MANGLED = re.compile(r"(?:user at project|original uploader)(.+?)wikipedia[a-z]{2}", re.IGNORECASE)
_SPOKEN_BY = re.compile(r"spoken by:\s*(.+?)(?:\s+sound editing by:.*)?$", re.IGNORECASE | re.DOTALL)


def reader_name(raw: str | None) -> str | None:
    """One reader field with wiki-link forms and case merged: `[[User:Jcb|Jcb]]`, `[[Benutzer:Jcb]]`, `Jcb`,
    `jcb` and `user at projectJcbwikipedianl` are one name. The first link counts when there are several, and
    anything that is not a user form (a publisher, say) stays as written.
    """
    if not raw:
        return None
    text = raw
    if m := _SPOKEN_BY.search(text):
        text = m.group(1)
    if m := _WIKI_LINK.search(text):
        text = m.group(1).split("|")[0]
    elif m := _MANGLED.search(text):
        text = m.group(1)
    text = _HOME_WIKI.sub("", _USER_NAMESPACE.sub("", _COLON_PREFIX.sub("", text.strip())))
    return " ".join(text.replace("_", " ").split()).casefold() or None


def article_readers(info: dict) -> set[str]:
    """Every name an article's info.json gives its reader: the audio file's and the article's."""
    fields = ((info.get("audio_file_parsed") or {}).get("reader"), (info.get("article_parsed") or {}).get("reader"))
    return {name for field in fields if (name := reader_name(field))}


def merge_readers(articles: Iterable[set[str]]) -> dict[str, str]:
    """Canonical name per name: names that share an article are one person (connected components), and the
    alphabetically first name of a component stands for it.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for names in articles:
        for other in names:
            parent[find(other)] = find(min(names))
    components: dict[str, set[str]] = {}
    for name in parent:
        components.setdefault(find(name), set()).add(name)
    return {name: min(members) for members in components.values() for name in members}


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


def _licence(folder: Path) -> str:
    meta = folder / "audiometa.txt"
    return audio_licence(meta.read_text(errors="replace")) if meta.exists() else "unknown"


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    articles = []
    for language_dir, language in LANGUAGES.items():
        if languages is not None and language not in languages:
            continue
        for article in sorted((raw / language_dir).glob("*/aligned.swc")):
            folder = article.parent
            audio = folder / "audio.ogg"
            if not audio.exists():
                # shortcut: 135 aligned articles are split over audio1.ogg, audio2.ogg ... and their word times
                # run over the concatenation, so they need one joined file first; skipped until that is built.
                continue
            names = article_readers(json.loads((folder / "info.json").read_text()))
            articles.append((language_dir, language, folder, audio, names))
    canonical = merge_readers(names for *_, names in articles)
    for language_dir, language, folder, audio, names in articles:
        reader = canonical[min(names)] if names else None
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
            licence=_licence(folder),
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
    """Aligned words, and the masks: everything before the first and after the last aligned word, and each gap
    between two aligned words that has an unaligned token inside. Stretches the aligner never reached are unknown
    (ADR 0023), so they are masked and not labelled non-speech.
    """
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
            if words and unaligned and start > last_end:
                masks.append((last_end, start))
            unaligned = False
            words.append((start, end))
            last_end = max(last_end, end)
    if not words:
        return [], [(0, duration_ms)] if duration_ms > 0 else []
    first_start = min(s for s, _ in words)
    edges = [(0, first_start), (last_end, duration_ms)]
    return words, sorted([m for m in masks + edges if m[1] > m[0]])


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    for clip in clips:
        doc = ET.parse(raw / Path(clip.path).relative_to(RAW_PREFIX).parent / "aligned.swc").getroot().find("d")
        assert doc is not None
        speech, mask = _word_timings(doc, round(clip.duration_s * 1000))
        yield Timing(clip.clip_id, "word", speech, mask, None, "swc:aligned.swc")
