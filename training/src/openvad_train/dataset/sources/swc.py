# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Spoken Wikipedia Corpora 2.0: one clip per audio file of an article, word timings from `aligned.swc`.

Adapter contract: `clips(raw, languages)` (language codes, None for all; it reads nothing of the others) and,
for sources that are not forced-aligned by MFA, `timings(raw, clips)`. `raw` is `datasets/raw/swc`.
"""

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable, Iterator
from functools import cache, lru_cache
from pathlib import Path
from typing import Literal

import soundfile as sf

from openvad_train.dataset.config import SPEAKER_CAP_HOURS
from openvad_train.dataset.labels import apply_gap_rule
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


SPLIT_SHARES: dict[Literal["train", "validation", "test"], float] = {"train": 0.8, "validation": 0.1, "test": 0.1}


def assign_splits(hours: dict[str, float]) -> dict[str, Literal["train", "validation", "test"]]:
    """80/10/10 by reader and by aligned hours (ADR 0023): readers in rank order, each placed in the split furthest
    below its target share once the reader is counted, with the hours capped at 30 h as the cap will leave them.
    A reader over the cap goes to train. Ties go to the earlier split of `SPLIT_SHARES`.
    """
    placed = dict.fromkeys(SPLIT_SHARES, 0.0)
    assigned: dict[str, Literal["train", "validation", "test"]] = {}
    for reader in sorted(hours, key=lambda r: (rank(r), r)):
        counted = min(hours[reader], SPEAKER_CAP_HOURS)
        total = sum(placed.values()) + counted
        assigned[reader] = (
            "train"
            if hours[reader] > SPEAKER_CAP_HOURS
            else max(SPLIT_SHARES, key=lambda s: SPLIT_SHARES[s] * total - placed[s])
        )
        placed[assigned[reader]] += counted
    return assigned


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


def _parts(folder: Path) -> list[Path]:
    """An article's audio: one `audio.ogg`, or `audio1.ogg`, `audio2.ogg` ... in order."""
    single = folder / "audio.ogg"
    if single.exists():
        return [single]
    parts = sorted(folder.glob("audio[0-9]*.ogg"), key=lambda p: int(p.stem.removeprefix("audio")))
    if not parts:
        raise FileNotFoundError(f"{folder}: an aligned article without audio")
    return parts


@cache
def _durations(folder: Path) -> tuple[float, ...]:
    return tuple(audio_duration_s(p) for p in _parts(folder))


def clips(raw: Path, languages: set[str] | None = None) -> Iterator[Clip]:
    """One clip per audio file; an article in several parts gives `<language>/<article>#<part>` per part."""
    articles = []
    for language_dir, language in LANGUAGES.items():
        if languages is not None and language not in languages:
            continue
        for article in sorted((raw / language_dir).glob("*/aligned.swc")):
            folder = article.parent
            names = article_readers(json.loads((folder / "info.json").read_text()))
            articles.append((language_dir, language, folder, names))
    canonical = merge_readers(names for *_, names in articles)
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, language, folder, names in articles:
        if names:
            speech = apply_gap_rule(_article_timings(folder)[0], 0)
            hours[language][canonical[min(names)]] += sum(e - s for s, e in speech) / 3_600_000
    splits = {language: assign_splits(per_reader) for language, per_reader in hours.items()}
    for language_dir, language, folder, names in articles:
        reader = canonical[min(names)] if names else None
        split = splits[language][reader] if reader else "train"
        parts = _parts(folder)
        for part, duration in zip(parts, _durations(folder), strict=True):
            key = f"{language_dir}/{folder.name}"
            yield Clip(
                clip_id=key if len(parts) == 1 else f"{key}#{part.stem.removeprefix('audio')}",
                source=SOURCE,
                language=language,
                speaker=reader,
                split=split,
                path=(RAW_PREFIX / key / part.name).as_posix(),
                offset_s=0.0,
                # The header's length, not the last word's end: the mask after the last word needs it.
                duration_s=duration,
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


@lru_cache(maxsize=4)
def _article_timings(folder: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Words and masks over the article's whole timeline, which runs over the concatenation of its parts."""
    doc = ET.parse(folder / "aligned.swc").getroot().find("d")
    assert doc is not None
    return _word_timings(doc, round(sum(_durations(folder)) * 1000))


def _rebase(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """The parts of the intervals inside [start, end), counted from `start`; a word crossing a boundary is cut."""
    return [(max(s, start) - start, min(e, end) - start) for s, e in intervals if min(e, end) > max(s, start)]


def timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]:
    for clip in clips:
        path = Path(clip.path).relative_to(RAW_PREFIX)
        folder = raw / path.parent
        durations = _durations(folder)
        k = [p.name for p in _parts(folder)].index(path.name)
        start, end = (round(sum(durations[:i]) * 1000) for i in (k, k + 1))
        speech, mask = _article_timings(folder)
        yield Timing(
            clip.clip_id, "word", _rebase(speech, start, end), _rebase(mask, start, end), None, "swc:aligned.swc"
        )
