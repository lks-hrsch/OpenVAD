# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A synthetic SWC tree shaped like the real one (see docs/notes/datasets.md), for the SWC and index tests."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf


# Words are (start_ms, end_ms), or None for a token the aligner left untimed.
ARTICLE_WORDS = [
    (100, 300),
    None,
    (450, 600),
    (700, 800),
]


def _swc(words: list[tuple[int, int] | None], boilerplate: bool) -> str:
    tokens = "".join(
        '<t text="w"><n pronunciation="w" /></t>' if w is None else f'<t text="w"><n start="{w[0]}" end="{w[1]}" /></t>'
        for w in words
    )
    extra = '<extra><s><t text="intro"><n /></t></s><s><t text="outro"><n /></t></s></extra>' if boilerplate else ""
    return (
        "<article><meta /><d>"
        f"{extra}"
        f'<ignored><s><t text="image"><n /></t></s></ignored><p><s>{tokens}<t text="." /></s></p>'
        "</d></article>"
    )


def add_article(
    raw: Path,
    language: str,
    name: str,
    reader: str | None,
    *,
    article_reader: str | None = None,
    seconds: float = 2.0,
    words=ARTICLE_WORDS,
    swc=True,
    boilerplate=True,
    parts: tuple[float, ...] | None = None,
) -> None:
    folder = raw / language / name
    folder.mkdir(parents=True)
    # An article in several parts has audio1.ogg, audio2.ogg ... and no audio.ogg.
    audio = {"audio.ogg": seconds} if parts is None else {f"audio{i}.ogg": d for i, d in enumerate(parts, 1)}
    for name_, length in audio.items():
        sf.write(folder / name_, np.zeros(int(16000 * length)), 16000, format="OGG", subtype="VORBIS")
    info = {"article_parsed": {"reader": article_reader}, "audio_file_parsed": {"reader": reader}}
    (folder / "info.json").write_text(json.dumps(info))
    (folder / "audiometa.txt").write_text("{{self|cc-by-sa-3.0}}")
    if swc:
        (folder / "aligned.swc").write_text(_swc(words, boilerplate))


def make_swc_tree(raw: Path) -> Path:
    add_article(raw, "english", "Aaa", "[[User:Jcb|Jcb]]")
    add_article(raw, "english", "Bbb", "jcb")
    add_article(raw, "german", "Ccc", None)
    add_article(raw, "german", "Ddd", "Zed_Q")
    add_article(raw, "dutch", "Eee", "zed q")
    add_article(raw, "dutch", "Fff", "Other", swc=False)
    return raw
