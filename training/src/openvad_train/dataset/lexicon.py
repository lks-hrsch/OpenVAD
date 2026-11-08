# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Rule-based lexicons for languages with an acoustic model and no lexicon (ADR 0025, spec section `align`).

The route is Epitran. XPF is not supported because no language needs it yet. A lexicon is only useful if its phones
are ones the acoustic model knows, so `build_lexicon` takes the model's phone inventory (`model_phones`) and leaves
out every word with a phone outside it. No phone mapping is applied: whether a language's Epitran segmentation fits
its model is what the quality bar and the failure rule judge, and a mapping would hide the mismatch.
"""

import importlib
import json
import zipfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path


# Epitran's code for each language of the lexicon route (`lexicon route` in the note column of `mfa_models.tsv`).
EPITRAN_CODES = {"ca": "cat-Latn"}

SYNC_COMMAND = "uv sync --all-packages --group align"


class ModelMetaError(RuntimeError):
    """An acoustic model zip has no `meta.json`, so it has no phone inventory to read."""


class EpitranMissingError(RuntimeError):
    """Epitran is not installed: it lives in the `align` dependency group, which `uv sync` does not install."""


def model_phones(acoustic: Path) -> frozenset[str]:
    """The phone inventory in the `meta.json` of a downloaded MFA acoustic model (a zip)."""
    with zipfile.ZipFile(acoustic) as z:
        meta = next((n for n in z.namelist() if n.endswith("meta.json")), None)
        if meta is None:
            raise ModelMetaError(f"{acoustic} has no meta.json: not an MFA acoustic model")
        return frozenset(json.loads(z.read(meta))["phones"])


def _epitran(language: str) -> Callable[[str], list[str]]:
    if language not in EPITRAN_CODES:
        raise ValueError(f"no Epitran code for language {language!r}; have {sorted(EPITRAN_CODES)}")
    try:
        epitran = importlib.import_module("epitran")
    except ImportError as e:
        raise EpitranMissingError(f"Epitran is not installed: run `{SYNC_COMMAND}`") from e
    return epitran.Epitran(EPITRAN_CODES[language]).trans_list


def build_lexicon(
    words: Iterable[str],
    language: str,
    phones: Collection[str] | None = None,
    transliterate: Callable[[str], list[str]] | None = None,
    left_out: list[str] | None = None,
) -> dict[str, str]:
    """Word to its space-separated phones, the line format of an MFA dictionary.

    `phones` is the acoustic model's inventory (`model_phones`); a word with a phone outside it is not in the result
    and is appended to `left_out` when that list is given, so the caller can count the words lost to a phone set that
    does not match. A word with no phones at all is skipped without being counted. `transliterate` replaces Epitran
    (a word to its list of phones), for tests. Epitran can emit a bare combining mark as a phone (Catalan `cançó`);
    with a model inventory such a word is left out, as it should be.
    """
    transliterate = transliterate or _epitran(language)
    lexicon: dict[str, str] = {}
    for word in dict.fromkeys(words):
        pron = transliterate(word)
        if not pron:
            continue
        if phones is not None and not all(p in phones for p in pron):
            if left_out is not None:
                left_out.append(word)
            continue
        lexicon[word] = " ".join(pron)
    return lexicon
