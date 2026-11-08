# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Praat TextGrid parsing, shared by `align` and the adapters that read ready-made alignments (VOiCES)."""

import re


def _closes(quoted: str) -> bool:
    """Whether `quoted`, which starts with a quote, ends its string: a doubled quote inside is an escaped quote."""
    i = 1
    while i < len(quoted):
        if quoted[i] == '"':
            if quoted[i + 1 : i + 2] != '"':
                return i == len(quoted.rstrip()) - 1
            i += 1
        i += 1
    return False


def parse_textgrid(text: str) -> dict[str, list[tuple[float, float, str]]]:
    """The interval tiers of a TextGrid in Praat's long format, which MFA writes: name to (xmin, xmax, text).

    A quoted text may span lines, as Praat writes a label with a newline in it, and a doubled quote is a quote.
    A tier whose parsed intervals differ in number from its `intervals: size` raises, so a truncated or damaged
    file cannot lose intervals silently.
    """
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    sizes: dict[str, int] = {}
    tier: list[tuple[float, float, str]] | None = None
    name = ""
    interval_tier = False
    xmin = xmax = None
    lines = iter(text.splitlines())
    for line in lines:
        s = line.strip()
        if s.startswith("class = "):
            interval_tier, tier = s == 'class = "IntervalTier"', None
        elif s.startswith("name = ") and interval_tier:
            name = s[len("name = ") :].strip().strip('"')
            tier = tiers.setdefault(name, [])
        elif tier is not None and (m := re.fullmatch(r"intervals: size = (\d+)", s)):
            sizes[name] = int(m.group(1))
        elif tier is not None and s.startswith("intervals ["):
            xmin = xmax = None
        elif tier is not None and s.startswith("xmin = "):
            xmin = float(s[len("xmin = ") :])
        elif tier is not None and s.startswith("xmax = "):
            xmax = float(s[len("xmax = ") :])
        elif tier is not None and s.startswith('text = "') and xmin is not None and xmax is not None:
            quoted = s[len("text = ") :]
            while not _closes(quoted):
                following = next(lines, None)
                if following is None:
                    raise ValueError(f"tier {name!r}: a label is never closed")
                quoted += "\n" + following
            tier.append((xmin, xmax, quoted.rstrip()[1:-1].replace('""', '"')))
            xmin = xmax = None
    if not tiers:
        raise ValueError("no interval tier found: expected a TextGrid in the long format")
    for name, size in sizes.items():
        if len(tiers[name]) != size:
            raise ValueError(f"tier {name!r} says {size} intervals but {len(tiers[name])} were read: damaged TextGrid")
    return tiers
