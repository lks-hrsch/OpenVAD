# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from align_support import FIXTURES
from openvad_train.dataset.textgrid import parse_textgrid


def test_a_real_textgrid_parses_to_both_tiers():
    tiers = parse_textgrid((FIXTURES / "u000002.TextGrid").read_text(encoding="utf-8"))
    assert set(tiers) == {"words", "phones"}
    assert tiers["words"][1] == (0.03, 0.23, "wir")
    assert tiers["phones"][2] == (0.06, 0.14, "i\u02d0")


def test_a_short_format_textgrid_is_refused():
    with pytest.raises(ValueError, match="long format"):
        parse_textgrid('"ooTextFile"\n"TextGrid"\n0\n1\n<exists>\n')


def _speech_tier(*cells: str, size: int | None = None) -> str:
    """A one-tier TextGrid whose `cells` are the raw `text = ...` values (quotes and line breaks included)."""
    lines = ['class = "IntervalTier"', 'name = "speech"', f"intervals: size = {len(cells) if size is None else size}"]
    for i, cell in enumerate(cells):
        lines += [f"intervals [{i + 1}]:", f"xmin = {i}", f"xmax = {i + 1}", f"text = {cell}"]
    return "\n".join(lines) + "\n"


def test_a_label_that_spans_lines_is_one_interval_with_its_text():
    tiers = parse_textgrid(_speech_tier('"first\nxmin = 9\nsecond ""quoted"""', '""'))
    assert tiers["speech"] == [(0.0, 1.0, 'first\nxmin = 9\nsecond "quoted"'), (1.0, 2.0, "")]


def test_a_tier_with_fewer_intervals_than_it_says_is_refused():
    with pytest.raises(ValueError, match="says 3 intervals but 2 were read"):
        parse_textgrid(_speech_tier('""', '"speech"', size=3))


def test_a_label_that_is_never_closed_is_refused():
    with pytest.raises(ValueError, match="never closed"):
        parse_textgrid(_speech_tier('"speech\nmore'))
