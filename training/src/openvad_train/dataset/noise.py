# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The DEMAND noise pool: scene categories, split channels and reproducible draws.

City, babble, SSN and MUSAN join the pool in a later task.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf


# LibriVAD's categories, keyed by the first letter of the DEMAND scene name.
_CATEGORIES = {
    "D": "domestic",
    "N": "nature",
    "O": "office",
    "P": "public",
    "S": "street",
    "T": "transport",
}
# Channels 1 to 12 train, 13 and 14 validation, 15 and 16 test (owner, 2026-11-06).
_CHANNELS = {"train": range(1, 13), "validation": range(13, 15), "test": range(15, 17)}


class NoiseDraw(NamedTuple):
    file: Path
    channel: int
    offset_n: int


def category_of_demand(scene: str) -> str:
    if scene[:1] not in _CATEGORIES:
        raise ValueError(f"not a DEMAND scene: {scene!r}")
    return _CATEGORIES[scene[0]]


def demand_channels(split: str) -> range:
    if split not in _CHANNELS:
        raise ValueError(f"unknown split: {split!r}")
    return _CHANNELS[split]


class NoisePool:
    """DEMAND under `root` as `<SCENE>/ch01.wav` to `ch16.wav`.

    A draw is a pure function of the rng and the sorted scene names, never of
    directory listing order, so a plan row reproduces it.
    """

    def __init__(self, root: Path) -> None:
        scenes = sorted(p for p in root.iterdir() if p.name[:1] in _CATEGORIES and (p / "ch01.wav").is_file())
        self._scenes = {c: [p for p in scenes if category_of_demand(p.name) == c] for c in _CATEGORIES.values()}
        # A header read costs about 50 ms cold on the share, which adds up over millions of draws.
        self._frames: dict[Path, int] = {}

    def draw(self, category: str, split: str, rng: np.random.Generator) -> NoiseDraw:
        if category not in self._scenes:
            raise ValueError(f"not a DEMAND category: {category!r}")
        scenes = self._scenes[category]
        if not scenes:
            raise ValueError(f"no DEMAND scene in category {category!r}")
        channels = demand_channels(split)
        scene = scenes[int(rng.integers(len(scenes)))]
        channel = int(rng.integers(channels.start, channels.stop))
        file = scene / f"ch{channel:02d}.wav"
        if file not in self._frames:
            self._frames[file] = sf.info(file).frames
        # The offset is uniform over the file; the caller wraps it around with `loop_slice`.
        return NoiseDraw(file, channel, int(rng.integers(self._frames[file])))
