# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Synthetic WHAM! and MUSAN trees, and reserved talkers, for the noise and plan tests."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from openvad_train.dataset.noise import musan_noise_categories
from openvad_train.dataset.schema import Clip


def make_wham_tree(root: Path, per_split: int = 5, n: int = 1600) -> Path:
    """Stereo float wavs under `wham_noise/tr`, `cv` and `tt`, as the real download has them."""
    for split in ("tr", "cv", "tt"):
        (root / "wham_noise" / split).mkdir(parents=True)
        for i in range(per_split):
            x = np.random.default_rng(i).standard_normal((n, 2)).astype(np.float32) * 0.1
            sf.write(root / "wham_noise" / split / f"{split}{i:03d}.wav", x, 16000, subtype="FLOAT")
    (root / "manifest.json").write_text(json.dumps({"licence": "CC BY-NC 4.0"}))
    return root


def make_musan_tree(root: Path, n: int = 160, licence: str = "CC BY 3.0") -> Path:
    """Every MUSAN noise file the category table uses, as short noise, under `musan/noise/<folder>/`."""
    for file_id, category in musan_noise_categories().items():
        if category != "unused":
            path = root / "musan/noise" / file_id.removeprefix("noise-").rsplit("-", 1)[0] / f"{file_id}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, 0.1 * np.random.default_rng(len(file_id)).standard_normal(n), 16000, subtype="PCM_16")
    (root / "manifest.json").write_text(json.dumps({"licence": licence}))
    return root


def talker(
    clip_id: str, language: str = "de", split: str = "train", speaker: str | None = None, source: str = "mls"
) -> Clip:
    """A reserved whole-file clip of 4 s, as the pool takes it."""
    return Clip(
        clip_id,
        source,
        language,
        speaker or f"spk-{clip_id}",
        split,  # pyright: ignore[reportArgumentType]
        f"datasets/raw/{source}/{clip_id}.flac",
        0.0,
        4.0,
        "CC BY 4.0",
        None,
        True,
    )
