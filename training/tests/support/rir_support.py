# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A synthetic RIR tree as `fetch rirs` leaves it: wavs under `but/` and `openslr28/`, and `index.tsv`."""

from pathlib import Path

import numpy as np
import soundfile as sf


def synthetic_rir(seed: int = 0, predelay_n: int = 0, n: int = 4000) -> np.ndarray:
    """A unit direct path after `predelay_n` zeros, then an exponentially decaying noise tail that stays below it."""
    rir = np.zeros(predelay_n + n, np.float32)
    rir[predelay_n] = 1.0
    tail = np.random.default_rng(seed).standard_normal(n - 1) * np.exp(-np.arange(n - 1) / 600)
    rir[predelay_n + 1 :] = 0.2 * tail / np.abs(tail).max()
    return rir


def make_rir_tree(root: Path, rooms: int = 20, per_room: int = 3) -> Path:
    """`rooms` rooms of `per_room` RIRs each, half under `but/`, half under `openslr28/`; returns `root`."""
    lines = ["path\troom"]
    for r in range(rooms):
        for k in range(per_room):
            name = f"but/PLACE{r:02d}/Mic01/Spk01/01/RIR/rir{k}.wav" if r % 2 == 0 else f"openslr28/room{r:02d}_{k}.wav"
            room = f"but-PLACE{r:02d}" if r % 2 == 0 else f"openslr28-room{r:02d}"
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            sf.write(root / name, synthetic_rir(100 * r + k, predelay_n=10 * k), 16000, subtype="FLOAT")
            lines.append(f"{name}\t{room}")
    (root / "index.tsv").write_text("\n".join(lines) + "\n")
    return root
