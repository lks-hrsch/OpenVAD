# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A synthetic DEMAND tree, for the noise and plan tests."""

from pathlib import Path

import numpy as np
import soundfile as sf


def make_demand_tree(root: Path, scenes=("DKITCHEN", "NPARK", "OOFFICE", "PCAFETER", "SCAFE", "TBUS")) -> Path:
    for scene in scenes:
        (root / scene).mkdir(parents=True)
        for channel in range(1, 17):
            sf.write(root / scene / f"ch{channel:02d}.wav", np.zeros(1600, np.float32), 16000, subtype="PCM_16")
    (root / "manifest.json").write_text("{}")
    return root
