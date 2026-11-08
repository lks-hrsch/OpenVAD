# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""One adapter per source, mirroring `openvad_train.fetch`."""

import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf


# Header reads over the share take about 30 ms each; this many at once hide the latency.
_THREADS = 16


def count_lines(path: Path) -> int:
    """Lines of a text or `.gz` file, for the manifest's count of what an adapter skipped."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as f:
        return sum(1 for _ in f)


def wav_durations(files: list[Path]) -> list[float]:
    """Seconds of each audio file from its header, read in parallel."""
    with ThreadPoolExecutor(_THREADS) as pool:
        return list(pool.map(lambda f: sf.info(f).duration, files))
