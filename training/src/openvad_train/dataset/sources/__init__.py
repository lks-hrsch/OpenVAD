# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""One adapter per source, mirroring `openvad_train.fetch`."""

import gzip
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf


# Header reads over the share take about 30 ms each; this many at once hide the latency.
_THREADS = 16


def count_lines(path: Path) -> int:
    """Lines of a text or `.gz` file, for the manifest's count of what an adapter skipped."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as f:
        return sum(1 for _ in f)


def _seconds(path: Path) -> float:
    """Length from the header. A plain PCM WAV is read from its first 4 KB in one request, since libsndfile's many
    small reads cost a round trip each over the share (7 times slower on LibriVAD's 252,396 wavs); anything else
    is left to soundfile.
    """
    try:
        with path.open("rb") as f:
            head = f.read(4096)
        with wave.open(io.BytesIO(head)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError):
        return sf.info(path).duration


def wav_durations(files: list[Path]) -> list[float]:
    """Seconds of each audio file from its header, read in parallel."""
    with ThreadPoolExecutor(_THREADS) as pool:
        return list(pool.map(_seconds, files))
