# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`load_audio`, the one reader of the audio a dataset row points at."""

import asyncio
import io
from pathlib import Path

import numpy as np
import soundfile as sf

from openvad_train.dataset.codecs import decode_16k
from openvad_train.dataset.parquet_io import read_parquet_row
from openvad_train.fetch.common import data_dir


def _read_stored(data: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if rate != 16000:
        raise ValueError(f"stored audio must be 16 kHz, got {rate}")
    return audio.mean(axis=1, dtype=np.float32)


def load_audio(row: dict, data_root: Path | None = None) -> np.ndarray:
    """Mono float32 audio at 16 kHz of one row, whose `audio_filepath` is relative to `data_root` (by default
    `$OPENVAD_DATA_DIR`). `audio_format` selects how the other pointer columns are read.
    """
    fmt = row["audio_format"]
    if fmt not in ("flac_tar", "file", "callhome_parquet"):
        raise ValueError(f"unknown audio_format {fmt!r}")
    path = (data_dir() if data_root is None else data_root) / row["audio_filepath"]
    if fmt == "flac_tar":
        with path.open("rb") as f:
            f.seek(row["byte_offset"])
            data = f.read(row["byte_length"])
        # A short read would decode as a shorter sample and pass for a real one.
        if len(data) != row["byte_length"]:
            raise ValueError(f"{path}: read {len(data)} of {row['byte_length']} bytes at {row['byte_offset']}")
        return _read_stored(data)
    if fmt == "file":
        return decode_16k(str(path), row["offset"], row["duration"])
    audio = asyncio.run(read_parquet_row(path, row["row_in_file"], ["audio"]))["audio"]
    start = round(row["offset"] * 16000)
    return _read_stored(audio["bytes"])[start : start + round(row["duration"] * 16000)]
