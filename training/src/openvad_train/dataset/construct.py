# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `construct`: each plan row becomes stored audio and labels, a pure function of the row."""

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from openvad_train.dataset.codecs import DecodeError, decode_16k, delay_ms, roundtrip
from openvad_train.dataset.config import DEMAND_CATEGORIES
from openvad_train.dataset.cutting import rebase
from openvad_train.dataset.labels import SAMPLES_PER_FRAME, SAMPLES_PER_MS, apply_gap_rule, rasterise
from openvad_train.dataset.mix import SAMPLE_RATE, dither, loop_slice, mix_at_snr, place, protect_peak


_CONDITIONS_BUILT = {"clean", *DEMAND_CATEGORIES}
_CODECS_BUILT = {"none", "g722"}


def _legs_not_built(row: dict) -> list[str]:
    """The fields of `row` that ask for a leg construction does not have yet: Task 18 and Task 19 add them."""
    asked = {
        "room": row["room"],
        "handset": row["handset"] is not None,
        "agc": row["agc"] is not None,
        "clip_drive_db": row["clip_drive_db"] is not None,
        "suppression": row["suppression"] is not None,
        "transients": bool(row["transients"]),
        "segments": bool(row["segments"]),
        "band": row["band"] != "wide",
        "codec": row["codec"] not in _CODECS_BUILT,
        "condition": row["condition"] not in _CONDITIONS_BUILT,
    }
    return [field for field, wanted in asked.items() if wanted]


def _read_noise(path: Path, offset_n: int, n: int) -> np.ndarray:
    """What `loop_slice` gives on the whole file, reading only the part it needs unless the slice wraps around."""
    with sf.SoundFile(path) as f:
        if f.samplerate != SAMPLE_RATE:
            raise ValueError(f"{path}: noise must be 16 kHz, got {f.samplerate}")
        if offset_n + n <= f.frames:
            f.seek(offset_n)
            return f.read(n, dtype="float32")
        return loop_slice(f.read(dtype="float32"), offset_n, n)


def _noise_of(row: dict, raw: Path) -> np.ndarray:
    files, offsets = row["noise_source_ids"], row["noise_offsets"]
    if len(files) != 1 or len(offsets) != 1:
        raise ValueError(f"{row['sample_id']}: a {row['condition']} row needs one noise, got {files} at {offsets}")
    return _read_noise(raw / files[0], offsets[0], row["total_n"])


def codec_delay_ms(row: dict) -> int:
    """The whole milliseconds by which the row's stored audio lags its input, which its labels are shifted by."""
    return 0 if row["codec"] == "none" else round(delay_ms(row["codec"]))


def construct_sample(row: dict, raw: Path) -> tuple[bytes, np.ndarray]:
    """16-bit 16 kHz mono FLAC and one label per 10 ms frame for a plan row.

    `raw` is the root the row's paths are relative to, `$OPENVAD_DATA_DIR` in a build. The only draw made here is
    the dither of a clean row: its `total_n` normals from `default_rng(row["seed"])`. Bytes then depend only on the
    row, its source files and the ffmpeg version.

    Labels come from the audio produced, never from file metadata. `DecodeError` means this clip cannot be built: a
    damaged file, a decode further than one 10 ms frame from `source_n`, or silence under the words. A decode can
    run long only by `decode_16k` breaking its contract, because it stops at `source_n`. `NotImplementedError` names
    the fields of a row asking for a leg that does not exist yet, and nothing is skipped silently.
    """
    missing = _legs_not_built(row)
    if missing:
        raise NotImplementedError(f"{row['sample_id']} asks for legs not built yet: {', '.join(missing)}")
    offset_n, total_n, source_n = row["speech_offset_n"], row["total_n"], row["source_n"]
    if offset_n % SAMPLES_PER_MS:
        raise ValueError(f"{row['sample_id']}: padding offset {offset_n} is off the millisecond grid labels use")

    path = str(raw / row["source_path"])
    speech = decode_16k(path, row["source_offset_n"] / SAMPLE_RATE, source_n / SAMPLE_RATE)
    if abs(len(speech) - source_n) > SAMPLES_PER_FRAME:
        raise DecodeError(f"{path}: decoded {len(speech)} samples for source_n {source_n}, so its labels would drift")
    # Words the audio ends before are no speech: an MP3's overstated length leaves up to a frame of them.
    end_ms = -(-len(speech) // SAMPLES_PER_MS)
    speech_ms, mask_ms = rebase(row["speech_ms"], 0, end_ms), rebase(row["mask_ms"], 0, end_ms)

    x = place(speech, total_n, offset_n)
    # Audio decisions use the word union at G = 0, so a change of G leaves the audio as it is.
    active = np.zeros(total_n, bool)
    for start, end in apply_gap_rule(speech_ms, 0):
        active[offset_n + start * SAMPLES_PER_MS : offset_n + end * SAMPLES_PER_MS] = True
    if not x[active].any():
        raise DecodeError(f"{path}: silent under the words of {row['sample_id']}")
    if row["condition"] == "clean":
        x = x + dither(total_n, row["dither_dbfs"], np.random.default_rng(row["seed"]))
    else:
        noise = _noise_of(row, raw)
        if not noise[active].any():
            raise DecodeError(f"{row['noise_source_ids'][0]}: silent under the words of {row['sample_id']}")
        x = mix_at_snr(x, noise, active, row["snr_db"])
    x = protect_peak(x)
    if row["codec"] != "none":
        x = roundtrip(x, row["codec"], round(row["bitrate"] * 1000))  # the plan's bitrates are in kbit/s

    shift_ms = offset_n // SAMPLES_PER_MS + codec_delay_ms(row)
    labels = rasterise(apply_gap_rule(speech_ms, row["gap_ms"]), mask_ms, len(x), shift_ms)
    flac = io.BytesIO()
    sf.write(flac, x, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return flac.getvalue(), labels
