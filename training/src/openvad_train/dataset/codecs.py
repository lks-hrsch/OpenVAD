# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ffmpeg as a tool: stored codec round trips and their measured delays.

Everything goes through pipes, so no temporary file exists to leak or collide. `subprocess.run(input=...)`
writes stdin while it reads stdout and stderr, so a long input cannot fill a pipe and deadlock.
"""

import json
import re
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
from scipy import signal


class MissingCodec(RuntimeError):  # noqa: N818 (the name is the plan's interface)
    """ffmpeg, or an encoder the construction needs, is absent on this machine."""


# codec -> (ffmpeg encoder, container, rate in Hz). AMR-NB runs at 8 kHz: its caller decimates, runs the
# round trip and interpolates (spec, stored codec leg), so the audio is already at the codec's rate.
_CODECS = {
    "g722": ("g722", "g722", 16000),
    "amr_wb": ("libvo_amrwbenc", "amr", 16000),
    "amr_nb": ("libopencore_amrnb", "amr", 8000),
    "opus": ("libopus", "ogg", 16000),
}
# Zeros appended before encoding so the encoder flushes whole frames; AMR and Opus frames are 20 ms.
_FRAME_MS = 20


def _run(cmd: list[str], data: bytes | None, error: type[RuntimeError]) -> subprocess.CompletedProcess[bytes]:
    """Run `cmd`; a missing binary is `MissingCodec`, a non-zero exit is `error` carrying stderr."""
    try:
        # No input means an empty stdin, so a command that reads it sees end of file at once.
        done = subprocess.run(cmd, capture_output=True, input=data or b"")
    except FileNotFoundError as e:
        raise MissingCodec(f"{cmd[0]} is not installed") from e
    if done.returncode != 0:
        raise error(f"{cmd[0]} failed with {done.returncode}: {done.stderr.decode(errors='replace').strip()}")
    return done


def _ffmpeg(args: list[str], data: bytes | None, error: type[RuntimeError]) -> subprocess.CompletedProcess[bytes]:
    return _run(["ffmpeg", "-v", "error", "-nostdin", *args], data, error)


def preflight(required: set[str]) -> dict:
    """ffmpeg's version and build flags; `MissingCodec` names an absent ffprobe or any absent encoder in `required`."""
    version_out = _run(["ffmpeg", "-version"], None, MissingCodec).stdout.decode()
    encoders_out = _run(["ffmpeg", "-hide_banner", "-encoders"], None, MissingCodec).stdout.decode()
    _run(["ffprobe", "-version"], None, MissingCodec)
    version = re.search(r"ffmpeg version (\S+)", version_out)
    flags = re.search(r"^configuration:(.*)$", version_out, re.MULTILINE)
    present = {m.group(1) for m in re.finditer(r"^\s*[AVS][.\w]{5}\s+(\S+)", encoders_out, re.MULTILINE)}
    absent = sorted(required - present)
    if absent:
        raise MissingCodec(f"this ffmpeg has no encoder {', '.join(absent)}")
    return {
        "version": version.group(1) if version else "unknown",
        "configuration": flags.group(1).split() if flags else [],
    }


def roundtrip(audio: np.ndarray, codec: str, bitrate: int | None) -> np.ndarray:
    """Encode and decode `audio` (float32, mono, at the codec's rate: 16 kHz, or 8 kHz for amr_nb).

    The result has the input's length and dtype. The output lags the input by `delay_ms(codec)`; nothing here
    removes it, because the labels are shifted by the table instead. A decode more than one codec frame short
    of the input raises `RuntimeError` instead of being zero-padded into a plausible result.
    """
    if codec not in _CODECS:
        raise ValueError(f"unknown codec {codec!r}, expected one of {sorted(_CODECS)}")
    encoder, container, rate = _CODECS[codec]
    frame = rate * _FRAME_MS // 1000
    padded = np.concatenate([audio.astype(np.float32), np.zeros(frame, np.float32)])
    quality = [] if bitrate is None else ["-b:a", str(bitrate)]
    encoded = _ffmpeg(
        [
            "-f",
            "f32le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            encoder,
            *quality,
            "-f",
            container,
            "pipe:1",
        ],
        padded.tobytes(),
        RuntimeError,
    ).stdout
    decoded = _ffmpeg(
        ["-f", container, "-i", "pipe:0", "-f", "f32le", "-ar", str(rate), "-ac", "1", "pipe:1"], encoded, RuntimeError
    ).stdout
    out = np.frombuffer(decoded, np.float32)[: len(audio)]
    if len(audio) - len(out) > frame:
        raise RuntimeError(f"{codec} decode returned {len(out)} of {len(audio)} samples")
    return np.pad(out, (0, len(audio) - len(out)))


def measure_delay(codec: str, bitrate: int | None) -> float:
    """Delay of a round trip in ms by cross-correlation on a chirp; positive means the output lags the input."""
    rate = _CODECS[codec][2]
    n = 2 * rate
    t = np.arange(n) / rate
    x = (0.5 * signal.chirp(t, 100, 2.0, 0.45 * rate)).astype(np.float32)
    # Silence in front keeps the correlation peak away from the edge, and a chirp has one sharp peak.
    x[: rate // 8] = 0
    y = roundtrip(x, codec, bitrate)
    corr = signal.correlate(y, x, mode="full", method="fft")
    return float((np.argmax(corr) - (n - 1)) / rate * 1000)


_DELAYS = Path(__file__).with_name("codec_delays.json")


@cache
def _delay_table() -> dict:
    return json.loads(_DELAYS.read_text())


def delay_ms(codec: str) -> float:
    """The measured delay of a codec in ms (positive: output lags input), from `codec_delays.json`."""
    delays = _delay_table()["delays_ms"]
    if codec not in delays:
        raise KeyError(f"no delay measured for {codec!r}; measured: {sorted(delays)}")
    return delays[codec]
