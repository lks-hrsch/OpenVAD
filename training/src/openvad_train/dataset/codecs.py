# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ffmpeg as a tool: stored codec round trips, their measured delays, and decoding to 16 kHz.

Everything goes through pipes, so no temporary file exists to leak or collide. `subprocess.run(input=...)`
writes stdin while it reads stdout and stderr, so a long input cannot fill a pipe and deadlock.
"""

import json
import re
import subprocess
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


class MissingCodec(RuntimeError):  # noqa: N818 (the name is the plan's interface)
    """ffmpeg, or an encoder the construction needs, is absent on this machine."""


class DecodeError(RuntimeError):
    """A file is damaged or unreadable; Task 8 catches this and drops the clip.

    A missing tool is `MissingCodec` and a bad argument is `ValueError`, so those abort a run instead.
    """


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


@lru_cache(maxsize=4096)
def _verify_intact(path: str) -> None:
    """Decode the whole file once per process and raise `DecodeError` if ffmpeg says anything.

    ffmpeg exits 0 on a damaged stream and reports only on stderr, while soundfile gives no signal at all.
    Without this, a 120 s Ogg with 20 kB overwritten returned the full length with audio shifted by 4 s.
    One extra full decode per file per process is accepted; a failure is not cached.
    """
    done = _ffmpeg(["-i", path, "-f", "null", "-"], None, DecodeError)
    if done.stderr.strip():
        raise DecodeError(f"{path}: {done.stderr.decode(errors='replace').strip()}")


def _refuse_8k(path: str) -> ValueError:
    return ValueError(f"{path} is 8 kHz, which needs the OpenVAD resampler (ADR 0009); it does not exist yet")


def decode_16k(path: str, offset_s: float, duration_s: float) -> np.ndarray:
    """Mono float32 audio at 16 kHz from `offset_s` for `duration_s`, shorter if the file ends first.

    The result has `round(duration_s * 16000)` samples when the file holds that much audio after the offset.
    If it ends first, what exists is returned, never padded (an offset past the end gives an empty array), so
    callers rasterising labels from the length see the real audio. Channels are averaged, and the array is
    writable. Raises `DecodeError` for a file that is unreadable or damaged (every file is first decoded once
    per process to find damage, see `_verify_intact`), `MissingCodec` if ffmpeg or ffprobe is absent, and
    `ValueError` for a negative argument or an 8 kHz file, which belongs to the OpenVAD resampler (ADR 0009).

    Seeking is done by soundfile (libvorbisfile, libFLAC and so on), never by ffmpeg's `-ss`: on real SWC
    Ogg Vorbis files ffmpeg's seek landed 328 samples (7 ms) off at 30 s and exact at 60.123 s, and on
    synthetic Vorbis 128 to 256 samples off, which would shift audio against its labels. Mid-file segments
    from soundfile agree with a full ffmpeg decode to 2e-7. Seeks into the last Ogg page are not exact
    (up to 832 source samples off), so a read starting within 5 s of the end seeks to 5 s before the end
    and slices. A file at a rate other than 16 kHz is read with 100 ms of context on each side, resampled by
    ffmpeg through a pipe (version recorded in the manifest, ADR 0009) and trimmed, so the resampler's edges
    do not touch the result. The start is rounded to the nearest source sample, which is at most 62 us off
    for rates above 8 kHz.

    A file soundfile cannot open (for example an Ogg with a Skeleton stream) is decoded by ffmpeg from the
    start without seeking and sliced, which is exact but costs time in proportion to `offset_s`.
    """
    if offset_s < 0 or duration_s < 0:
        raise ValueError(f"offset and duration must not be negative, got {offset_s} and {duration_s}")
    n = round(duration_s * 16000)
    _verify_intact(path)
    try:
        info = sf.info(path)
    except RuntimeError:
        return _decode_16k_without_seeking(path, offset_s, n)
    rate = info.samplerate
    if rate == 8000:
        raise _refuse_8k(path)
    start = round(offset_s * rate)
    if start >= info.frames:
        return np.empty(0, np.float32)
    context = 0 if rate == 16000 else rate // 10
    first = max(start - context, 0)
    end = start + round(n * rate / 16000) + 1 + context
    # Seeks into the last Ogg page land wrong, so read the tail sequentially from well before it.
    base = first if info.frames - first >= 5 * rate else max(0, info.frames - 5 * rate)
    try:
        x = sf.read(path, start=base, frames=end - base, dtype="float32", always_2d=True)[0][first - base :]
    except RuntimeError as e:
        raise DecodeError(f"{path}: {e}") from e
    mono = x.mean(axis=1, dtype=np.float32)
    if rate == 16000:
        return mono[:n]
    y = _resample_to_16k(mono, rate)
    skip = round((start - first) * 16000 / rate)
    return y[skip : skip + n]


def _resample_to_16k(mono: np.ndarray, rate: int) -> np.ndarray:
    raw = _ffmpeg(
        ["-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0", "-ar", "16000", "-f", "f32le", "pipe:1"],
        mono.tobytes(),
        DecodeError,
    ).stdout
    return np.frombuffer(raw, np.float32).copy()  # frombuffer is read-only; callers may scale in place


def _decode_16k_without_seeking(path: str, offset_s: float, n: int) -> np.ndarray:
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate",
            "-of",
            "default=nw=1",
            path,
        ],
        None,
        DecodeError,
    ).stdout.decode()
    fields = dict(line.split("=", 1) for line in probe.split())
    if "channels" not in fields:
        raise DecodeError(f"{path}: ffprobe found no audio stream")
    if fields["sample_rate"] == "8000":
        raise _refuse_8k(path)
    start = round(offset_s * 16000)
    # Ten milliseconds of slack because the resampler may emit a sample fewer than the ideal count.
    raw = _ffmpeg(
        [
            "-t",
            f"{(start + n) / 16000 + 0.01:.6f}",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        None,
        DecodeError,
    ).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, int(fields["channels"]))
    return x.mean(axis=1, dtype=np.float32)[start : start + n]
