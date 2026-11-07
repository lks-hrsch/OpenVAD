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

from openvad_rs import decimate_2x, interpolate_2x


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
# The modes of the AMR codecs in bit/s. ffmpeg silently encodes any other bitrate at some other mode, and None at
# the highest, so `roundtrip` refuses them; plan.py draws from this table too.
AMR_MODES_BPS = {
    "amr_wb": (6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850),
    "amr_nb": (4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200),
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


def encoder_of(codec: str) -> str:
    """The ffmpeg encoder `preflight` must find for a codec of a plan row."""
    return _CODECS[codec][0]


def roundtrip(audio: np.ndarray, codec: str, bitrate: int | None, rate: int = 16000) -> np.ndarray:
    """Encode and decode `audio` (float32, mono) sampled at `rate`, which must be the codec's own rate.

    `bitrate` is in bit/s. For amr_nb and amr_wb it must be one of `AMR_MODES_BPS`, `None` included in the refusal,
    else `ValueError`: ffmpeg would encode another mode without a word.

    That is 16 kHz for every codec but amr_nb, which needs `rate=8000`: the audio carries no rate, so a caller
    declares it, and a mismatch raises `ValueError` instead of encoding audio at the wrong speed. `narrow_roundtrip`
    is the 16 kHz entry for amr_nb.

    The result has the input's length and dtype. The output lags the input by `delay_ms(codec)`; nothing here
    removes it, because the labels are shifted by the table instead. A decode more than one codec frame short
    of the input raises `RuntimeError` instead of being zero-padded into a plausible result.
    """
    if codec not in _CODECS:
        raise ValueError(f"unknown codec {codec!r}, expected one of {sorted(_CODECS)}")
    encoder, container, codec_rate = _CODECS[codec]
    if rate != codec_rate:
        raise ValueError(f"{codec} runs at {codec_rate} Hz, got audio declared at {rate} Hz")
    if codec in AMR_MODES_BPS and bitrate not in AMR_MODES_BPS[codec]:
        raise ValueError(f"{codec} has no mode at {bitrate} bit/s, expected one of {AMR_MODES_BPS[codec]}")
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
    y = roundtrip(x, codec, bitrate, rate)
    corr = signal.correlate(y, x, mode="full", method="fft")
    return float((np.argmax(corr) - (n - 1)) / rate * 1000)


def narrow_roundtrip(audio: np.ndarray, codec: str, bitrate: int | None) -> np.ndarray:
    """The AMR-NB leg: 16 kHz float32 in, 16 kHz float32 of the same length out. `bitrate` is in bit/s, an AMR-NB mode.

    Decimates with the OpenVAD resampler, clips to the int16 range the codec takes, round-trips at 8 kHz,
    interpolates, and trims the one sample that an odd length gains. A decode more than one codec frame short
    raises `RuntimeError`, as in `roundtrip`.

    The output keeps its delay; nothing is compensated. It lags the input by `delay_ms(codec)` (the codec alone,
    measured at 8 kHz) plus the resampler round trip of `2 * GROUP_DELAY_16K` samples at 16 kHz (66 samples,
    4.125 ms). The label shift of a plan row therefore needs that sum before it rounds to the 10 ms grid;
    `codec_delay_ms` in `construct.py` reads the table alone and must add the resampler share when the leg is
    restored.
    """
    if codec != "amr_nb":
        raise ValueError(f"narrow_roundtrip only runs amr_nb, got {codec!r}")
    narrow = decimate_2x(audio)
    # The codec takes int16; clip here, not in ffmpeg's float conversion. The filter overshoots by up to 14%.
    np.clip(narrow, -1.0, 32767 / 32768, out=narrow)
    return interpolate_2x(roundtrip(narrow, codec, bitrate, 8000))[: len(audio)]


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


# Above this, seeking plus one cached whole-file check beats one decode per read. Rank-ordered shards interleave
# thousands of short files between chunks of one article, so a cache that also held short files would evict it.
_LONG_FILE_S = 60.0


def _raise_if_ffmpeg_reported(path: str, done: subprocess.CompletedProcess[bytes]) -> None:
    # ffmpeg exits 0 on a damaged stream and reports only on stderr. Without this, a 120 s Ogg with 20 kB overwritten
    # returned the full length with audio shifted by 4 s.
    if done.stderr.strip():
        raise DecodeError(f"{path}: {done.stderr.decode(errors='replace').strip()}")


@lru_cache(maxsize=4096)
def _verify_intact(path: str) -> None:
    """Decode a long file once per process and raise `DecodeError` if ffmpeg says anything; soundfile gives no signal.

    One extra full decode per long file per process is accepted; a failure is not cached.
    """
    _raise_if_ffmpeg_reported(path, _ffmpeg(["-i", path, "-f", "null", "-"], None, DecodeError))


def _refuse_8k(path: str) -> ValueError:
    return ValueError(f"{path} is 8 kHz, which needs the OpenVAD resampler (ADR 0009); it does not exist yet")


def decode_16k(path: str, offset_s: float, duration_s: float) -> np.ndarray:
    """Mono float32 audio at 16 kHz from `offset_s` for `duration_s`, shorter if the file ends first.

    The result has `round(duration_s * 16000)` samples when the file holds that much audio after the offset.
    If it ends first, what exists is returned, never padded (an offset past the end gives an empty array), so
    callers rasterising labels from the length see the real audio. Channels are averaged, and the array is
    writable. Raises `DecodeError` for a file that is unreadable or damaged, `MissingCodec` if ffmpeg or ffprobe is
    absent, and `ValueError` for a negative argument or an 8 kHz file, which belongs to the OpenVAD resampler
    (ADR 0009).

    A file of at most `_LONG_FILE_S` is decoded whole by one ffmpeg call with its stderr checked, then sliced, so
    the integrity check is built in and nothing is cached. The length comes from soundfile, or from ffprobe for a
    file soundfile cannot open.

    A longer file is first decoded once per process by `_verify_intact`, then read by soundfile (libvorbisfile,
    libFLAC and so on), never with ffmpeg's `-ss`: on real SWC Ogg Vorbis files ffmpeg's seek landed 328 samples
    (7 ms) off at 30 s and exact at 60.123 s, and on synthetic Vorbis 128 to 256 samples off, which would shift
    audio against its labels. Mid-file segments from soundfile agree with a full ffmpeg decode to 2e-7. Seeks into
    the last Ogg page are not exact (up to 832 source samples off), so a read starting within 5 s of the end seeks
    to 5 s before the end and slices. A file at a rate other than 16 kHz is read with 100 ms of context on each
    side, resampled by ffmpeg through a pipe (version recorded in the manifest, ADR 0009) and trimmed, so the
    resampler's edges do not touch the result. The start is rounded to the nearest source sample and the trim to
    the nearest 16 kHz sample, which is at most half a source sample plus 31 us off, so 77 us at 11025 Hz. A long
    file soundfile cannot open (for example an Ogg with a Skeleton stream) is decoded by ffmpeg from the start
    without seeking and sliced, which is exact but costs time in proportion to `offset_s`.
    """
    if offset_s < 0 or duration_s < 0:
        raise ValueError(f"offset and duration must not be negative, got {offset_s} and {duration_s}")
    n = round(duration_s * 16000)
    try:
        info = sf.info(path)
    except RuntimeError:
        info = None
        rate, channels, seconds = _probe(path)
    else:
        rate, channels, seconds = info.samplerate, info.channels, info.duration
    if rate == 8000:
        raise _refuse_8k(path)
    if seconds <= _LONG_FILE_S:
        return _decode_whole_file(path, offset_s, n, channels, seconds)
    _verify_intact(path)
    if info is None:
        return _decode_16k_without_seeking(path, offset_s, n, channels)
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


def _probe(path: str) -> tuple[int, int, float]:
    """Rate, channels and seconds of the first audio stream; an unknown length is infinite, so it counts as long."""
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate:format=duration",
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
    duration = fields.get("duration", "N/A")
    return int(fields["sample_rate"]), int(fields["channels"]), float("inf") if duration == "N/A" else float(duration)


def _decode_whole_file(path: str, offset_s: float, n: int, channels: int, seconds: float) -> np.ndarray:
    # -ac keeps the array's shape whatever layout ffmpeg settles on; the mean below is the actual downmix.
    done = _ffmpeg(
        ["-i", path, "-map", "0:a:0", "-ac", str(channels), "-ar", "16000", "-f", "f32le", "pipe:1"], None, DecodeError
    )
    _raise_if_ffmpeg_reported(path, done)
    # The resampler may emit a sample past the source's end, which must not turn a past-end read into a sample.
    if offset_s >= seconds:
        return np.empty(0, np.float32)
    start = round(offset_s * 16000)
    x = np.frombuffer(done.stdout, np.float32).reshape(-1, channels)
    return x.mean(axis=1, dtype=np.float32)[start : start + n]


def _decode_16k_without_seeking(path: str, offset_s: float, n: int, channels: int) -> np.ndarray:
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
    x = np.frombuffer(raw, np.float32).reshape(-1, channels)
    return x.mean(axis=1, dtype=np.float32)[start : start + n]
