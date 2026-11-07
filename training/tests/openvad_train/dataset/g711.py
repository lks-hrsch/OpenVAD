# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import shutil
import subprocess

import numpy as np
import pytest

from openvad_train.dataset.g711 import alaw_roundtrip, mulaw_roundtrip


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

LAWS = [(alaw_roundtrip, "pcm_alaw", "alaw"), (mulaw_roundtrip, "pcm_mulaw", "mulaw")]
EVERY_INT16 = np.arange(-32768, 32768, dtype="<i2")


def run_ffmpeg(args: list[str], data: bytes) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", *args], input=data, capture_output=True, check=True).stdout


def ffmpeg_encode_decode(encoder: str, container: str) -> np.ndarray:
    """Every int16 value through ffmpeg's encoder and decoder, as int16."""
    encoded = run_ffmpeg(
        ["-f", "s16le", "-ar", "8000", "-ac", "1", "-i", "pipe:0", "-c:a", encoder, "-f", container, "pipe:1"],
        EVERY_INT16.tobytes(),
    )
    return ffmpeg_decode(encoded, container)


def ffmpeg_decode(codes: bytes, container: str) -> np.ndarray:
    decoded = run_ffmpeg(["-f", container, "-ar", "8000", "-ac", "1", "-i", "pipe:0", "-f", "s16le", "pipe:1"], codes)
    return np.frombuffer(decoded, "<i2")


def as_int16(y: np.ndarray) -> np.ndarray:
    return (y.astype(np.float64) * 32768).astype(np.int32)


def at(roundtrip, *values: int) -> list[int]:
    return as_int16(roundtrip(np.array(values, np.float32) / 32768)).tolist()


@needs_ffmpeg
@pytest.mark.parametrize(("roundtrip", "_encoder", "container"), LAWS)
def test_decoded_levels_equal_ffmpegs_decoder_for_every_code(roundtrip, _encoder, container):
    # The decoders agree with the standard; the 255 mu-law levels are 256 codes with the two zeros equal.
    ffmpeg_levels = np.unique(ffmpeg_decode(bytes(range(256)), container))
    assert np.array_equal(np.unique(as_int16(roundtrip(EVERY_INT16.astype(np.float32) / 32768))), ffmpeg_levels)


@needs_ffmpeg
@pytest.mark.parametrize(("roundtrip", "encoder", "container"), LAWS)
def test_ffmpegs_encoder_differs_from_g711_at_964_values(roundtrip, encoder, container):
    # ffmpeg shifts its negative bins 4 LSB toward zero and puts segment edges at level midpoints, so it is not
    # G.711. 712 + 252 values on A-law, 708 + 256 on mu-law.
    out = as_int16(roundtrip(EVERY_INT16.astype(np.float32) / 32768))
    assert np.count_nonzero(out != ffmpeg_encode_decode(encoder, container)) == 964


@pytest.mark.parametrize("roundtrip", [alaw_roundtrip, mulaw_roundtrip])
def test_negative_values_mirror_positive_ones(roundtrip):
    # G.711 codes a negative value as its one's complement, so f(-v - 1) == -f(v) at every int16 value. For mu-law
    # that includes zero: v = -1 .. -4 and v = 0 .. 3 both decode to 0, and -0 == 0.
    out = as_int16(roundtrip(EVERY_INT16.astype(np.float32) / 32768))
    assert np.array_equal(out[:32768][::-1], -out[32768:])


def test_alaw_hand_vectors_from_the_standard():
    # 13-bit value v >> 3, step 2 in the first two segments, one's complement below zero.
    assert at(alaw_roundtrip, 0, 7, 8, 15) == [8, 8, 8, 8]
    assert at(alaw_roundtrip, 16, 31, 32) == [24, 24, 40]
    assert at(alaw_roundtrip, -1, -8, -9, -16) == [-8, -8, -8, -8]
    assert at(alaw_roundtrip, -17, -24) == [-24, -24]
    # Segment 6 ends at 16383 and segment 7 starts at 16384 with a 1024-wide step.
    assert at(alaw_roundtrip, 16383, 16384, 16511, 17407, 17408) == [16128, 16896, 16896, 16896, 17920]
    assert at(alaw_roundtrip, 32767, -32768) == [32256, -32256]


def test_mulaw_hand_vectors_from_the_standard():
    # 14-bit value v >> 2 with bias 33; -1 .. -4 is the negative zero, which decodes to 0.
    assert at(mulaw_roundtrip, 0, 3, -1, -4) == [0, 0, 0, 0]
    assert at(mulaw_roundtrip, 4, 7, -5, -8) == [8, 8, -8, -8]
    # Segment 0 ends at 123 with level 120; segment 1 starts at 124 with level 132; the mirror is -125 .. -128.
    assert at(mulaw_roundtrip, 123, 124, -124, -125, -128) == [120, 132, -120, -132, -132]
    # Segment 7 steps up at 31612; the magnitude saturates at 8158, so the full-scale ends share the top level.
    assert at(mulaw_roundtrip, 31611, 31612, 32767, -32768) == [31100, 32124, 32124, -32124]


def test_a_value_between_two_int16_steps_rounds_half_to_even():
    # Level boundaries sit at 15|16 (up) and -13|-12 (down). 15.5 rounds to 16, so it is not floored, and -12.5 to
    # -12, so it is not rounded away from zero.
    x = np.array([15.5, -12.5], np.float32) / 32768
    assert as_int16(alaw_roundtrip(x)).tolist() == [24, -8]


@pytest.mark.parametrize("roundtrip", [alaw_roundtrip, mulaw_roundtrip])
def test_beyond_full_scale_is_clipped_not_wrapped(roundtrip):
    x = np.array([1.0, 1.75, 100.0, -1.0, -1.75, -100.0, np.inf, -np.inf], np.float32)
    out = roundtrip(x)
    assert out[0] > 0.9
    assert out[3] < -0.9
    assert np.array_equal(out[:3], np.full(3, out[0]))
    assert np.array_equal(out[3:6], np.full(3, out[3]))
    assert out[6] == out[0]
    assert out[7] == out[3]


@pytest.mark.parametrize("roundtrip", [alaw_roundtrip, mulaw_roundtrip])
def test_output_is_a_new_float32_array_and_the_input_is_unchanged(roundtrip):
    x = np.linspace(-1.5, 1.5, 101, dtype=np.float32)
    before = x.copy()
    out = roundtrip(x)
    assert out.dtype == np.float32
    assert out is not x
    assert out.shape == x.shape
    assert np.array_equal(x, before)


@pytest.mark.parametrize("roundtrip", [alaw_roundtrip, mulaw_roundtrip])
def test_empty_input_gives_empty_output(roundtrip):
    assert roundtrip(np.zeros(0, np.float32)).shape == (0,)
