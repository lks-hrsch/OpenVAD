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


def ffmpeg_reference(encoder: str, container: str) -> np.ndarray:
    """Every int16 value through ffmpeg's encoder and decoder, as int16."""

    def run(args: list[str], data: bytes) -> bytes:
        return subprocess.run(["ffmpeg", "-v", "error", *args], input=data, capture_output=True, check=True).stdout

    encoded = run(
        ["-f", "s16le", "-ar", "8000", "-ac", "1", "-i", "pipe:0", "-c:a", encoder, "-f", container, "pipe:1"],
        EVERY_INT16.tobytes(),
    )
    decoded = run(["-f", container, "-ar", "8000", "-ac", "1", "-i", "pipe:0", "-f", "s16le", "pipe:1"], encoded)
    return np.frombuffer(decoded, "<i2")


def as_int16(y: np.ndarray) -> np.ndarray:
    return (y.astype(np.float64) * 32768).astype(np.int32)


@needs_ffmpeg
@pytest.mark.parametrize(("roundtrip", "encoder", "container"), LAWS)
def test_matches_ffmpeg_for_every_int16(roundtrip, encoder, container):
    # k / 32768 is exact in float32, so this compares the integer domain: one wrong segment fails at its values.
    out = roundtrip(EVERY_INT16.astype(np.float32) / 32768)
    assert np.array_equal(as_int16(out), ffmpeg_reference(encoder, container).astype(np.int32))


def test_alaw_matches_standard_table():
    # ITU-T G.711 extremes and the smallest steps: 0xD5 and 0x55 are the A-law codes next to zero.
    out = as_int16(alaw_roundtrip(np.array([0, 3, 4, 15, 16, -1, 32767, -32768], np.float32) / 32768))
    assert out.tolist() == [8, 8, 8, 8, 24, -8, 32256, -32256]


def test_mulaw_matches_standard_table():
    out = as_int16(mulaw_roundtrip(np.array([0, 3, 4, 8, -1, -4, 32767, -32768], np.float32) / 32768))
    assert out.tolist() == [0, 0, 8, 8, -8, -8, 32124, -32124]


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
