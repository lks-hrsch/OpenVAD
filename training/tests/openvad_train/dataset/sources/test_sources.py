# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests of `sources/__init__.py`, which cannot have a `tests/.../__init__.py` (it would shadow the package)."""

import struct

import numpy as np
import soundfile as sf

from as_is_support import write_noise
from openvad_train.dataset.sources import wav_durations


def test_a_wav_length_is_the_header_s_and_equals_what_soundfile_reports(tmp_path):
    write_noise(tmp_path / "a.wav", 1.5)
    write_noise(tmp_path / "b.wav", 12.34)
    assert wav_durations([tmp_path / "a.wav", tmp_path / "b.wav"]) == [
        sf.info(tmp_path / "a.wav").duration,
        sf.info(tmp_path / "b.wav").duration,
    ]


def test_a_file_that_is_not_plain_pcm_wav_or_has_a_long_header_goes_to_soundfile(tmp_path):
    sf.write(tmp_path / "c.flac", np.zeros(24000, np.float32), 16000)
    write_noise(tmp_path / "d.wav", 1.0)
    # A 5 KB LIST chunk in front of the data, so the first 4 KB do not reach it.
    raw = (tmp_path / "d.wav").read_bytes()
    chunk = b"LIST" + struct.pack("<I", 5000) + bytes(5000)
    body = raw[12:36] + chunk + raw[36:]
    (tmp_path / "e.wav").write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)
    assert wav_durations([tmp_path / "c.flac", tmp_path / "e.wav"]) == [1.5, 1.0]
