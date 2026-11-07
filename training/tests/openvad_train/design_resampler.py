# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re

import numpy as np
from scipy.signal import freqz

from openvad_train.design_resampler import OUTPUT, coefficients


def test_stopband_at_least_70_db():
    w, h = freqz(coefficients(), worN=8192, fs=16000)
    assert 20 * np.log10(np.abs(h[w >= 4500]).max()) <= -70


def test_passband_flat_to_3400_hz():
    w, h = freqz(coefficients(), worN=8192, fs=16000)
    assert np.ptp(20 * np.log10(np.abs(h[w <= 3400]))) <= 0.1


def test_half_band_structure():
    c = coefficients()
    centre = len(c) // 2
    assert len(c) % 4 == 3
    assert np.array_equal(c, c[::-1])
    assert c[centre] == np.float32(0.5)
    distance = np.abs(np.arange(len(c)) - centre)
    assert not c[(distance % 2 == 0) & (distance > 0)].any()
    # Half-band symmetry puts DC at a ripple extremum; measured +0.0019 dB.
    assert abs(20 * np.log10(c.sum(dtype=np.float64))) <= 0.0025


def test_committed_rust_coefficients_match():
    text = OUTPUT.read_text()
    committed = np.array([float(v) for v in re.findall(r"^    (-?[\d.e-]+),$", text, re.M)])
    assert f"[f32; {len(committed)}]" in text
    assert np.allclose(committed, coefficients(), rtol=0, atol=1e-6)
