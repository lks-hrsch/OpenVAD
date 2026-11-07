# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import pytest
import safetensors.numpy
from scipy import signal

import openvad_rs
from openvad_rs import GROUP_DELAY_16K, decimate_2x, half_band_taps, interpolate_2x
from openvad_train.design_resampler import coefficients


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"


def test_bindings_match_fixture():
    t = safetensors.numpy.load_file(FIXTURE)
    out = openvad_rs.linear(t["input"].ravel().tolist(), t["weight"].ravel().tolist(), t["bias"].tolist(), 4)
    np.testing.assert_allclose(out, t["expected"].ravel(), atol=1e-6, rtol=0)


def reference_decimate(x):
    """Float64 filtering of the design taps with scipy, independent of the Rust code."""
    return signal.lfilter(coefficients().astype(np.float64), 1.0, x.astype(np.float64))[::2]


def reference_interpolate(x):
    return 2.0 * signal.upfirdn(coefficients().astype(np.float64), x.astype(np.float64), up=2)[: 2 * len(x)]


def test_taps_are_the_design_taps():
    taps = half_band_taps()
    assert taps.dtype == np.float32
    np.testing.assert_array_equal(taps, coefficients())
    assert GROUP_DELAY_16K == (len(taps) - 1) // 2 == 33


def test_rust_matches_scipy_reference():
    x = np.random.default_rng(0).standard_normal(16000, dtype=np.float32)
    np.testing.assert_allclose(decimate_2x(x), reference_decimate(x), atol=1e-6)


def test_interpolate_matches_scipy_reference():
    # Audio range: standard normal input reaches outputs of 3.7, within 2% of the 1e-6 limit.
    x = np.random.default_rng(1).uniform(-1, 1, 8001).astype(np.float32)
    y = interpolate_2x(x)
    assert len(y) == 2 * len(x)
    np.testing.assert_allclose(y, reference_interpolate(x), atol=1e-6)


def test_odd_length_decimates_to_ceil_half_and_round_trips_to_n_plus_one():
    x = np.ones(1001, np.float32)
    assert len(decimate_2x(x)) == 501
    assert len(interpolate_2x(decimate_2x(x))) == 1002


def test_empty_input_gives_empty_output():
    assert decimate_2x(np.empty(0, np.float32)).shape == (0,)
    assert interpolate_2x(np.empty(0, np.float32)).shape == (0,)


@pytest.mark.parametrize("fn", [decimate_2x, interpolate_2x])
def test_rejects_wrong_dtype(fn):
    with pytest.raises(TypeError):
        fn(np.zeros(10, dtype=np.float64))


@pytest.mark.parametrize("fn", [decimate_2x, interpolate_2x])
def test_rejects_wrong_dimension(fn):
    with pytest.raises(TypeError):
        fn(np.zeros((4, 4), np.float32))
    with pytest.raises(TypeError):
        fn(np.float32(0.0))


@pytest.mark.parametrize("fn", [decimate_2x, interpolate_2x])
def test_rejects_a_strided_view(fn):
    with pytest.raises(ValueError, match="C-contiguous"):
        fn(np.zeros(20, np.float32)[::2])


@pytest.mark.parametrize("fn", [decimate_2x, interpolate_2x])
def test_output_is_a_new_writable_array(fn):
    x = np.ones(64, np.float32)
    y = fn(x)
    assert y.flags.writeable
    assert not np.shares_memory(x, y)
    y[0] = 5.0
    assert x[0] == 1.0
