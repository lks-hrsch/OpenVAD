# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import subprocess
import sys
from typing import cast

import numpy as np
import pytest
from scipy import signal

from openvad_rs import GROUP_DELAY_16K, decimate_2x, interpolate_2x
from openvad_train.dataset.codecs import MissingCodec, delay_ms, narrow_roundtrip, preflight
from openvad_train.dataset.construct import codec_delay_ms
from openvad_train.dataset.g711 import alaw_roundtrip, mulaw_roundtrip
from openvad_train.dataset.narrowband import (
    apply_narrowband,
    eval_seed,
    label_shift_frames,
    label_shift_samples,
    resample_only,
    telephone,
)
from openvad_train.dataset.schema import rank, sample_seed


def noise(n: int = 16000, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(-0.5, 0.5, n)).astype(np.float32)


def tone(freq: float, n: int = 16000) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / 16000)).astype(np.float32)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class _FixedDraw:
    def __init__(self, u: float):
        self.u = u

    def random(self) -> float:
        return self.u


def fixed_draw(u: float) -> np.random.Generator:
    """A stand-in generator whose single draw is `u`, to pick the G.711 law."""
    return cast(np.random.Generator, _FixedDraw(u))


def through(x: np.ndarray, law) -> np.ndarray:
    return interpolate_2x(law(decimate_2x(x)))[: len(x)]


def test_native_is_untouched():
    x = noise()
    assert apply_narrowband(x, "narrow_native", rng(), 1.0) is x


def test_leg_always_processed():
    x = noise()
    assert not np.array_equal(apply_narrowband(x, "narrow_leg", rng(), 0.0), x)


def test_wide_respects_p():
    x = noise()
    assert apply_narrowband(x, "wide", rng(), 0.0) is x
    assert not np.array_equal(apply_narrowband(x, "wide", rng(), 1.0), x)


def test_wide_takes_the_telephone_path_about_p_of_the_time():
    x = noise(320)
    hits = sum(apply_narrowband(x, "wide", rng(s), 0.35) is not x for s in range(2000))
    assert 0.31 < hits / 2000 < 0.39


def test_unknown_band_raises():
    with pytest.raises(ValueError, match="unknown band"):
        apply_narrowband(noise(), "Wide", rng())


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_p_narrow_outside_the_unit_interval_raises(p):
    with pytest.raises(ValueError, match="p_narrow"):
        apply_narrowband(noise(), "wide", rng(), p)


@pytest.mark.parametrize("band", ["wide", "narrow_leg", "narrow_native"])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_apply_narrowband_takes_two_draws_whatever_the_outcome(band, p):
    used, fresh = rng(7), rng(7)
    apply_narrowband(noise(320), band, used, p)
    fresh.random(2)
    assert used.random() == fresh.random()


def test_telephone_takes_one_draw():
    used, fresh = rng(7), rng(7)
    telephone(noise(320), used)
    fresh.random()
    assert used.random() == fresh.random()


def test_telephone_law_split_is_nine_to_one():
    x = noise(1600)
    assert np.array_equal(telephone(x, fixed_draw(0.89)), through(x, alaw_roundtrip))
    assert np.array_equal(telephone(x, fixed_draw(0.91)), through(x, mulaw_roundtrip))


@pytest.mark.parametrize(("seed", "law"), [(1, mulaw_roundtrip), (4, alaw_roundtrip)])
def test_apply_narrowband_uses_its_second_draw_for_the_law(seed, law):
    # Seed 1 draws [0.51, 0.95] and seed 4 [0.94, 0.51]: the law differs from what the first draw alone would pick.
    x = noise(1600)
    assert np.array_equal(apply_narrowband(x, "narrow_leg", rng(seed)), through(x, law))
    assert not np.array_equal(through(x, alaw_roundtrip), through(x, mulaw_roundtrip))


def test_length_is_preserved():
    assert len(telephone(noise(16001), rng())) == 16001


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_edge_lengths_keep_their_length(n):
    x = noise(n)
    assert len(resample_only(x)) == n
    assert len(telephone(x, rng())) == n


def test_resample_only_has_no_quantisation_noise():
    x = tone(1000)
    delay = label_shift_samples()
    error = resample_only(x)[delay:] - x[:-delay]
    snr_db = 10 * np.log10(np.mean(x[:-delay] ** 2) / np.mean(error[delay:] ** 2))
    assert snr_db > 60


def test_telephone_has_quantisation_noise():
    x = tone(1000)
    delay = label_shift_samples()
    assert np.max(np.abs(telephone(x, rng())[delay:] - resample_only(x)[delay:])) > 1e-4


def test_label_shift_is_the_round_trip_and_rounds_to_zero_frames():
    assert label_shift_samples() == 2 * GROUP_DELAY_16K == 66
    assert label_shift_frames() == 0


def test_label_shift_rounds_up_for_a_longer_filter():
    # An 83-tap filter delays by 41 samples one way, 82 in all, which is 0.5125 of a frame.
    assert label_shift_samples(41) == 82
    assert label_shift_frames(41) == 1


def test_narrow_leg_residual_lag_stays_under_one_frame():
    resampler_ms = label_shift_samples() / 16
    stored_ms = delay_ms("amr_nb") + resampler_ms
    labelled_ms = round(stored_ms)
    residual_ms = (stored_ms - labelled_ms) + resampler_ms
    assert residual_ms == pytest.approx(4.25)
    assert residual_ms < 10


def test_narrow_leg_residual_lag_measured_end_to_end():
    # The stored AMR-NB leg and the train-time transform, with the shift construction really uses.
    try:
        preflight({"libopencore_amrnb"})
    except MissingCodec as e:
        pytest.skip(str(e))
    n = 2 * 16000
    x = (0.5 * signal.chirp(np.arange(n) / 16000, 100, 2.0, 3500)).astype(np.float32)
    x[:2000] = 0
    out = telephone(narrow_roundtrip(x, "amr_nb", 12200), rng())
    lag_ms = (np.argmax(signal.correlate(out, x, mode="full", method="fft")) - (n - 1)) / 16
    assert lag_ms == pytest.approx(13.25, abs=0.25)  # 5.0 codec + 4.125 resampler in the leg + 4.125 in the transform
    residual_ms = lag_ms - codec_delay_ms({"codec": "amr_nb"})
    assert residual_ms == pytest.approx(4.25, abs=0.25)
    assert residual_ms < 10


@pytest.mark.parametrize("fn", [resample_only, lambda x: telephone(x, rng())])
def test_input_is_unchanged_and_output_is_a_new_float32_array(fn):
    x = noise(1601)
    before = x.copy()
    out = fn(x)
    assert out.dtype == np.float32
    assert out is not x
    assert np.array_equal(x, before)


@pytest.mark.parametrize("law", [alaw_roundtrip, mulaw_roundtrip])
def test_overshoot_beyond_full_scale_is_clipped_not_wrapped(law):
    # A full-scale square wave leaves [-1, 1] after the decimator; a wrapped table index would flip its sign.
    x = np.tile(np.repeat(np.array([1.0, -1.0], np.float32), 20), 50)
    narrow = decimate_2x(x)
    assert np.abs(narrow).max() > 1.0
    out = law(narrow)
    assert np.abs(out).max() <= 1.0
    assert np.array_equal(np.sign(out[40:-40]), np.sign(narrow[40:-40]))


@pytest.mark.parametrize("u", [0.0, 0.99])
def test_telephone_survives_input_beyond_full_scale(u):
    x = np.tile(np.repeat(np.array([3.0, -3.0], np.float32), 20), 50)
    out = telephone(x, fixed_draw(u))
    assert len(out) == len(x)
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 2.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("fn", [resample_only, lambda x: telephone(x, rng())])
def test_non_finite_input_raises_with_the_count(fn, bad):
    x = noise(100)
    x[[3, 50, 51]] = bad
    with pytest.raises(ValueError, match="3 non-finite"):
        fn(x)


def test_eval_draw_is_repeatable():
    a = telephone(noise(), np.random.default_rng(eval_seed("x")))
    assert np.array_equal(a, telephone(noise(), np.random.default_rng(eval_seed("x"))))


def test_eval_seed_has_its_own_domain():
    assert eval_seed("x") != rank("x")
    assert eval_seed("x") != sample_seed(0, "x", "x")
    assert eval_seed("x") != eval_seed("y")


def test_eval_seed_does_not_depend_on_the_hash_seed():
    code = "from openvad_train.dataset.narrowband import eval_seed; print(eval_seed('clip-1'))"
    seen = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("0", "12345")
    }
    assert seen == {f"{eval_seed('clip-1')}\n"}
