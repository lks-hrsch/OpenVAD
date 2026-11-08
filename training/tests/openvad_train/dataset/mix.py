# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest
import soundfile as sf
from scipy.special import digamma

from openvad_train.dataset.codecs import DecodeError
from openvad_train.dataset.mix import (
    _WADA_DB,
    _WADA_G,
    add_transients,
    dither,
    loop_slice,
    mix_at_snr,
    overlay_transient,
    padded_length,
    place,
    protect_peak,
    read_span,
    synth_click,
    synth_dtmf,
    wada_snr,
)


def sine(n=16000):
    return (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / 16000)).astype(np.float32)


def white(n=16000, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1


def speech_like(n=160000, seed=1):
    """Gamma(0.4) amplitudes with random signs: the amplitude model WADA-SNR assumes."""
    rng = np.random.default_rng(seed)
    return (rng.gamma(0.4, 0.2, n) * rng.choice([-1, 1], n)).astype(np.float32)


def snr_of(speech, noise, active=None):
    active = np.ones(len(speech), bool) if active is None else active
    return 10 * np.log10(np.mean(speech[active] ** 2) / np.mean(noise[active] ** 2))


def test_measured_snr_matches_draw():
    speech, noise = sine(), white()
    out = mix_at_snr(speech, noise, np.ones_like(speech, bool), 5.0)
    assert out.dtype == np.float32
    assert snr_of(speech, out - speech) == pytest.approx(5.0, abs=0.5)


def test_snr_uses_active_samples_but_scales_all_noise():
    speech = place(sine(8000), 16000, 4000)
    noise = white()
    active = np.zeros(16000, bool)
    active[4000:12000] = True
    out = mix_at_snr(speech, noise, active, 10.0)
    assert snr_of(speech, out - speech, active) == pytest.approx(10.0, abs=0.01)
    assert np.abs(out[:4000]).max() > 0  # the gain also reaches the padding


def test_silent_noise_or_no_active_samples_raise():
    speech = sine()
    everywhere = np.ones_like(speech, bool)
    with pytest.raises(ValueError, match="silent"):
        mix_at_snr(speech, np.zeros_like(speech), everywhere, 5.0)
    with pytest.raises(ValueError, match="active"):
        mix_at_snr(speech, white(), np.zeros_like(speech, bool), 5.0)


def test_non_boolean_active_mask_raises():
    speech = sine()
    with pytest.raises(TypeError, match="boolean"):
        mix_at_snr(speech, white(), np.ones_like(speech, np.int8), 5.0)


def test_place_puts_speech_in_one_block():
    out = place(np.ones(3, np.float32), 8, 2)
    assert out.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="fit"):
        place(np.ones(3, np.float32), 8, 6)


def test_noise_shorter_than_sample_loops():
    assert loop_slice(np.arange(4.0), 3, 6).tolist() == [3, 0, 1, 2, 3, 0]


def test_loop_slice_rejects_stereo_empty_and_negative_length():
    with pytest.raises(ValueError, match="one-dimensional"):
        loop_slice(np.zeros((4, 2)), 0, 3)
    with pytest.raises(ValueError, match="one-dimensional"):
        loop_slice(np.zeros(0), 0, 3)
    with pytest.raises(ValueError, match="samples"):
        loop_slice(np.arange(4.0), 0, -1)


def test_dither_level_and_reproducibility():
    d = dither(160000, -80.0, np.random.default_rng(3))
    assert 20 * np.log10(np.sqrt(np.mean(d.astype(np.float64) ** 2))) == pytest.approx(-80.0, abs=0.2)
    assert np.array_equal(d, dither(160000, -80.0, np.random.default_rng(3)))


def test_peak_protection_keeps_snr():
    x = np.array([2.0, -1.0, 0.5])
    y = protect_peak(x)
    assert np.abs(y).max() <= 1.0
    assert y[0] / y[1] == pytest.approx(-2.0)


def test_peak_protection_leaves_quiet_samples_alone():
    x = np.array([1.0, -0.5], np.float32)
    assert protect_peak(x).tolist() == x.tolist()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_peak_protection_rejects_non_finite_samples(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        protect_peak(np.array([0.5, bad], np.float32))


def test_transient_peak_ratio_and_cut_at_end():
    x = sine(1000)  # peak 0.3
    out = overlay_transient(x, np.array([0.0, 1.0, -0.5], np.float32), 998, -6.0)
    assert out[999] - x[999] == pytest.approx(0.3 * 10 ** (-6 / 20), abs=1e-4)
    assert out[998] == x[998]
    assert len(out) == 1000


def test_transient_scaled_against_peak_of_x():
    x = np.zeros(10, np.float32)
    x[0] = 0.5
    out = overlay_transient(x, np.array([0.0, 2.0, -1.0], np.float32), 4, 0.0)
    assert out[5] == pytest.approx(0.5)
    assert out[6] == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="offset"):
        overlay_transient(x, np.ones(2, np.float32), 10, 0.0)


@pytest.mark.parametrize("synth", [synth_click, synth_dtmf])
def test_synth_events_have_unit_peak_and_are_reproducible(synth):
    event = synth(np.random.default_rng(5))
    assert event.dtype == np.float32
    assert np.abs(event).max() == pytest.approx(1.0)
    assert np.array_equal(event, synth(np.random.default_rng(5)))


def test_padded_length_gives_33_percent():
    assert 3.3 / padded_length(3.3) == pytest.approx(0.33, abs=0.01)


def test_padded_length_counts_edges_and_never_shortens():
    assert padded_length(3.3, edge_s=1.0) == pytest.approx(10.0)  # edges are inside the total
    assert padded_length(3.3, edge_s=20.0) == pytest.approx(23.3)  # already below the target ratio


def test_padded_length_rejects_nonsense():
    with pytest.raises(ValueError, match="padding"):
        padded_length(0.0)


def test_wada_snr_orders_clean_above_noisy():
    clean = speech_like()
    assert wada_snr(clean) > wada_snr(clean + white(len(clean), 2))


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_wada_snr_recovers_known_snr(snr_db):
    speech = speech_like()
    noise = white(len(speech), 7)
    noisy = mix_at_snr(speech, noise, np.ones_like(speech, bool), snr_db)
    assert wada_snr(noisy) == pytest.approx(snr_db, abs=2.0)


def test_wada_snr_rejects_silence():
    with pytest.raises(ValueError, match="silent"):
        wada_snr(np.zeros(100, np.float32))


def test_wada_snr_counts_exact_zeros_as_near_silence_not_as_missing():
    x = speech_like()
    x = mix_at_snr(x, white(len(x), 7), np.ones_like(x, bool), 5.0)
    with_zeros = x.copy()
    with_zeros[::2] = 0.0
    assert wada_snr(with_zeros) > wada_snr(x[1::2]) + 10  # dropping the zeros would give the same value


def test_wada_snr_does_not_depend_on_float_scale_of_16_bit_audio():
    ints = np.round(speech_like() * 32768).astype(np.float32)  # many exact zeros after quantisation
    assert (ints == 0).any()
    assert wada_snr(ints / 32768) == pytest.approx(wada_snr(ints))


def test_wada_table_spans_minus_20_to_100_between_the_analytic_limits():
    assert len(_WADA_G) == len(_WADA_DB) == 121
    assert np.all(np.diff(_WADA_G) >= 0)
    assert _WADA_G[0] == pytest.approx(0.5 * np.log(2 / np.pi) - (digamma(0.5) + np.log(2)) / 2, abs=1e-3)  # Gaussian
    assert _WADA_G[-1] == pytest.approx(np.log(0.4) - digamma(0.4), abs=0.02)  # clean Gamma(0.4)


def test_wada_snr_clamps_at_100_for_clean_speech():
    assert wada_snr(speech_like()) == 100.0


def test_read_span_reads_from_the_offset_and_repeats_what_is_left_when_the_file_ends(tmp_path):
    x = np.arange(1, 801, dtype=np.float32) / 1000
    sf.write(tmp_path / "a.wav", x, 16000, subtype="FLOAT")
    assert np.array_equal(read_span(str(tmp_path / "a.wav"), 100, 300), x[100:400])
    wrapped = read_span(str(tmp_path / "a.wav"), 600, 500)
    assert np.array_equal(wrapped, np.tile(x[600:], 3)[:500])


def test_read_span_refuses_silence_and_an_offset_past_the_end(tmp_path):
    sf.write(tmp_path / "zero.wav", np.zeros(800, np.float32), 16000, subtype="FLOAT")
    sf.write(tmp_path / "a.wav", np.ones(800, np.float32) / 2, 16000, subtype="FLOAT")
    with pytest.raises(DecodeError, match="silence"):
        read_span(str(tmp_path / "zero.wav"), 0, 100)
    with pytest.raises(DecodeError, match="no audio"):
        read_span(str(tmp_path / "a.wav"), 900, 100)


def test_add_transients_puts_each_event_at_its_offset_and_peak_ratio():
    x = sine(16000)  # peak 0.3
    out = add_transients(x, [{"kind": "click", "offset_n": 4000, "peak_db": -6.0}], np.random.default_rng(0))
    added = out - x
    assert not added[:4000].any()
    assert not added[4000 + 80 + 1 :].any()
    assert np.abs(added).max() == pytest.approx(0.3 * 10 ** (-6 / 20), rel=0.02)
    dtmf = add_transients(x, [{"kind": "dtmf", "offset_n": 8000, "peak_db": 0.0}], np.random.default_rng(0)) - x
    assert np.abs(dtmf).max() == pytest.approx(0.3, rel=0.02)
    assert np.count_nonzero(dtmf) > 800  # 50 to 150 ms


def test_add_transients_is_reproducible_and_refuses_an_unknown_kind():
    x, event = sine(16000), [{"kind": "dtmf", "offset_n": 100, "peak_db": -3.0}]
    assert np.array_equal(
        add_transients(x, event, np.random.default_rng(1)), add_transients(x, event, np.random.default_rng(1))
    )
    assert np.array_equal(add_transients(x, [], np.random.default_rng(1)), x)
    with pytest.raises(ValueError, match="typing"):
        add_transients(x, [{"kind": "typing", "offset_n": 0, "peak_db": 0.0}], np.random.default_rng(1))


def test_a_transient_at_0_db_can_double_the_peak_so_protect_peak_must_come_after_it():
    x = sine(16000) / 0.3 * 0.9  # peak 0.9, as a loud mix leaves it
    out = add_transients(x, [{"kind": "click", "offset_n": 0, "peak_db": 0.0}], np.random.default_rng(0))
    assert np.abs(out).max() > 1.0
    assert np.abs(protect_peak(out)).max() <= 1.0
