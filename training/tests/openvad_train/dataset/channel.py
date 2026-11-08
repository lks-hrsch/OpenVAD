# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from itertools import chain

import numpy as np
import pytest
import soundfile as sf

from openvad_train.dataset.channel import (
    SUPPRESSION_METHODS,
    RirPool,
    agc,
    apply_rir,
    handset,
    load_rir,
    soft_clip,
    split_rooms,
    suppress,
)
from rir_support import make_rir_tree, synthetic_rir


RATE = 16000


def tone(freq: float, amp: float = 0.5, seconds: float = 1.0) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(round(seconds * RATE)) / RATE)).astype(np.float32)


def white(seconds: float = 4.0, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(round(seconds * RATE))).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def dbfs(x: np.ndarray) -> float:
    return 20 * float(np.log10(rms(x)))


def impulse_at(i: int, n: int = 8000) -> np.ndarray:
    x = np.zeros(n, np.float32)
    x[i] = 1.0
    return x


FLAT = {"highpass_hz": 20.0, "lowpass_hz": 7500.0, "peaks": []}


def test_rir_does_not_shift_onset():
    y = apply_rir(impulse_at(1000), synthetic_rir(predelay_n=50))
    assert np.argmax(np.abs(y)) == 1000


def test_rir_keeps_length_and_adds_a_tail():
    x = impulse_at(1000)
    y = apply_rir(x, synthetic_rir())
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.abs(y[1001:3000]).max() > 0.01


def test_rir_keeps_the_power_of_broadband_audio():
    x = white()
    assert dbfs(apply_rir(x, synthetic_rir())) == pytest.approx(dbfs(x), abs=1.0)


def test_rir_cut_at_its_peak_not_at_its_start():
    # A reflection stronger than the direct path would be the peak: the response starts there.
    rir = np.zeros(100, np.float32)
    rir[10], rir[30] = 0.5, 1.0
    assert np.argmax(np.abs(apply_rir(impulse_at(500), rir))) == 500


@pytest.mark.parametrize("rir", [np.zeros(10, np.float32), np.empty(0, np.float32)], ids=["silent", "empty"])
def test_a_rir_without_a_direct_path_is_refused(rir):
    with pytest.raises(ValueError, match="direct path"):
        apply_rir(white(), rir)


def test_load_rir_takes_the_first_channel_and_demands_16_khz(tmp_path):
    two = np.stack([synthetic_rir(), -synthetic_rir()], axis=1)
    sf.write(tmp_path / "a.wav", two, RATE, subtype="FLOAT")
    np.testing.assert_array_equal(load_rir(tmp_path / "a.wav"), synthetic_rir())
    sf.write(tmp_path / "b.wav", synthetic_rir(), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="16 kHz"):
        load_rir(tmp_path / "b.wav")


def test_handset_highpass_removes_low_tones():
    params = FLAT | {"highpass_hz": 200.0, "lowpass_hz": 3400.0}
    assert rms(handset(tone(50), params)) < 0.1 * rms(tone(50))
    assert rms(handset(tone(1000), params)) == pytest.approx(rms(tone(1000)), rel=0.05)


def test_handset_lowpass_removes_high_tones():
    params = FLAT | {"highpass_hz": 200.0, "lowpass_hz": 3400.0}
    assert rms(handset(tone(7000), params)) < 0.1 * rms(tone(7000))


@pytest.mark.parametrize("gain_db", [6.0, -6.0])
def test_handset_peak_has_the_drawn_gain(gain_db):
    params = FLAT | {"peaks": [{"freq_hz": 1500.0, "gain_db": gain_db}]}
    ratio = rms(handset(tone(1500), params)) / rms(tone(1500))
    assert 20 * np.log10(ratio) == pytest.approx(gain_db, abs=0.5)


def test_handset_adds_no_delay():
    params = FLAT | {"highpass_hz": 300.0, "lowpass_hz": 3400.0}
    assert np.argmax(np.abs(handset(impulse_at(4000), params))) == 4000


def test_agc_reaches_target():
    params = {"target_dbfs": -20.0, "attack_ms": 10.0, "release_ms": 200.0}
    assert dbfs(agc(tone(1000, amp=0.01), params)) == pytest.approx(-20, abs=2)
    assert dbfs(agc(tone(1000, amp=0.9), params)) == pytest.approx(-20, abs=2)


def test_agc_release_time_sets_how_long_a_loud_burst_keeps_the_gain_down():
    x = np.concatenate([tone(1000, 0.8, 0.5), tone(1000, 0.01, 1.0)])
    fast = agc(x, {"target_dbfs": -20.0, "attack_ms": 5.0, "release_ms": 100.0})
    slow = agc(x, {"target_dbfs": -20.0, "attack_ms": 5.0, "release_ms": 500.0})
    window = slice(RATE // 2 + RATE // 5, RATE // 2 + RATE // 4)  # 200 to 250 ms after the burst
    assert rms(slow[window]) < rms(fast[window])


def test_agc_does_not_raise_digital_silence():
    out = agc(np.zeros(RATE, np.float32), {"target_dbfs": -20.0, "attack_ms": 10.0, "release_ms": 200.0})
    assert not out.any()


@pytest.mark.parametrize("n", [RATE, RATE + 7])
def test_agc_keeps_length_and_dtype(n):
    out = agc(white()[:n], {"target_dbfs": -20.0, "attack_ms": 10.0, "release_ms": 200.0})
    assert out.shape == (n,)
    assert out.dtype == np.float32


def test_clip_stays_in_range():
    assert np.abs(soft_clip(tone(1000, amp=1.0), 12)).max() <= 1.0


def test_clip_keeps_the_peak_and_drive_flattens_it():
    x = tone(1000, amp=0.2)
    mild, hard = soft_clip(x, 3), soft_clip(x, 12)
    assert np.abs(mild).max() == pytest.approx(0.2, rel=1e-3)
    assert np.abs(hard).max() == pytest.approx(0.2, rel=1e-3)
    assert rms(hard) > rms(mild) > rms(x) * 0.95  # a flatter top has more power at the same peak


def test_clip_of_silence_is_silence():
    assert not soft_clip(np.zeros(100, np.float32), 12).any()


def test_suppress_lowers_stationary_noise():
    noise = white()
    for method in SUPPRESSION_METHODS:
        out = suppress(noise, method)
        assert out.shape == noise.shape
        assert out.dtype == np.float32
        assert rms(out) < 0.7 * rms(noise)


def test_suppress_keeps_speech_like_bursts_better_than_noise():
    noise = white(seed=1)
    x = noise.copy()
    x[RATE : 2 * RATE] += tone(700, 0.3)[:RATE]
    for method in SUPPRESSION_METHODS:
        out = suppress(x, method)
        assert rms(out[RATE : 2 * RATE]) > 0.8 * rms(x[RATE : 2 * RATE])
        assert rms(out[3 * RATE :]) < 0.7 * rms(noise[3 * RATE :])


def test_suppress_leaves_digital_silence_silent_and_finite():
    out = suppress(np.zeros(RATE, np.float32), "wiener")
    assert np.isfinite(out).all()
    assert not out.any()


def test_suppress_refuses_an_unknown_method():
    with pytest.raises(ValueError, match="rnnoise"):
        suppress(white(), "rnnoise")


ROOMS = {f"but-R{i:02d}" for i in range(30)}


def test_rooms_split_80_10_10_without_overlap():
    split = split_rooms(ROOMS)
    assert {k: len(v) for k, v in split.items()} == {"train": 24, "validation": 3, "test": 3}
    assert sorted(chain.from_iterable(split.values())) == sorted(ROOMS)


def test_every_split_gets_a_room_even_when_there_are_few():
    assert {k: len(v) for k, v in split_rooms({"a", "b", "c", "d", "e"}).items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    with pytest.raises(ValueError, match="at least 3"):
        split_rooms({"a", "b"})


def test_the_pool_does_not_depend_on_the_order_of_the_index(tmp_path):
    root = make_rir_tree(tmp_path / "rirs")
    before = RirPool(root)
    header, *lines = (root / "index.tsv").read_text().splitlines()
    (root / "index.tsv").write_text("\n".join([header, *reversed(lines)]) + "\n")
    after = RirPool(root)
    assert [after.rooms(s) for s in ("train", "validation", "test")] == [
        before.rooms(s) for s in ("train", "validation", "test")
    ]
    assert after.draw("train", np.random.default_rng(3)) == before.draw("train", np.random.default_rng(3))


@pytest.fixture
def rir_pool(tmp_path) -> RirPool:
    return RirPool(make_rir_tree(tmp_path / "rirs"))


def test_pool_needs_the_fetched_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch rirs"):
        RirPool(tmp_path)


def test_rooms_are_disjoint_between_splits(rir_pool):
    rooms = {s: set(rir_pool.rooms(s)) for s in ("train", "validation", "test")}
    assert rooms["train"].isdisjoint(rooms["validation"])
    assert rooms["train"].isdisjoint(rooms["test"])
    assert rooms["validation"].isdisjoint(rooms["test"])
    assert len(rooms["validation"]) == len(rooms["test"]) == 2
    drawn = {s: {rir_pool.draw(s, np.random.default_rng(i)).room for i in range(300)} for s in rooms}
    assert drawn == rooms


def test_a_draw_is_a_function_of_the_rng(rir_pool):
    a = [rir_pool.draw("train", np.random.default_rng(i)) for i in range(20)]
    assert a == [rir_pool.draw("train", np.random.default_rng(i)) for i in range(20)]
    assert all(d.file.is_file() and rir_pool.room_of(d.file) == d.room for d in a)


def test_a_draw_picks_the_room_before_the_rir(tmp_path):
    root = make_rir_tree(tmp_path / "rirs", rooms=10, per_room=1)
    # One room gets 49 more RIRs; it must still be drawn about as often as any other room of its split.
    extra = root / "openslr28"
    lines = (root / "index.tsv").read_text().splitlines()
    room = next(line.split("\t")[1] for line in lines if line.split("\t")[1].startswith("openslr28"))
    for k in range(49):
        sf.write(extra / f"extra{k}.wav", synthetic_rir(k), RATE, subtype="FLOAT")
        lines.append(f"openslr28/extra{k}.wav\t{room}")
    (root / "index.tsv").write_text("\n".join(lines) + "\n")
    pool = RirPool(root)
    split = next(s for s in ("train", "validation", "test") if room in pool.rooms(s))
    rng = np.random.default_rng(0)
    share = np.mean([pool.draw(split, rng).room == room for _ in range(2000)])
    assert share == pytest.approx(1 / len(pool.rooms(split)), abs=0.05)
