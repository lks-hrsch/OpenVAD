# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import csv
import dataclasses
from collections import Counter

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from demand_support import make_demand_tree
from noise_support import make_musan_tree, make_wham_tree, talker
from openvad_train.dataset.codecs import DecodeError
from openvad_train.dataset.config import DEMAND_CATEGORIES
from openvad_train.dataset.mix import rms
from openvad_train.dataset.noise import (
    MUSAN_CATEGORIES,
    OPEN_LANGUAGES,
    NoisePool,
    Segment,
    TalkerPool,
    build_babble,
    build_ssn,
    category_of_demand,
    demand_channels,
    fit_lpc,
    mix_competing,
    musan_noise_categories,
    talker_offset,
    wham_split,
)
from openvad_train.fetch.musan import musan_licences
from rir_support import make_rir_tree


@pytest.fixture
def demand_tree(tmp_path):
    return make_demand_tree(tmp_path / "demand", ["DKITCHEN", "DLIVING", "TBUS"])


def test_demand_scene_categories():
    assert category_of_demand("DKITCHEN") == "domestic"
    assert category_of_demand("SPSQUARE") == "street"
    assert category_of_demand("TBUS") == "transport"
    with pytest.raises(ValueError, match="scene"):
        category_of_demand("XYZ")


def test_demand_channels_split_like_librivad():
    assert list(demand_channels("train")) == list(range(1, 13))
    assert list(demand_channels("validation")) == [13, 14]
    assert list(demand_channels("test")) == [15, 16]
    with pytest.raises(ValueError, match="split"):
        demand_channels("dev")


def test_pool_draws_only_split_channels(demand_tree):
    pool = NoisePool(demand_tree)
    draws = [pool.draw("domestic", "test", np.random.default_rng(i)) for i in range(50)]
    assert {d.channel for d in draws} <= {15, 16}
    assert {d.channel for d in draws} == {15, 16}
    assert {d.file.parent.name for d in draws} == {"DKITCHEN", "DLIVING"}
    assert all(0 <= d.offset_n < 1600 and d.file.name == f"ch{d.channel}.wav" for d in draws)


def test_pool_draw_is_a_function_of_rng_and_not_listing_order(tmp_path):
    forward = NoisePool(make_demand_tree(tmp_path / "a", ["DKITCHEN", "DLIVING", "DWASHING"]))
    backward = NoisePool(make_demand_tree(tmp_path / "b", ["DWASHING", "DLIVING", "DKITCHEN"]))
    for i in range(20):
        a = forward.draw("domestic", "train", np.random.default_rng(i))
        b = backward.draw("domestic", "train", np.random.default_rng(i))
        assert (a.file.relative_to(tmp_path / "a"), a.channel, a.offset_n) == (
            b.file.relative_to(tmp_path / "b"),
            b.channel,
            b.offset_n,
        )


def test_pool_rejects_unknown_or_empty_category(demand_tree):
    pool = NoisePool(demand_tree)
    with pytest.raises(ValueError, match="no noise in category"):
        pool.draw("street", "train", np.random.default_rng(0))
    with pytest.raises(ValueError, match="category"):
        pool.draw("music", "train", np.random.default_rng(0))


def test_pool_reads_each_header_once(demand_tree, monkeypatch):
    reads = []
    real_info = sf.info
    monkeypatch.setattr(sf, "info", lambda file: reads.append(file) or real_info(file))
    pool = NoisePool(demand_tree)
    draws = [pool.draw("transport", "test", np.random.default_rng(i)) for i in range(40)]
    assert len(reads) == len(set(reads))
    assert {d.file for d in draws} <= set(reads)
    assert draws[0] == NoisePool(demand_tree).draw("transport", "test", np.random.default_rng(0))  # caching keeps draws


def test_the_musan_table_has_one_row_for_every_noise_file_of_the_licence_table():
    noise_files = {f for f in musan_licences() if f.startswith("noise-")}
    with MUSAN_CATEGORIES.open() as f:
        listed = [row[0] for row in csv.reader(f, delimiter="\t")][1:]
    assert len(listed) == len(set(listed))
    assert set(listed) == noise_files == set(musan_noise_categories())


def test_musan_files_only_join_categories_that_are_drawn_from_files():
    # Babble and SSN are built from reserved speakers, and City is WHAM!, so no MUSAN file can belong to them.
    assert set(musan_noise_categories().values()) <= {*DEMAND_CATEGORIES, "unused"}
    assert any(c != "unused" for c in musan_noise_categories().values())


def test_wham_split_mapping():
    assert wham_split("train") == "tr"
    assert wham_split("validation") == "cv"
    assert wham_split("test") == "tt"
    with pytest.raises(ValueError, match="split"):
        wham_split("dev")


@pytest.mark.parametrize(("split", "folder"), [("train", "tr"), ("validation", "cv"), ("test", "tt")])
def test_city_noise_is_a_wham_file_of_the_split_and_one_of_its_channels(tmp_path, split, folder):
    pool = NoisePool(wham=make_wham_tree(tmp_path / "wham"))
    draws = [pool.draw("city", split, np.random.default_rng(i)) for i in range(60)]
    assert {d.file.parent.name for d in draws} == {folder}
    assert {d.channel for d in draws} == {0, 1}
    assert all(d.stereo and 0 <= d.offset_n < 1600 for d in draws)
    assert {d.source_id(tmp_path) for d in draws} >= {f"wham/wham_noise/{folder}/{folder}000.wav#0"}


def test_the_test_split_of_wham_never_reaches_train_or_validation(tmp_path):
    pool = NoisePool(wham=make_wham_tree(tmp_path / "wham"))
    ids = [
        pool.draw("city", s, np.random.default_rng(i)).source_id(tmp_path)
        for i in range(200)
        for s in ("train", "validation")
    ]
    assert not any("/tt/" in i for i in ids)


def test_city_needs_wham():
    with pytest.raises(ValueError, match="WHAM"):
        NoisePool().draw("city", "train", np.random.default_rng(0))


def test_a_musan_category_draws_its_files_in_proportion_to_their_duration(tmp_path):
    # Scenes of 1600 samples and MUSAN files of 160: the four domestic MUSAN files hold 640 of 3840 samples.
    pool = NoisePool(
        make_demand_tree(tmp_path / "demand", ["DKITCHEN", "DLIVING"]), musan=make_musan_tree(tmp_path / "musan")
    )
    draws = [pool.draw("domestic", "test", np.random.default_rng(i)) for i in range(3000)]
    musan = [d for d in draws if "musan" in d.file.parts]
    assert len(musan) / len(draws) == pytest.approx(640 / 3840, abs=0.03)
    assert {d.channel for d in musan} == {0}
    assert all(0 <= d.offset_n < 160 and not d.stereo for d in musan)
    ids = {musan_noise_categories()[d.file.stem] for d in musan}
    assert ids == {"domestic"}
    assert {d.channel for d in draws if "musan" not in d.file.parts} == {15, 16}


def test_musan_is_shared_by_every_split(tmp_path):
    pool = NoisePool(musan=make_musan_tree(tmp_path / "musan"))
    for split in ("train", "validation", "test"):
        assert {pool.draw("nature", split, np.random.default_rng(i)).file for i in range(200)} == {
            pool.draw("nature", "train", np.random.default_rng(i)).file for i in range(200)
        }


def test_a_musan_file_that_is_missing_is_an_error_at_construction(tmp_path):
    root = make_musan_tree(tmp_path / "musan")
    next(root.glob("musan/noise/sound-bible/*.wav")).unlink()
    with pytest.raises(FileNotFoundError, match="MUSAN is incomplete"):
        NoisePool(musan=root)


def test_a_draw_with_musan_does_not_depend_on_the_listing_order(tmp_path):
    a = NoisePool(make_demand_tree(tmp_path / "a", ["DWASHING", "DKITCHEN"]), musan=make_musan_tree(tmp_path / "ma"))
    b = NoisePool(make_demand_tree(tmp_path / "b", ["DKITCHEN", "DWASHING"]), musan=make_musan_tree(tmp_path / "mb"))
    for i in range(30):
        x, y = (
            a.draw("domestic", "train", np.random.default_rng(i)),
            b.draw("domestic", "train", np.random.default_rng(i)),
        )
        assert (x.file.name, x.channel, x.offset_n) == (y.file.name, y.channel, y.offset_n)


def reserved_pool(languages: dict[str, int], split: str = "train", min_speakers: int = 3) -> TalkerPool:
    clips = [talker(f"{lang}{i}", lang, split) for lang, n in languages.items() for i in range(n)]
    return TalkerPool(clips, min_speakers)


def test_babble_talkers_are_reserved_of_the_same_language_and_split_with_different_speakers():
    pool = reserved_pool({"de": 10, "fr": 10})
    pool_test = TalkerPool([talker(f"t{i}", "de", "test") for i in range(10)], 3)
    for i in range(50):
        picked = pool.draw("de", "train", 6, np.random.default_rng(i))
        assert {c.language for c in picked} == {"de"}
        assert len({c.speaker for c in picked}) == 6
        assert {c.split for c in pool_test.draw("de", "test", 6, np.random.default_rng(i))} == {"test"}


def test_a_split_never_hears_the_reserved_speakers_of_another():
    train, test = [talker(f"a{i}", "de", "train") for i in range(8)], [talker(f"b{i}", "de", "test") for i in range(8)]
    pool = TalkerPool(train + test, 3)
    for split, own in (("train", train), ("validation", None), ("test", test)):
        if own is None:
            with pytest.raises(ValueError, match="validation"):
                pool.draw("de", split, 6, np.random.default_rng(0))
            continue
        assert all(c in own for i in range(30) for c in pool.draw("de", split, 6, np.random.default_rng(i)))


def test_a_language_with_too_few_reserved_speakers_borrows_from_every_language():
    pool = reserved_pool({"kk": 2, "de": 10, "fr": 10}, min_speakers=3)
    languages = Counter(c.language for i in range(100) for c in pool.draw("kk", "train", 6, np.random.default_rng(i)))
    assert languages["de"] > 0
    assert languages["fr"] > 0
    assert languages["kk"] > 0


def test_japanese_and_mandarin_take_their_own_language_half_the_time():
    assert {"ja", "zh", "zh-TW"} == OPEN_LANGUAGES
    pool = reserved_pool({"ja": 20, "de": 200}, min_speakers=3)
    same = [all(c.language == "ja" for c in pool.draw("ja", "train", 6, np.random.default_rng(i))) for i in range(1000)]
    assert sum(same) / len(same) == pytest.approx(0.5, abs=0.05)


def test_a_talker_pool_refuses_a_clip_that_is_not_a_reserved_whole_file():
    with pytest.raises(ValueError, match="reserved whole-file"):
        TalkerPool([dataclasses.replace(talker("a"), reserved=False)], 3)
    with pytest.raises(ValueError, match="reserved whole-file"):
        TalkerPool([dataclasses.replace(talker("a"), offset_s=1.0)], 3)


def test_a_pool_with_fewer_speakers_than_talkers_repeats_a_clip_rather_than_failing():
    picked = reserved_pool({"de": 2}, min_speakers=1).draw("de", "train", 6, np.random.default_rng(0))
    assert len(picked) == 6


def test_a_talker_offset_stays_inside_the_clip_or_is_zero():
    clip = talker("a")  # 4 s
    assert all(0 <= talker_offset(clip, 16000, np.random.default_rng(i)) <= 3 * 16000 for i in range(50))
    assert talker_offset(clip, 5 * 16000, np.random.default_rng(0)) == 0


def tone_file(root, name, freq, amplitude, seconds=1.0):
    path = root / f"{name}.wav"
    t = np.arange(round(seconds * 16000)) / 16000
    sf.write(path, (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), 16000, subtype="FLOAT")
    return f"{name}.wav"


def test_babble_sums_the_talkers_at_equal_energy(tmp_path):
    names = [tone_file(tmp_path, f"t{i}", 300 + 400 * i, a) for i, a in enumerate([0.01, 0.5, 0.1])]
    babble = build_babble(names, [0, 0, 0], 16000, tmp_path)
    spectrum = np.abs(np.fft.rfft(babble))
    peaks = [spectrum[300 + 400 * i] for i in range(3)]
    assert max(peaks) / min(peaks) == pytest.approx(1.0, rel=0.01)
    assert np.sqrt(np.mean(babble**2)) == pytest.approx(1.0, rel=0.01)


def test_babble_wraps_a_short_talker_and_reads_from_the_offset(tmp_path):
    names = [tone_file(tmp_path, "short", 500, 0.2, seconds=0.25), tone_file(tmp_path, "long", 900, 0.2, seconds=2.0)]
    babble = build_babble(names, [0, 8000], 32000, tmp_path)
    assert len(babble) == 32000
    assert babble.dtype == np.float32
    spectrum = np.abs(np.fft.rfft(babble)) / 16000
    assert spectrum[1000] > 0.2
    assert spectrum[1800] > 0.2


def test_babble_refuses_a_silent_talker_and_mismatched_offsets(tmp_path):
    sf.write(tmp_path / "zero.wav", np.zeros(16000, np.float32), 16000, subtype="FLOAT")
    with pytest.raises(DecodeError, match="silence"):
        build_babble(["zero.wav", tone_file(tmp_path, "t", 400, 0.1)], [0, 0], 8000, tmp_path)
    with pytest.raises(ValueError, match="offset"):
        build_babble(["zero.wav"], [], 8000, tmp_path)


def speech_like(seconds: float = 10.0) -> np.ndarray:
    """White noise through three resonances: a stationary signal with a speech-like envelope."""
    x = np.random.default_rng(0).standard_normal(round(seconds * 16000))
    for freq, radius in ((500, 0.95), (1500, 0.9), (2500, 0.85)):
        theta = 2 * np.pi * freq / 16000
        x = signal.lfilter([1.0], [1.0, -2 * radius * np.cos(theta), radius**2], x)
    return x / np.abs(x).max()


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference in dB of the two long-term spectra below 4 kHz, after matching their means."""
    (_, pa), (_, pb) = (signal.welch(x, 16000, nperseg=512) for x in (a, b))
    la, lb = (10 * np.log10(p[1:129]) for p in (pa, pb))
    return float(np.sqrt(np.mean((la - la.mean() - (lb - lb.mean())) ** 2)))


def test_ssn_follows_the_speaker_envelope():
    speech = speech_like()
    lpc = fit_lpc(speech, order=12)
    assert len(lpc) == 13
    assert lpc[0] == 1.0
    ssn = build_ssn(lpc, 160000, np.random.default_rng(1))
    assert spectral_distance(ssn, speech) < 3.0
    assert spectral_distance(np.random.default_rng(2).standard_normal(160000), speech) > 10.0


def test_ssn_is_stable_has_unit_rms_and_follows_the_rng():
    lpc = fit_lpc(speech_like(), order=12)
    assert np.all(np.abs(np.roots(lpc)) < 1)
    a, b = build_ssn(lpc, 8000, np.random.default_rng(3)), build_ssn(lpc, 8000, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert np.sqrt(np.mean(a**2)) == pytest.approx(1.0, rel=1e-3)
    assert not np.array_equal(a, build_ssn(lpc, 8000, np.random.default_rng(4)))


def test_an_all_pole_fit_to_silence_is_a_decode_error():
    with pytest.raises(DecodeError, match="silence"):
        fit_lpc(np.zeros(1000))


def competing_setup(tmp_path):
    """A 4 s foreground tone active from 1 to 3 s, a noise talker file and a RIR tree, all under `tmp_path`."""
    t = np.arange(64000) / 16000
    speech = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    active = np.zeros(64000, bool)
    active[16000:48000] = True
    speech[~active] = 0
    sf.write(tmp_path / "talker.wav", (0.2 * np.random.default_rng(5).standard_normal(64000)).astype(np.float32), 16000)
    make_rir_tree(tmp_path / "rirs")
    return speech, active


def segment(start_n, end_n, offset_n=0, lowpass_hz=3000.0, rir: str | None = "rirs/openslr28/room01_0.wav"):
    return Segment("t", start_n, end_n, rir, lowpass_hz, "talker.wav", offset_n)


def measured_tir(mixed, speech, active, segments):
    track = mixed - speech
    covered = np.zeros(len(speech), bool)
    for s in segments:
        covered[s.start_n : s.end_n] = True
    assert not track[~covered].any()
    return 20 * np.log10(rms(speech[active]) / rms(track[covered]))


@pytest.mark.parametrize("tir_db", [0.0, 7.5, 15.0])
def test_measured_tir_matches_the_drawn_one(tmp_path, tir_db):
    speech, active = competing_setup(tmp_path)
    segments = [segment(8000, 24000), segment(40000, 52000, offset_n=3000, lowpass_hz=1200.0)]
    mixed = mix_competing(speech, active, segments, tir_db, tmp_path)
    assert measured_tir(mixed, speech, active, segments) == pytest.approx(tir_db, abs=0.5)


def high_band_share(x):
    """The share of the energy of the first 2 s that lies above 4.5 kHz."""
    spectrum = np.abs(np.fft.rfft(x[:32000])) ** 2
    return spectrum[np.fft.rfftfreq(32000, 1 / 16000) > 4500].sum() / spectrum.sum()


def test_the_talker_is_low_passed_and_reverberant_and_follows_its_segment(tmp_path):
    speech, active = competing_setup(tmp_path)
    low = mix_competing(speech, active, [segment(0, 32000, lowpass_hz=1000.0)], 5.0, tmp_path) - speech
    high = mix_competing(speech, active, [segment(0, 32000, lowpass_hz=4000.0)], 5.0, tmp_path) - speech
    assert high_band_share(low) < 1e-4
    assert high_band_share(low) < high_band_share(high) / 20
    assert not low[32000:].any()
    dry = sf.read(tmp_path / "talker.wav", dtype="float32")[0][:32000]
    assert not np.allclose(low[:32000] / rms(low[:32000]), dry / rms(dry), atol=0.1)  # the RIR changed it


def test_a_different_offset_is_a_different_part_of_the_talker(tmp_path):
    speech, active = competing_setup(tmp_path)
    a = mix_competing(speech, active, [segment(0, 8000, offset_n=0)], 5.0, tmp_path)
    b = mix_competing(speech, active, [segment(0, 8000, offset_n=20000)], 5.0, tmp_path)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, mix_competing(speech, active, [segment(0, 8000, offset_n=0)], 5.0, tmp_path))


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ([], "at least one segment"),
        ([segment(100, 100)], "empty"),
        ([segment(60000, 70000)], "leaves"),
        ([segment(0, 9000), segment(8000, 12000)], "overlaps"),
        ([segment(0, 9000, rir=None)], "rir_id"),
    ],
)
def test_a_malformed_segment_list_is_refused(tmp_path, segments, message):
    speech, active = competing_setup(tmp_path)
    with pytest.raises(ValueError, match=message):
        mix_competing(speech, active, segments, 5.0, tmp_path)


def test_a_silent_foreground_has_no_tir_to_measure(tmp_path):
    speech, active = competing_setup(tmp_path)
    with pytest.raises(ValueError, match="no speech"):
        mix_competing(np.zeros_like(speech), active, [segment(0, 8000)], 5.0, tmp_path)


def test_a_baseline_lowers_the_competing_talker_when_part_of_the_clip_is_its_own_noise(tmp_path):
    speech, active = competing_setup(tmp_path)
    segments = [segment(8000, 24000)]
    plain = mix_competing(speech, active, segments, 10.0, tmp_path)
    based = mix_competing(speech, active, segments, 10.0, tmp_path, baseline_snr_db=0.0)
    assert measured_tir(plain, speech, active, segments) == pytest.approx(10.0, abs=0.5)
    assert measured_tir(based, speech, active, segments) == pytest.approx(10.0 + 3.01, abs=0.5)
