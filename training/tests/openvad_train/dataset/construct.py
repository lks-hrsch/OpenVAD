# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
import hashlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import soundfile as sf
from scipy import signal

from openvad_train.dataset import codecs, construct
from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.channel import RirPool
from openvad_train.dataset.codecs import DecodeError, MissingCodec
from openvad_train.dataset.config import SLICE, BuildConfig, dataset_root, work_dir
from openvad_train.dataset.construct import OutOfSpace, PlanMismatch, construct_sample, run_construct, write_shard
from openvad_train.dataset.cutting import Chunk
from openvad_train.dataset.labels import apply_gap_rule, rasterise
from openvad_train.dataset.mix import dither, loop_slice, place, wada_snr
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.plan import PLAN_SCHEMA, draw_sample
from openvad_train.dataset.schema import Clip, Timing
from openvad_train.fetch.common import cache_dir, data_dir, take_share_lock, write_manifest
from rir_support import make_rir_tree, synthetic_rir


GB = 1_000_000_000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
CHAIN_OFF = {"room": False, "handset": None, "agc": None, "clip_drive_db": None, "suppression": None, "transients": []}
CLEAN = {"condition": "clean", "snr_db": None, "noise_source_ids": [], "noise_offsets": [], "dither_dbfs": -80.0}
NO_CODEC = {"codec": "none", "bitrate": None, "band": "wide"}
G722 = {"codec": "g722", "bitrate": 64.0, "band": "wide"}
NOISE_FILE = "datasets/raw/demand/OOFFICE/ch01.wav"
NOISE_N = 400_000


@pytest.fixture(autouse=True)
def raw(tmp_path, monkeypatch) -> Path:
    """The data root of every test here, so no test can reach the real data directory by forgetting a fixture."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    noise = tmp_path / NOISE_FILE
    noise.parent.mkdir(parents=True)
    x = (0.1 * np.random.default_rng(7).standard_normal(NOISE_N)).astype(np.float32)
    sf.write(noise, x, 16000, subtype="PCM_16")
    return tmp_path


def noisy(snr_db: float = 5.0, offset_n: int = 1000) -> dict:
    return {
        "condition": "office",
        "snr_db": snr_db,
        "noise_source_ids": [NOISE_FILE],
        "noise_offsets": [offset_n],
        "dither_dbfs": None,
    }


def speech_file(x: np.ndarray, name: str = "c0") -> str:
    path = data_dir() / "datasets/raw/swc" / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, 16000, subtype="PCM_16")
    return path.relative_to(data_dir()).as_posix()


def tone(n: int, amplitude: float = 0.3) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(n) / 16000)).astype(np.float32)


def make_row(
    x: np.ndarray | None = None,
    speech=((500, 2500),),
    mask=(),
    name="c0",
    duration_s=None,
    **overrides,
) -> dict:
    """A plan row drawn by the plan's own code, with the chain's probabilistic legs off and `overrides` forced."""
    x = tone(3 * 16000) if x is None else x
    seconds = len(x) / 16000 if duration_s is None else duration_s
    clip = Clip(name, "swc", "de", "s", "train", speech_file(x, name), 0.0, seconds, "CC0", None)
    timing = Timing(name, "word", [tuple(s) for s in speech], [tuple(m) for m in mask], None, "test")
    return draw_sample(clip, timing, SLICE) | CHAIN_OFF | NO_CODEC | CLEAN | overrides


@pytest.fixture(params=["whole_file", "seek"])
def route(request, monkeypatch) -> None:
    """Both routes through `decode_16k`: a limit of 0 sends every file down the seek route, as real SWC articles go."""
    if request.param == "seek":
        monkeypatch.setattr(codecs, "_LONG_FILE_S", 0.0)


def stored(flac: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(flac), dtype="float32")
    assert rate == 16000
    return audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def db(x: float) -> float:
    return 20 * float(np.log10(x))


ROW_KINDS = {"clean": CLEAN | NO_CODEC, "clean_g722": CLEAN | G722, "noisy": NO_CODEC, "noisy_g722": G722}


@pytest.fixture(params=sorted(ROW_KINDS))
def row(request) -> dict:
    return make_row(**ROW_KINDS[request.param], **(noisy() if "noisy" in request.param else {}))


def test_same_row_same_bytes(row, raw):
    assert construct_sample(row, raw)[0] == construct_sample(row, raw)[0]


def test_labels_match_audio_length(row, raw):
    flac, y = construct_sample(row, raw)
    audio, sr = sf.read(io.BytesIO(flac))
    assert sr == 16000
    assert len(y) == round(len(audio) / 160)
    assert len(audio) == row["total_n"]
    assert y.dtype == np.uint8


def test_stored_flac_is_16_bit_mono(row, raw):
    info = sf.info(io.BytesIO(construct_sample(row, raw)[0]))
    assert (info.format, info.subtype, info.channels) == ("FLAC", "PCM_16", 1)


def impulse_row(at_s: float = 1.5, **overrides) -> dict:
    x = np.zeros(3 * 16000, np.float32)
    x[round(at_s * 16000)] = 0.8
    return make_row(x, speech=[(round(at_s * 1000) - 100, round(at_s * 1000) + 100)], **overrides)


@pytest.mark.parametrize("codec", [NO_CODEC, G722], ids=["none", "g722"])
def test_impulse_survives_padding_and_codec(raw, codec):
    flac, y = construct_sample(impulse_row(**codec), raw)
    audio = stored(flac)
    # The label block is 200 ms long around the impulse, so its middle is where the impulse must land.
    assert abs(np.argmax(np.abs(audio)) // 160 - (np.argmax(y == 1) + 10)) <= 1


@pytest.mark.parametrize(("codec", "lag_n"), [(NO_CODEC, 0), (G722, 22)], ids=["none", "g722"])
def test_impulse_lands_at_padding_offset_plus_codec_delay(raw, codec, lag_n):
    row = impulse_row(**codec)
    audio = stored(construct_sample(row, raw)[0])
    assert np.argmax(np.abs(audio)) == row["speech_offset_n"] + round(1.5 * 16000) + lag_n


@pytest.mark.parametrize(("codec", "first_frame_is_speech"), [(NO_CODEC, 1), (G722, 0)], ids=["none", "g722"])
def test_labels_shift_by_padding_offset_then_codec_delay(raw, codec, first_frame_is_speech):
    # At 3004 ms a word covers 6 ms of frame 300; G.722's 1 ms moves it to 3005, a tie, which is non-speech.
    row = make_row(speech=[(1004, 2000)], total_n=6 * 16000, speech_offset_n=2 * 16000, **codec)
    y = construct_sample(row, raw)[1]
    assert y[300] == first_frame_is_speech
    assert y[299] == 0
    assert y[301:399].tolist() == [1] * 98
    assert y[401:].tolist() == [0] * (len(y) - 401)


def chirp(n: int, f1: float = 3500.0) -> np.ndarray:
    t = np.arange(n) / 16000
    return (0.5 * signal.chirp(t, 100, n / 16000, f1)).astype(np.float32)


def need_encoder(codec: str) -> None:
    try:
        codecs.preflight({codecs.encoder_of(codec)})
    except MissingCodec as e:
        pytest.skip(str(e))


STORED = {
    "opus": [("opus", 12.0), ("opus", 32.0)],
    "amr_wb": [("amr_wb", 6.6), ("amr_wb", 23.85)],
    "amr_nb": [("amr_nb", 4.75), ("amr_nb", 12.2)],
}
CODEC_CASES = [pytest.param(c, b, id=f"{c}-{b}") for pairs in STORED.values() for c, b in pairs] + [
    pytest.param("g722", 64.0, id="g722-64.0")
]


def codec_overrides(codec: str, bitrate: float) -> dict:
    return {"codec": codec, "bitrate": bitrate, "band": "narrow_leg" if codec == "amr_nb" else "wide"}


@pytest.mark.parametrize(
    ("codec", "ms"), [("none", 0), ("g722", 1), ("opus", 0), ("amr_wb", 6), ("amr_nb", 9)], ids=lambda v: str(v)
)
def test_the_label_shift_of_each_codec_in_whole_milliseconds(codec, ms):
    # AMR-NB: 5.0 ms for the codec and 4.125 ms for the resampler round trip, which rounds to 9.
    assert construct.codec_delay_ms({"codec": codec}) == ms


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "bitrate"), CODEC_CASES)
def test_each_stored_codec_keeps_length_and_lags_by_its_delay(raw, codec, bitrate):
    need_encoder(codec)
    x = chirp(3 * 16000)
    plain = stored(construct_sample(make_row(x, speech=[(0, 3000)]), raw)[0])
    coded = stored(construct_sample(make_row(x, speech=[(0, 3000)], **codec_overrides(codec, bitrate)), raw)[0])
    assert len(coded) == len(plain)
    lag_ms = (np.argmax(signal.correlate(coded, plain, mode="full", method="fft")) - (len(plain) - 1)) / 16
    assert lag_ms == pytest.approx(codecs.stored_delay_ms(codec), abs=0.5)


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["g722", "opus", "amr_wb"])
def test_an_impulse_survives_padding_and_each_wideband_codec(raw, codec):
    need_encoder(codec)
    bitrate = 64.0 if codec == "g722" else 32.0 if codec == "opus" else 23.85
    flac, y = construct_sample(impulse_row(**codec_overrides(codec, bitrate)), raw)
    assert abs(np.argmax(np.abs(stored(flac))) // 160 - (np.argmax(y == 1) + 10)) <= 1


@needs_ffmpeg
def test_the_amr_nb_leg_shifts_the_labels_by_nine_milliseconds(raw):
    need_encoder("amr_nb")
    row = make_row(chirp(3 * 16000, 3000), speech=[(500, 2500)], total_n=6 * 16000, speech_offset_n=2 * 16000)
    row = row | codec_overrides("amr_nb", 12.2)
    flac, y = construct_sample(row, raw)
    expected = rasterise(apply_gap_rule(row["speech_ms"], row["gap_ms"]), row["mask_ms"], row["total_n"], 2000 + 9)
    assert y.tolist() == expected.tolist()
    audio = stored(flac)
    assert rms(audio[: 2 * 16000]) < 0.01 < rms(audio[2 * 16000 + 4000 : 2 * 16000 + 8000])


CHAIN_LEGS = {
    "handset": {"highpass_hz": 300.0, "lowpass_hz": 3400.0, "peaks": [{"freq_hz": 1000.0, "gain_db": 6.0}] * 2},
    "agc": {"target_dbfs": -20.0, "attack_ms": 10.0, "release_ms": 200.0},
    "clip_drive_db": 12.0,
    "suppression": "wiener",
}


@pytest.mark.parametrize("leg", sorted(CHAIN_LEGS))
def test_each_chain_leg_changes_the_audio_and_leaves_the_labels(raw, leg):
    plain_flac, plain_labels = construct_sample(make_row(**noisy()), raw)
    flac, labels = construct_sample(make_row(**noisy(), **{leg: CHAIN_LEGS[leg]}), raw)
    assert len(stored(flac)) == len(stored(plain_flac))
    assert not np.array_equal(stored(flac), stored(plain_flac))
    assert labels.tolist() == plain_labels.tolist()
    assert construct_sample(make_row(**noisy(), **{leg: CHAIN_LEGS[leg]}), raw)[0] == flac


def test_the_whole_chain_stays_inside_full_scale_and_builds(raw):
    row = make_row(tone(3 * 16000, 0.9), **noisy(0.0), **CHAIN_LEGS)
    audio = stored(construct_sample(row, raw)[0])
    assert np.abs(audio).max() <= 1.0
    assert np.isfinite(audio).all()


def room_row(rir: np.ndarray, **overrides) -> dict:
    path = data_dir() / "datasets/raw/rirs/but/room/rir.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, rir, 16000, subtype="FLOAT")
    return impulse_row(room=True, rir_id=path.relative_to(data_dir()).as_posix(), **overrides)


def test_a_room_does_not_shift_the_onset(raw):
    plain = impulse_row()
    room = room_row(synthetic_rir(predelay_n=800))
    assert np.argmax(np.abs(stored(construct_sample(room, raw)[0]))) == np.argmax(
        np.abs(stored(construct_sample(plain, raw)[0]))
    )


def test_the_reverb_tail_is_non_speech(raw):
    row = room_row(synthetic_rir(n=12000))
    flac, labels = construct_sample(row, raw)
    audio = stored(flac)
    speech_end = row["speech_offset_n"] + round(1.6 * 16000)  # the impulse sits at 1.5 s, its word ends at 1.6 s
    assert np.abs(audio[speech_end : speech_end + 4000]).max() > 1e-3  # the tail is in the audio
    assert (labels[speech_end // 160 + 21 :] == 0).all()  # G = 200 ms closes no gap after the last word
    assert labels.tolist() == construct_sample(impulse_row(), raw)[1].tolist()


@needs_ffmpeg
def test_slice_rows_draw_the_whole_chain_and_every_codec_and_all_construct(raw):
    for codec in ("g722", "opus", "amr_wb", "amr_nb"):
        need_encoder(codec)
    rirs = RirPool(make_rir_tree(raw / "datasets/raw/rirs"))
    path = speech_file(tone(3 * 16000))
    rows = []
    for i in range(80):
        clip = Clip(f"s{i}", "swc", "de", "s", "train", path, 0.0, 3.0, "CC0", None)
        drawn = draw_sample(clip, Timing(f"s{i}", "word", [(500, 2500)], [], None, "test"), SLICE, rirs=rirs)
        # The noise pool is not wired here: the condition is the only thing replaced, and the chain and codec are kept.
        row = drawn | (CLEAN if drawn["condition"] == "clean" else noisy())
        flac, labels = construct_sample(row, raw)
        assert len(stored(flac)) == row["total_n"]
        assert len(labels) == round(row["total_n"] / 160)
        rows.append(row)
    assert {r["room"] for r in rows} == {True, False}
    for leg in ("handset", "agc", "clip_drive_db", "suppression"):
        assert {r[leg] is not None for r in rows} == {True, False}, leg
    assert {r["codec"] for r in rows} == set(SLICE.codec_shares)


def test_a_room_row_without_a_rir_is_refused(raw):
    with pytest.raises(ValueError, match="rir_id"):
        construct_sample(make_row(room=True, rir_id=None), raw)


@pytest.mark.parametrize(("rate", "tolerance_n"), [(16000, 0), (44100, 1)])
def test_a_chunk_takes_its_audio_from_its_offset_in_the_file(raw, route, rate, tolerance_n):
    # Nearly every real row is a chunk of a long file, so `source_offset_n` is what maps its labels to its audio.
    marker_ms = 12_345
    x = np.zeros(20 * rate, np.float32)
    x[round(marker_ms / 1000 * rate)] = 0.8
    path = data_dir() / "datasets/raw/swc/article.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, rate, subtype="FLOAT")
    clip = Clip("article", "swc", "de", "s", "train", path.relative_to(data_dir()).as_posix(), 0.0, 20.0, "CC0", None)
    timing = Timing("article", "word", [(12_300, 12_400)], [], None, "test")
    drawn = draw_sample(clip, timing, SLICE, chunk=Chunk(10_000, 14_000))
    row = drawn | CHAIN_OFF | NO_CODEC | CLEAN | {"speech_offset_n": 2000 * 16}
    assert row["source_offset_n"] == 10_000 * 16
    assert row["speech_ms"] == [(2300, 2400)]

    flac, labels = construct_sample(row, raw)
    # 2000 ms of padding, then the marker 2345 ms into the chunk.
    assert abs(np.argmax(np.abs(stored(flac))) - (2000 + 2345) * 16) <= tolerance_n
    assert np.flatnonzero(labels == 1).tolist() == list(range(430, 440))


def test_masks_shift_with_speech_and_win_over_it(raw):
    row = make_row(speech=[(1000, 2000)], mask=[(1200, 1300)], total_n=6 * 16000, speech_offset_n=2 * 16000)
    y = construct_sample(row, raw)[1]
    assert y[300:320].tolist() == [1] * 20
    assert y[320:330].tolist() == [2] * 10
    assert y[330:400].tolist() == [1] * 70


def test_gap_changes_labels_and_leaves_the_audio_alone(raw):
    speech = [(500, 1000), (1100, 2000)]
    row = make_row(speech=speech, speech_offset_n=0, **noisy())
    flac_g200, y_g200 = construct_sample(row | {"gap_ms": 200}, raw)
    flac_g0, y_g0 = construct_sample(row | {"gap_ms": 0}, raw)
    assert flac_g200 == flac_g0
    assert y_g200[105] == 1
    assert y_g0[105] == 0


def word_bursts(n: int, words: list[tuple[int, int]]) -> np.ndarray:
    x = np.zeros(n, np.float32)
    for start, end in words:
        x[start * 16 : end * 16] = tone((end - start) * 16, amplitude=0.05)  # low enough that no peak scaling applies
    return x


@pytest.mark.parametrize("snr_db", [-5.0, 5.0, 20.0])
def test_snr_is_measured_over_the_words_at_g0(raw, snr_db):
    # The gaps are under G, so measuring over merged speech would read about 3 dB low.
    words = [(500, 600), (790, 890), (1080, 1180)]
    x = word_bursts(3 * 16000, words)
    row = make_row(x, speech=words, **noisy(snr_db))
    audio = stored(construct_sample(row, raw)[0])
    offset = row["speech_offset_n"]
    placed = np.zeros(row["total_n"], np.float32)
    placed[offset : offset + len(x)] = x
    active = np.zeros(row["total_n"], bool)
    for start, end in words:
        active[offset + start * 16 : offset + end * 16] = True
    measured = db(rms(placed[active]) / rms((audio - placed)[active]))
    assert measured == pytest.approx(snr_db, abs=0.5)


def test_loud_mix_is_scaled_down_and_keeps_its_snr(raw):
    x = tone(3 * 16000, amplitude=0.99)
    row = make_row(x, speech=[(0, 3000)], total_n=3 * 16000, speech_offset_n=0, **noisy(-5.0))
    audio = stored(construct_sample(row, raw)[0])
    # Unscaled, the sum would clip at full scale on many samples.
    assert np.sum(np.abs(audio) > 0.999) <= 2
    scale = float(np.dot(audio, x) / np.dot(x, x))
    assert scale < 0.9
    assert db(rms(scale * x) / rms(audio - scale * x)) == pytest.approx(-5.0, abs=0.5)


def test_clean_padding_is_dither_at_the_drawn_level(raw):
    row = make_row(dither_dbfs=-70.0, speech=[(0, 3000)], total_n=6 * 16000, speech_offset_n=0)
    audio = stored(construct_sample(row, raw)[0])
    padding = audio[3 * 16000 + 100 :]
    assert db(rms(padding)) == pytest.approx(-70.0, abs=1.0)
    assert np.count_nonzero(padding) > 0.5 * len(padding)


def test_dither_has_its_own_stream_and_not_the_plans(raw):
    row = make_row(dither_dbfs=-70.0, speech=[(0, 3000)], total_n=6 * 16000, speech_offset_n=0)
    padding = stored(construct_sample(row, raw)[0])[3 * 16000 :]
    own = dither(row["total_n"], -70.0, np.random.default_rng([row["seed"], 1]))[3 * 16000 :]
    np.testing.assert_allclose(padding, own, atol=1 / 16384)


def test_noise_read_matches_loop_slice_inside_and_across_the_end(raw):
    whole = sf.read(raw / NOISE_FILE, dtype="float32")[0]
    for offset_n, n in ((1000, 5000), (NOISE_N - 100, 5000), (5, 3 * NOISE_N)):
        got = construct._read_noise(raw / NOISE_FILE, offset_n, n)
        np.testing.assert_array_equal(got, loop_slice(whole, offset_n, n))


def test_noise_at_another_rate_is_a_broken_setup(raw):
    path = raw / "datasets/raw/demand/OOFFICE/ch02.wav"
    sf.write(path, np.ones(100, np.float32) / 4, 48000)
    with pytest.raises(ValueError, match="48000"):
        construct._read_noise(path, 0, 10)


def test_decoded_length_within_a_frame_rasterises_from_the_real_audio(raw):
    # A Common Voice MP3 without a Xing header overstates its length by 5.6 to 7.5 ms.
    x = tone(3 * 16000 - 120)
    row = make_row(x, speech=[(2000, 3000)], duration_s=3.0, speech_offset_n=0, total_n=3 * 16000)
    flac, y = construct_sample(row, raw)
    assert len(stored(flac)) == 3 * 16000
    assert len(y) == 300
    # 7.5 ms of the last frame hold no speech, so that frame is not labelled speech.
    assert y[298] == 1
    assert y[299] == 0


@pytest.mark.parametrize("short_n", [161, 16000, 3 * 16000])
def test_decoded_length_past_a_frame_is_a_failed_clip(raw, short_n):
    x = tone(3 * 16000 - short_n)
    row = make_row(x, duration_s=3.0, speech=[(0, 100)])
    with pytest.raises(DecodeError, match="source_n"):
        construct_sample(row, raw)


def test_exactly_one_frame_short_is_built(raw):
    row = make_row(tone(3 * 16000 - 160), duration_s=3.0, speech=[(0, 100)])
    assert len(construct_sample(row, raw)[1]) == round(row["total_n"] / 160)


@pytest.mark.parametrize("extra", [CLEAN, noisy()], ids=["clean", "noisy"])
def test_a_clip_silent_under_its_words_is_a_failed_clip(raw, extra):
    row = make_row(np.zeros(3 * 16000, np.float32), **extra)
    with pytest.raises(DecodeError, match="silent"):
        construct_sample(row, raw)


@pytest.mark.parametrize(("level_db", "builds"), [(-70.0, True), (-70.1, False)])
def test_words_under_70_dbfs_are_a_failed_clip(raw, level_db, builds):
    # The nudge keeps float32 rounding from deciding the boundary case; a float file keeps 16-bit steps out of it too.
    x = tone(3 * 16000)
    x = (x * 10 ** (level_db / 20) * (1 + 1e-6) / rms(x)).astype(np.float32)
    row = make_row(x, speech=[(0, 3000)])
    sf.write(raw / row["source_path"], x, 16000, subtype="FLOAT")
    if builds:
        assert len(construct_sample(row, raw)[1]) == round(row["total_n"] / 160)
    else:
        with pytest.raises(DecodeError, match="silent"):
            construct_sample(row, raw)


def test_a_quiet_clip_is_measured_over_its_words_only(raw):
    # Loud audio outside the words must not lift a word at digital-silence level over the limit.
    x = tone(3 * 16000)
    x[1000 * 16 : 2000 * 16] = 1e-5
    with pytest.raises(DecodeError, match="silent"):
        construct_sample(make_row(x, speech=[(1000, 2000)]), raw)


def test_noise_silent_under_the_words_is_a_failed_clip(raw):
    silent = "datasets/raw/demand/OOFFICE/ch03.wav"
    sf.write(raw / silent, np.zeros(NOISE_N, np.float32), 16000, subtype="PCM_16")
    row = make_row(**noisy()) | {"noise_source_ids": [silent]}
    with pytest.raises(DecodeError, match="silent"):
        construct_sample(row, raw)


def test_a_damaged_source_is_a_decode_error(raw):
    row = make_row()
    (raw / row["source_path"]).write_bytes(b"this is not audio" * 100)
    with pytest.raises(DecodeError):
        construct_sample(row, raw)


@pytest.mark.parametrize("condition", ["as_is", "music"])
def test_a_condition_without_a_noise_source_is_refused_instead_of_being_mixed_as_clean(raw, condition):
    with pytest.raises(ValueError, match="no noise for condition"):
        construct_sample(
            make_row(condition=condition, snr_db=5.0, noise_source_ids=[NOISE_FILE], noise_offsets=[0]), raw
        )


def speech_like(n: int, seed: int = 1) -> np.ndarray:
    """Gamma(0.4) amplitudes with random signs at about -20 dBFS: the amplitude model WADA-SNR assumes."""
    g = np.random.default_rng(seed).gamma(0.4, 1, n) * np.random.default_rng(seed + 1).choice([-1, 1], n)
    return np.clip(0.1 * g / np.std(g), -0.9, 0.9).astype(np.float32)


def place_speech(row: dict) -> np.ndarray:
    """The foreground of `row` where construction puts it, before any mix."""
    speech = codecs.decode_16k(str(data_dir() / row["source_path"]), 0.0, row["source_n"] / 16000)
    return place(speech, row["total_n"], row["speech_offset_n"])


def test_a_common_voice_clip_is_mixed_against_its_wada_snr_speech_power_and_records_it(raw):
    speech = speech_like(3 * 16000)
    row = make_row(speech, **noisy(snr_db=0.0), source="common-voice")
    baseline = wada_snr(speech)
    assert -20 <= baseline <= 100
    flac, _, recorded = construct._build(row, raw)
    assert recorded == baseline
    other = make_row(speech, **noisy(snr_db=0.0))
    assert construct._build(other, raw)[2] is None
    # Only the speech share of the clip counts as speech: the noise is that much quieter than for another source.
    ratio = 10 ** (baseline / 10) / (1 + 10 ** (baseline / 10))
    mixed_cv, mixed_other = stored(flac), stored(construct_sample(other, raw)[0])
    noise_cv, noise_other = mixed_cv - place_speech(row), mixed_other - place_speech(other)
    assert rms(noise_cv) / rms(noise_other) == pytest.approx(np.sqrt(ratio), rel=0.02)


def test_every_slice_row_names_a_rir_that_exists(raw):
    # Drawn with a RIR pool, as `run_plan` does: a room row without a `rir_id` would be refused by construct_sample.
    rirs = RirPool(make_rir_tree(raw / "datasets/raw/rirs"))
    rows = []
    for i in range(500):
        clip = Clip(f"c{i}", "swc", "de", "s", "train", "a.ogg", 0.0, 10.0, "CC-BY-SA", None)
        row = draw_sample(clip, Timing(f"c{i}", "word", [(100, 900)], [], None, "test"), SLICE, rirs=rirs)
        assert row["room"] == (row["rir_id"] is not None)
        assert row["rir_id"] is None or (raw / row["rir_id"]).is_file()
        rows.append(row)
    assert any(r["room"] for r in rows)
    # test_slice_rows_draw_the_whole_chain_and_every_codec_and_all_construct builds such rows.


def test_a_clip_drive_of_zero_is_still_applied(raw):
    plain = stored(construct_sample(make_row(), raw)[0])
    clipped = stored(construct_sample(make_row(clip_drive_db=0.0), raw)[0])
    assert not np.array_equal(plain, clipped)


def test_a_noise_row_without_exactly_one_noise_is_refused(raw):
    with pytest.raises(ValueError, match="one noise"):
        construct_sample(make_row(**noisy()) | {"noise_source_ids": [], "noise_offsets": []}, raw)


def test_an_offset_off_the_millisecond_grid_is_refused(raw):
    with pytest.raises(ValueError, match="millisecond"):
        construct_sample(make_row(speech_offset_n=17, total_n=6 * 16000), raw)


# write_shard


@pytest.fixture
def rows() -> list[dict]:
    return [
        make_row(name="c0", **CLEAN, **NO_CODEC),
        make_row(tone(2 * 16000), speech=[(100, 1800)], name="c1", **noisy(), **G722),
        make_row(tone(3 * 16000, 0.1), speech=[(0, 3000)], name="c2", **CLEAN, **G722),
    ]


@pytest.fixture
def twenty_rows(rows) -> list[dict]:
    extra = [make_row(tone(16000), speech=[(100, 900)], name=f"c{i}", **CLEAN, **NO_CODEC) for i in range(3, 20)]
    return rows + extra


def read_fragment(path: Path) -> list[dict]:
    return asyncio.run(read_parquet(path)).to_dicts()


def test_load_audio_reads_tar_member_by_offset(tmp_path, rows, raw):
    write_shard(rows, tmp_path / "00000.tar", tmp_path / "00000.parquet", raw)
    row = asyncio.run(read_parquet(tmp_path / "00000.parquet")).row(0, named=True)
    flac, _ = construct_sample(rows[0], raw)
    assert np.array_equal(load_audio(row, tmp_path), sf.read(io.BytesIO(flac), dtype="float32")[0])


def test_tar_is_uncompressed_and_opens_with_tarfile(tmp_path, rows, raw):
    write_shard(rows, tmp_path / "00000.tar", tmp_path / "00000.parquet", raw)
    with tarfile.open(tmp_path / "00000.tar", "r:") as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == [f"{r['sample_id']}.flac" for r in rows]
    assert all(m.name.endswith(".flac") for m in members)


def test_fragment_points_at_each_member_and_holds_its_labels(tmp_path, rows, raw):
    tar_path = raw / "datasets/processed/x/data/train/00000.tar"
    result = write_shard(rows, tar_path, raw / "work/fragments/train/00000.parquet", raw)
    assert result.n_ok == 3
    assert result.n_failed == 0
    fragment = read_fragment(raw / "work/fragments/train/00000.parquet")
    assert [r["sample_id"] for r in fragment] == [r["sample_id"] for r in rows]
    with tar_path.open("rb") as f:
        for row, source in zip(fragment, rows, strict=True):
            assert row["audio_filepath"] == "datasets/processed/x/data/train/00000.tar"
            assert row["audio_format"] == "flac_tar"
            f.seek(row["byte_offset"])
            flac, labels = construct_sample(source, raw)
            assert f.read(row["byte_length"]) == flac
            assert row["labels"] == labels.tolist()
            assert row["seed"] == source["seed"]
    assert [r["codec_delay_ms"] for r in fragment] == [0, 1, 1]


def test_a_sample_id_with_slashes_and_umlauts_points_right(tmp_path, rows, raw):
    rows[0]["sample_id"] = "swc/german/K\u00f6ln#2@" + "1" * 150
    write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    row = read_fragment(tmp_path / "s.parquet")[0]
    assert len(load_audio(row, tmp_path)) == rows[0]["total_n"]


def test_sidecar_is_the_hash_of_the_tar(tmp_path, rows, raw):
    result = write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert result.sha256 == hashlib.sha256((tmp_path / "s.tar").read_bytes()).hexdigest()
    assert (tmp_path / "s.tar.sha256").read_text() == f"{result.sha256}  s.tar\n"


def test_the_same_rows_give_the_same_shard(tmp_path, rows, raw):
    first = write_shard(rows, tmp_path / "a.tar", tmp_path / "a.parquet", raw)
    second = write_shard(rows, tmp_path / "b.tar", tmp_path / "b.parquet", raw)
    assert first.sha256 == second.sha256


def test_killed_worker_leaves_no_finished_shard(tmp_path, rows, raw, monkeypatch):
    def raise_keyboard_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(construct, "_publish_shard", raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_shard(rows, tmp_path / "00000.tar", tmp_path / "00000.parquet", raw)
    assert not (tmp_path / "00000.tar").exists()
    assert not (tmp_path / "00000.parquet").exists()


def finished_looking(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.is_file() and p.suffix in (".tar", ".parquet", ".sha256"))


def test_a_copy_dying_midway_leaves_nothing_on_the_share(tmp_path, rows, raw, monkeypatch):
    def dies_midway(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:1000])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(construct.shutil, "copyfile", dies_midway)
    with pytest.raises(OSError, match="No space"):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []
    assert not list(tmp_path.glob("*.tmp"))


def test_a_truncated_copy_is_not_replaced_into_place(tmp_path, rows, raw, monkeypatch):
    monkeypatch.setattr(
        construct.shutil, "copyfile", lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()[:1000])
    )
    with pytest.raises(OSError, match="copied"):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


KILLED_WORKER = """
import os, pickle, shutil, signal, sys
from pathlib import Path
from openvad_train.dataset.construct import write_shard

rows, tar, fragment, raw, when = pickle.loads(Path(sys.argv[1]).read_bytes())
die = lambda: os.kill(os.getpid(), signal.SIGKILL)
if when == "copying":
    def copyfile(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:1000])
        die()
    shutil.copyfile = copyfile
else:
    real_replace = os.replace
    def replace(src, dst):
        if Path(dst) == fragment:
            die()
        real_replace(src, dst)
    os.replace = replace
write_shard(rows, tar, fragment, raw)
"""


@pytest.mark.parametrize(("when", "tar_visible"), [("copying", False), ("before_the_fragment", True)])
def test_a_sigkilled_worker_leaves_no_fragment_and_a_rerun_recovers(tmp_path, rows, raw, when, tar_visible):
    share, stage = tmp_path / "share", tmp_path / "stage"
    share.mkdir()
    stage.mkdir()
    tar, fragment = share / "s.tar", share / "s.parquet"
    job = tmp_path / "job.pkl"
    job.write_bytes(pickle.dumps((rows, tar, fragment, raw, when)))
    env = os.environ | {"TMPDIR": str(stage)}
    done = subprocess.run([sys.executable, "-c", KILLED_WORKER, str(job)], env=env, capture_output=True, check=False)
    assert done.returncode == -9, done.stderr.decode()
    assert not fragment.exists()
    assert tar.exists() == tar_visible
    if when == "copying":
        assert (share / "s.tar.tmp").stat().st_size == 1000

    result = write_shard(rows, tar, fragment, raw)
    assert result.n_ok == 3
    assert hashlib.sha256(tar.read_bytes()).hexdigest() == result.sha256
    assert not list(share.glob("*.tmp"))


def test_the_fragment_replaces_last(tmp_path, rows, raw, monkeypatch):
    real, done = os.replace, []

    def recording(src, dst):
        done.append(Path(dst).name)
        real(src, dst)

    monkeypatch.setattr(construct.os, "replace", recording)
    write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert done[-1] == "s.parquet"
    assert {"s.tar", "s.tar.sha256", "s.failures.jsonl"} <= set(done)


def test_a_rebuild_removes_the_old_fragment_before_it_touches_the_tar(tmp_path, rows, raw, monkeypatch):
    write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)

    def dies(*_args):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(construct.shutil, "copyfile", dies)
    with pytest.raises(OSError, match="Input/output"):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert not (tmp_path / "s.parquet").exists()


def corrupt(row: dict, raw: Path) -> None:
    (raw / row["source_path"]).write_bytes(b"this is not audio" * 100)


@pytest.fixture
def rows_with_corrupt_clip(twenty_rows, raw) -> list[dict]:
    corrupt(twenty_rows[0], raw)
    return twenty_rows


def test_corrupt_clip_is_logged_and_skipped(rows_with_corrupt_clip, raw, tmp_path):
    r = write_shard(rows_with_corrupt_clip, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert r.n_failed == 1
    assert "c0" in (tmp_path / "s.failures.jsonl").read_text()
    assert r.n_ok == 19
    assert [row["clip_id"] for row in read_fragment(tmp_path / "s.parquet")] == [f"c{i}" for i in range(1, 20)]


def test_a_failure_line_names_the_sample_and_the_error(rows_with_corrupt_clip, raw, tmp_path):
    write_shard(rows_with_corrupt_clip, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    line = json.loads((tmp_path / "s.failures.jsonl").read_text())
    assert line["sample_id"] == "swc/c0"
    assert line["source_path"] == rows_with_corrupt_clip[0]["source_path"]
    assert "Invalid data" in line["error"]


def test_a_shard_without_failures_has_an_empty_failures_file(tmp_path, rows, raw):
    write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert (tmp_path / "s.failures.jsonl").read_text() == ""


def test_a_shard_with_no_row_built_aborts_without_publishing(rows_with_corrupt_clip, raw, tmp_path):
    with pytest.raises(RuntimeError, match="1 of 1 rows failed"):
        write_shard(rows_with_corrupt_clip[:1], tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


def test_one_row_in_twenty_is_within_the_failed_share(rows_with_corrupt_clip, raw, tmp_path):
    assert write_shard(rows_with_corrupt_clip, tmp_path / "s.tar", tmp_path / "s.parquet", raw).n_failed == 1


def test_one_row_in_nineteen_is_over_the_failed_share(rows_with_corrupt_clip, raw, tmp_path):
    with pytest.raises(RuntimeError, match="1 of 19 rows failed"):
        write_shard(rows_with_corrupt_clip[:19], tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


def test_the_failed_share_abort_names_the_first_error(rows_with_corrupt_clip, raw, tmp_path):
    with pytest.raises(RuntimeError, match="Invalid data"):
        write_shard(rows_with_corrupt_clip[:1], tmp_path / "s.tar", tmp_path / "s.parquet", raw)


def test_a_vanished_source_aborts_instead_of_emptying_the_shard(rows, raw, tmp_path):
    # Every file vanishes at once when the share drops, and a skip would mark the shard finished.
    (raw / rows[0]["source_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


def test_a_source_that_exists_but_cannot_be_read_aborts(rows, raw, tmp_path):
    # stat works and read does not, as on a failing SMB mount.
    source = raw / rows[0]["source_path"]
    source.chmod(0)
    try:
        with pytest.raises(PermissionError):
            write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    finally:
        source.chmod(0o644)
    assert finished_looking(tmp_path) == []


def test_a_transient_decode_failure_is_retried_and_keeps_the_row(rows, raw, tmp_path, monkeypatch):
    real, calls = construct._build, []

    def fails_once(row, root):
        calls.append(row["sample_id"])
        if len(calls) == 1:
            raise DecodeError("a transient failure")
        return real(row, root)

    monkeypatch.setattr(construct, "_build", fails_once)
    result = write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert (result.n_ok, result.n_failed) == (3, 0)
    assert calls[:2] == [rows[0]["sample_id"]] * 2
    assert (tmp_path / "s.failures.jsonl").read_text() == ""


def test_a_row_is_tried_twice_before_it_is_logged(rows_with_corrupt_clip, raw, tmp_path, monkeypatch):
    real, calls = construct._build, []

    def counting(row, root):
        calls.append(row["sample_id"])
        return real(row, root)

    monkeypatch.setattr(construct, "_build", counting)
    write_shard(rows_with_corrupt_clip, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert calls.count(rows_with_corrupt_clip[0]["sample_id"]) == 2
    assert calls.count(rows_with_corrupt_clip[1]["sample_id"]) == 1


@pytest.mark.parametrize(
    "error",
    [MissingCodec("no ffmpeg"), ValueError("bad row"), NotImplementedError("room")],
    ids=lambda e: type(e).__name__,
)
def test_a_broken_setup_aborts_the_shard(rows, raw, tmp_path, monkeypatch, error):
    def broken(*_args):
        raise error

    monkeypatch.setattr(construct, "_build", broken)
    with pytest.raises(type(error)):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


def test_verification_refuses_labels_that_do_not_match_the_audio(rows, raw, tmp_path, monkeypatch):
    real = construct._build

    def one_frame_short(row, root):
        flac, labels, baseline = real(row, root)
        return flac, labels[:-1], baseline

    monkeypatch.setattr(construct, "_build", one_frame_short)
    with pytest.raises(RuntimeError, match="labels"):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


def test_the_local_stage_is_removed(rows, raw, tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(construct.tempfile, "tempdir", str(stage))
    write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert list(stage.iterdir()) == []


# run_construct


@pytest.fixture(autouse=True)
def free_space(monkeypatch) -> list[int]:
    """Bytes free on the share, one per check, the last repeating; the test machine's real disk must not matter."""
    free = [10_000 * GB]
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: SimpleNamespace(free=free[0] if len(free) == 1 else free.pop(0))
    )
    return free


def write_plan(cfg: BuildConfig, rows: list[dict]) -> None:
    out = work_dir(cfg) / "plan"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({out / "plan.parquet": pl.DataFrame(rows, schema=PLAN_SCHEMA).lazy()}))
    write_manifest(out, {"stage": "plan", "plan_version": cfg.plan_version, "global_seed": cfg.global_seed})


@pytest.fixture
def planned_rows(rows) -> list[dict]:
    as_is = make_row(name="as_is") | {"treatment": "as_is", "shard": None, "condition": "as_is"}
    shards = ["train/00000", "train/00000", "validation/00000"]
    return [row | {"shard": shard} for row, shard in zip(rows, shards, strict=True)] + [as_is]


@pytest.fixture
def cfg(planned_rows) -> BuildConfig:
    built = dataclasses.replace(SLICE, name="test-build")
    write_plan(built, planned_rows)
    return built


def shard_files(cfg: BuildConfig) -> list[Path]:
    return sorted((dataset_root(cfg) / "data").rglob("*.tar*")) + sorted(
        (work_dir(cfg) / "fragments").rglob("*.parquet")
    )


def shard_mtimes(cfg: BuildConfig) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in shard_files(cfg)}


def test_run_builds_every_augmented_shard_and_load_audio_reads_them(cfg, raw):
    run_construct(cfg, workers=1)
    names = [p.relative_to(raw).as_posix() for p in shard_files(cfg)]
    assert names == [
        "datasets/processed/test-build/data/train/00000.tar",
        "datasets/processed/test-build/data/train/00000.tar.sha256",
        "datasets/processed/test-build/data/validation/00000.tar",
        "datasets/processed/test-build/data/validation/00000.tar.sha256",
        "datasets/processed/work/test-build/fragments/train/00000.parquet",
        "datasets/processed/work/test-build/fragments/validation/00000.parquet",
    ]
    fragment = read_fragment(work_dir(cfg) / "fragments/train/00000.parquet")
    assert [r["clip_id"] for r in fragment] == ["c0", "c1"]
    for row in fragment:
        assert len(load_audio(row)) == row["total_n"]
        assert len(row["labels"]) == round(row["total_n"] / 160)


def test_a_run_gives_the_bytes_of_construct_sample_after_the_plan_round_trip(cfg, rows, raw):
    run_construct(cfg, workers=1)
    row = read_fragment(work_dir(cfg) / "fragments/validation/00000.parquet")[0]
    flac, labels = construct_sample(rows[2], raw)
    with (raw / row["audio_filepath"]).open("rb") as f:
        f.seek(row["byte_offset"])
        assert f.read(row["byte_length"]) == flac
    assert row["labels"] == labels.tolist()


def test_rerun_skips_finished_shards(cfg):
    run_construct(cfg, workers=1)
    before = shard_mtimes(cfg)
    run_construct(cfg, workers=1)
    assert shard_mtimes(cfg) == before


def test_rerun_rebuilds_a_shard_without_its_fragment_and_leaves_the_others(cfg):
    run_construct(cfg, workers=1)
    (work_dir(cfg) / "fragments/train/00000.parquet").unlink()
    before = shard_mtimes(cfg)
    run_construct(cfg, workers=1)
    after = shard_mtimes(cfg)
    train_tar = dataset_root(cfg) / "data/train/00000.tar"
    validation_tar = dataset_root(cfg) / "data/validation/00000.tar"
    assert after[train_tar] != before[train_tar]
    assert after[validation_tar] == before[validation_tar]


@pytest.mark.parametrize("gone", ["data/train/00000.tar", "data/train/00000.tar.sha256"])
def test_a_shard_missing_its_tar_or_sidecar_is_not_finished(cfg, gone):
    run_construct(cfg, workers=1)
    (dataset_root(cfg) / gone).unlink()
    run_construct(cfg, workers=1)
    assert (dataset_root(cfg) / gone).exists()


def test_stops_when_space_is_short(cfg, free_space):
    free_space[0] = 49 * GB
    with pytest.raises(OutOfSpace):
        run_construct(cfg)
    assert shard_files(cfg) == []


def test_a_rerun_after_space_ran_short_continues(cfg, free_space):
    free_space[0] = 49 * GB
    with pytest.raises(OutOfSpace):
        run_construct(cfg)
    free_space[0] = 100 * GB
    run_construct(cfg, workers=1)
    assert len(shard_files(cfg)) == 6


@pytest.fixture
def in_process_pool(monkeypatch) -> list[str]:
    """Threads instead of processes, so a test can replace `write_shard`; returns the shards in submission order."""
    monkeypatch.setattr(construct, "ProcessPoolExecutor", lambda *_args, **_kwargs: ThreadPoolExecutor(1))
    submitted: list[str] = []

    def record(rows, tar_path, fragment_path, _raw):
        submitted.append(rows[0]["shard"])
        sidecar, failures = tar_path.with_name(tar_path.name + ".sha256"), fragment_path.with_suffix(".failures.jsonl")
        for path in (tar_path, sidecar, failures, fragment_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        return construct.ShardResult(len(rows), 0, "")

    monkeypatch.setattr(construct, "write_shard", record)
    return submitted


def many_shards(cfg, make, n: int = 5) -> None:
    # Ranks fall as shard numbers rise here, so order by number would differ from order by rank.
    rows = [make(name=f"m{i}", **CLEAN, **NO_CODEC) | {"rank": 1000 - i, "shard": f"train/{i:05d}"} for i in range(n)]
    write_plan(cfg, sorted(rows, key=lambda r: r["rank"]))


def test_shards_are_processed_in_plan_order_which_is_rank_order(cfg, in_process_pool):
    many_shards(cfg, make_row)
    run_construct(cfg, workers=1)
    assert in_process_pool == [f"train/{i:05d}" for i in range(4, -1, -1)]


def test_space_running_short_midway_leaves_a_clean_prefix(cfg, free_space, in_process_pool):
    many_shards(cfg, make_row)
    free_space[:] = [100 * GB, 100 * GB, 1 * GB]
    with pytest.raises(OutOfSpace):
        run_construct(cfg, workers=1)
    assert in_process_pool == ["train/00004", "train/00003"]


def test_the_estimate_counts_against_the_margin(cfg, free_space):
    # 63 MB per hour for under a minute of audio per shard is far under 10 MB, so only the margin decides here.
    free_space[0] = 50 * GB - 1
    with pytest.raises(OutOfSpace, match="50 GB"):
        run_construct(cfg, workers=1)
    free_space[0] = 50 * GB + 10_000_000
    run_construct(cfg, workers=1)
    assert len(shard_files(cfg)) == 6


def test_a_failing_shard_stops_new_work_and_is_reported(cfg, monkeypatch, in_process_pool):
    many_shards(cfg, make_row)

    def fails(rows, *_args):
        in_process_pool.append(rows[0]["shard"])
        raise MissingCodec("no ffmpeg")

    monkeypatch.setattr(construct, "write_shard", fails)
    with pytest.raises(MissingCodec):
        run_construct(cfg, workers=1)
    assert len(in_process_pool) <= 2


def test_a_failed_shard_leaves_a_hole_and_a_rerun_fills_only_it(cfg, monkeypatch, in_process_pool):
    many_shards(cfg, make_row)
    record = construct.write_shard

    def first_fails(rows, *args):
        if rows[0]["shard"] == "train/00004":
            raise MissingCodec("no ffmpeg")
        return record(rows, *args)

    monkeypatch.setattr(construct, "write_shard", first_fails)
    with pytest.raises(MissingCodec):
        run_construct(cfg, workers=2)
    assert "train/00003" in in_process_pool
    assert "train/00004" not in in_process_pool

    built = list(in_process_pool)
    in_process_pool.clear()
    monkeypatch.setattr(construct, "write_shard", record)
    run_construct(cfg, workers=2)
    assert in_process_pool[0] == "train/00004"
    assert not set(in_process_pool) & set(built)
    assert sorted(built + in_process_pool) == [f"train/{i:05d}" for i in range(5)]


def test_a_row_with_a_condition_nothing_builds_aborts_through_the_workers(cfg, planned_rows):
    write_plan(cfg, [planned_rows[0] | {"condition": "music", "snr_db": 5.0}, *planned_rows[1:]])
    with pytest.raises(ValueError, match="no noise for condition"):
        run_construct(cfg, workers=1)
    assert not (work_dir(cfg) / "fragments/train/00000.parquet").exists()


def test_preflight_runs_once_before_any_worker_with_the_encoders_of_the_plan(cfg, monkeypatch):
    seen: list[set[str]] = []
    real = construct.preflight

    def recording(required):
        seen.append(required)
        return real(required)

    monkeypatch.setattr(construct, "preflight", recording)
    run_construct(cfg, workers=1)
    assert seen == [{"g722"}]


def test_a_missing_encoder_aborts_before_any_worker(cfg, monkeypatch):
    def missing(_required):
        raise MissingCodec("no g722")

    def no_pool(*_args, **_kwargs):
        pytest.fail("a worker pool started")

    monkeypatch.setattr(construct, "preflight", missing)
    monkeypatch.setattr(construct, "ProcessPoolExecutor", no_pool)
    with pytest.raises(MissingCodec):
        run_construct(cfg)


def test_second_process_is_refused(cfg):
    lock = take_share_lock_for(cfg)
    with pytest.raises(SystemExit, match="held by host"):
        run_construct(cfg)
    assert lock.exists()
    assert shard_files(cfg) == []


def take_share_lock_for(cfg: BuildConfig) -> Path:
    cache_dir().mkdir(parents=True, exist_ok=True)
    return take_share_lock("construct-" + cfg.name)


def test_the_lock_is_released_after_a_run_and_after_a_failure(cfg, free_space):
    lock = cache_dir() / f"construct-{cfg.name}.fetching"
    run_construct(cfg, workers=1)
    assert not lock.exists()
    (work_dir(cfg) / "fragments/train/00000.parquet").unlink()
    free_space[0] = 0
    with pytest.raises(OutOfSpace):
        run_construct(cfg, workers=1)
    assert not lock.exists()


@pytest.fixture
def cfg_built(cfg) -> BuildConfig:
    run_construct(cfg, workers=1)
    return cfg


@pytest.fixture
def changed_plan(cfg_built) -> BuildConfig:
    return dataclasses.replace(cfg_built, plan_version="v2")


def test_plan_version_mismatch_refuses(cfg_built, changed_plan):
    before = shard_mtimes(cfg_built)
    with pytest.raises(PlanMismatch):
        run_construct(changed_plan)
    assert shard_mtimes(cfg_built) == before


def test_a_new_plan_under_the_same_version_refuses(cfg_built, planned_rows):
    write_plan(cfg_built, planned_rows[:2])
    with pytest.raises(PlanMismatch, match="plan_sha256"):
        run_construct(cfg_built)


def test_another_ffmpeg_refuses_mixing_shards(cfg_built, monkeypatch):
    monkeypatch.setattr(construct, "preflight", lambda _required: {"version": "0.0", "configuration": []})
    with pytest.raises(PlanMismatch, match="ffmpeg"):
        run_construct(cfg_built)


def test_a_deleted_build_manifest_does_not_let_a_new_plan_keep_old_shards(cfg_built, planned_rows):
    (work_dir(cfg_built) / "fragments/manifest.json").unlink()
    write_plan(cfg_built, planned_rows[:2])
    before = shard_mtimes(cfg_built)
    with pytest.raises(PlanMismatch, match="manifest"):
        run_construct(cfg_built)
    assert shard_mtimes(cfg_built) == before
    assert not (work_dir(cfg_built) / "fragments/manifest.json").exists()


def test_a_run_without_a_plan_says_so(raw):
    with pytest.raises(FileNotFoundError, match="plan"):
        run_construct(dataclasses.replace(SLICE, name="no-plan"))


def test_unreadable_sources_abort_the_run_and_a_rerun_repairs_it(cfg, planned_rows, raw):
    sources = [raw / row["source_path"] for row in planned_rows[:3]]
    for source in sources:
        source.chmod(0)
    try:
        with pytest.raises(PermissionError):
            run_construct(cfg, workers=1)
    finally:
        for source in sources:
            source.chmod(0o644)
    assert not list((work_dir(cfg) / "fragments").rglob("*.parquet"))
    run_construct(cfg, workers=1)
    assert len(read_fragment(work_dir(cfg) / "fragments/train/00000.parquet")) == 2


def test_a_shard_over_the_failed_share_aborts_the_run_and_a_rerun_rebuilds_it(cfg, planned_rows, raw):
    good = (raw / planned_rows[0]["source_path"]).read_bytes()
    corrupt(planned_rows[0], raw)
    with pytest.raises(RuntimeError, match="1 of 2 rows failed"):
        run_construct(cfg, workers=1)
    assert not (work_dir(cfg) / "fragments/train/00000.parquet").exists()
    (raw / planned_rows[0]["source_path"]).write_bytes(good)
    run_construct(cfg, workers=1)
    assert len(read_fragment(work_dir(cfg) / "fragments/train/00000.parquet")) == 2


def test_the_failed_total_reaches_the_manifest_and_a_warning(cfg, twenty_rows, raw, caplog):
    corrupt(twenty_rows[0], raw)
    write_plan(cfg, [row | {"shard": "train/00000"} for row in twenty_rows])
    run_construct(cfg, workers=1)
    assert json.loads((work_dir(cfg) / "fragments/manifest.json").read_text())["n_failed"] == 1
    assert "1 rows failed" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)
    # A re-run builds nothing, and the total still counts the shards it skipped.
    caplog.clear()
    run_construct(cfg, workers=1)
    assert json.loads((work_dir(cfg) / "fragments/manifest.json").read_text())["n_failed"] == 1
    assert "1 rows failed" in caplog.text


def test_a_run_without_failures_records_zero_and_stays_quiet(cfg, caplog):
    run_construct(cfg, workers=1)
    assert json.loads((work_dir(cfg) / "fragments/manifest.json").read_text())["n_failed"] == 0
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


def donor(name: str, seconds: float = 4.0, seed: int = 0, silent: bool = False) -> str:
    """A reserved talker's file under raw, as a path relative to the data root."""
    path = data_dir() / "datasets/raw/mls" / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = round(seconds * 16000)
    sf.write(path, np.zeros(n, np.float32) if silent else speech_like(n, seed), 16000, subtype="PCM_16")
    return path.relative_to(data_dir()).as_posix()


def snr_of(flac: bytes, row: dict) -> float:
    """Speech over (mixed - speech) in dB over the row's word union, with the foreground put back where it was."""
    speech, mixed = place_speech(row), stored(flac)
    active = np.zeros(row["total_n"], bool)
    active[row["speech_offset_n"] + 500 * 16 : row["speech_offset_n"] + 2500 * 16] = True
    return db(rms(speech[active]) / rms((mixed - speech)[active]))


def test_city_noise_takes_the_channel_named_in_its_id(raw):
    wham = raw / "datasets/raw/wham/wham_noise/tt/a.wav"
    wham.parent.mkdir(parents=True)
    two = 0.1 * np.random.default_rng(3).standard_normal((NOISE_N, 2)).astype(np.float32)
    sf.write(wham, two, 16000, subtype="FLOAT")
    results = {}
    for channel in (0, 1):
        row = make_row(**noisy(5.0) | {"condition": "city", "noise_source_ids": [f"{wham.relative_to(raw)}#{channel}"]})
        flac, _ = construct_sample(row, raw)
        results[channel] = stored(flac) - place_speech(row)
        assert snr_of(flac, row) == pytest.approx(5.0, abs=0.3)
    # The noise comes from the named channel: it correlates with that channel and not with the other.
    n = len(results[0]) - 1000
    for channel in (0, 1):
        own = np.corrcoef(results[channel][1000:], two[1000 + 1000 : 1000 + 1000 + n, channel])[0, 1]
        other = np.corrcoef(results[channel][1000:], two[1000 + 1000 : 1000 + 1000 + n, 1 - channel])[0, 1]
        assert abs(own) > 0.99 > abs(other) + 0.9


def test_a_two_channel_noise_file_needs_a_channel_and_a_mono_one_must_not_have_one(raw):
    wham = raw / "datasets/raw/wham/a.wav"
    wham.parent.mkdir(parents=True)
    sf.write(wham, np.ones((16000, 2), np.float32) * 0.1, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="2 channels"):
        construct_sample(
            make_row(**noisy()) | {"condition": "city", "noise_source_ids": [str(wham.relative_to(raw))]}, raw
        )
    with pytest.raises(ValueError, match="1 channels"):
        construct_sample(make_row(**noisy()) | {"noise_source_ids": [NOISE_FILE + "#0"]}, raw)


def babble_row(n_talkers: int = 6, **extra) -> dict:
    ids = [donor(f"t{i}", seed=i) for i in range(n_talkers)]
    return make_row(
        **noisy(5.0) | {"condition": "babble", "noise_source_ids": ids, "noise_offsets": [0] * n_talkers} | extra
    )


def test_babble_is_six_talkers_mixed_at_the_drawn_snr(raw):
    row = babble_row()
    flac, _ = construct_sample(row, raw)
    assert snr_of(flac, row) == pytest.approx(5.0, abs=0.3)
    assert construct_sample(row, raw)[0] == flac


def test_babble_needs_several_talkers_and_a_silent_talker_is_a_decode_error(raw):
    with pytest.raises(ValueError, match="several talkers"):
        construct_sample(babble_row(1), raw)
    row = babble_row(3)
    row["noise_source_ids"][1] = donor("zero", silent=True)
    with pytest.raises(DecodeError, match="silence"):
        construct_sample(row, raw)


def test_ssn_follows_its_own_stream_at_the_drawn_snr(raw):
    row = make_row(**noisy(10.0) | {"condition": "ssn", "noise_source_ids": [donor("s")], "noise_offsets": [0]})
    flac, _ = construct_sample(row, raw)
    assert snr_of(flac, row) == pytest.approx(10.0, abs=0.3)
    assert construct_sample(row, raw)[0] == flac
    assert construct_sample(row | {"seed": row["seed"] + 1}, raw)[0] != flac
    with pytest.raises(ValueError, match="one talker"):
        construct_sample(row | {"noise_source_ids": [donor("s"), donor("u")], "noise_offsets": [0, 0]}, raw)


def competing_row(**extra) -> dict:
    rir = "datasets/raw/rirs/openslr28/room01_0.wav"
    make_rir_tree(data_dir() / "datasets/raw/rirs")
    segments = [
        {
            "clip_id": "mls/a",
            "start_n": 0,
            "end_n": 20000,
            "rir_id": rir,
            "lowpass_hz": 3000.0,
            "source_path": donor("a"),
            "source_offset_n": 0,
        },
        {
            "clip_id": "mls/b",
            "start_n": 30000,
            "end_n": 40000,
            "rir_id": rir,
            "lowpass_hz": 1500.0,
            "source_path": donor("b", seed=2),
            "source_offset_n": 2000,
        },
    ]
    return make_row(
        **CLEAN
        | {"condition": "competing", "segments": segments, "tir_db": 8.0, "dither_dbfs": -80.0, "speech_offset_n": 8000}
        | extra
    )


def test_a_competing_talker_is_added_at_the_drawn_tir_and_the_labels_stay_those_of_the_foreground(raw):
    row = competing_row()
    flac, labels = construct_sample(row, raw)
    speech, mixed = place_speech(row), stored(flac)
    track = mixed - speech
    covered = np.zeros(row["total_n"], bool)
    for s in row["segments"]:
        covered[s["start_n"] : s["end_n"]] = True
    active = np.zeros(row["total_n"], bool)
    active[row["speech_offset_n"] + 500 * 16 : row["speech_offset_n"] + 2500 * 16] = True
    assert db(rms(speech[active]) / rms(track[covered])) == pytest.approx(8.0, abs=0.5)
    assert np.abs(track[~covered]).max() < 1e-3  # padding outside the segments is the dither
    assert rms(track[~covered]) == pytest.approx(10 ** (-80 / 20), rel=0.1)
    assert labels.tolist() == construct_sample(make_row(speech_offset_n=8000), raw)[1].tolist()
    assert construct_sample(row, raw)[0] == flac


def test_a_competing_row_needs_its_rirs_and_a_tir(raw):
    row = competing_row()
    with pytest.raises(ValueError, match="rir_id"):
        construct_sample(row | {"segments": [row["segments"][0] | {"rir_id": None}]}, raw)
    with pytest.raises(ValueError, match="at least one segment"):
        construct_sample(row | {"segments": []}, raw)


def test_a_transient_lands_at_its_offset_and_peak_ratio_on_the_mix(raw):
    event = {"kind": "click", "offset_n": 100, "peak_db": -6.0}
    plain, with_click = make_row(), make_row(transients=[event])
    a, b = stored(construct_sample(plain, raw)[0]), stored(construct_sample(with_click, raw)[0])
    added = b - a
    assert not added[:100].any()
    assert not added[100 + 81 :].any()
    assert np.abs(added).max() == pytest.approx(np.abs(a).max() * 10 ** (-6 / 20), rel=0.02)
    assert construct_sample(with_click, raw)[1].tolist() == construct_sample(plain, raw)[1].tolist()
    assert construct_sample(with_click, raw)[0] == construct_sample(with_click, raw)[0]


def test_no_constructed_sample_passes_full_scale_with_a_transient_added_at_the_loudest_peak(raw, monkeypatch):
    written = []
    real_write = sf.write
    monkeypatch.setattr(
        construct.sf, "write", lambda f, x, *a, **k: (written.append(np.array(x)), real_write(f, x, *a, **k))[1]
    )
    loud = make_row(tone(3 * 16000, 0.95))
    loud["transients"] = [{"kind": "dtmf", "offset_n": loud["speech_offset_n"] + 24000, "peak_db": 0.0}]
    written.clear()  # the speech file of the row
    flac, _ = construct_sample(loud, raw)
    assert np.abs(written[0]).max() == pytest.approx(1.0, abs=1e-6)  # scaled down by protect_peak, not clipped
    audio = stored(flac)
    assert np.abs(audio).max() <= 1.0
    assert np.count_nonzero(np.abs(audio) >= 0.9999) < 5
    # The sum would have reached 1.9: the peak was divided after the click was added.
    assert np.abs(audio[: loud["speech_offset_n"] + 16000]).max() < 0.6


def test_the_baseline_of_a_common_voice_clip_reaches_its_fragment(raw, tmp_path):
    cv = make_row(speech_like(3 * 16000), **noisy(0.0), source="common-voice")
    other = make_row(speech_like(3 * 16000), **noisy(0.0), name="c1")
    write_shard([cv, other], tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    got = {r["clip_id"]: r["baseline_snr_db"] for r in read_fragment(tmp_path / "s.parquet")}
    assert got == {"c0": wada_snr(speech_like(3 * 16000)), "c1": None}
