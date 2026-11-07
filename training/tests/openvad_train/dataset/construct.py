# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from openvad_train.dataset import construct
from openvad_train.dataset.codecs import DecodeError
from openvad_train.dataset.config import SLICE
from openvad_train.dataset.construct import construct_sample
from openvad_train.dataset.mix import loop_slice
from openvad_train.dataset.plan import draw_sample
from openvad_train.dataset.schema import Clip, Timing
from openvad_train.fetch.common import data_dir


CHAIN_OFF = {"room": False, "handset": None, "agc": None, "clip_drive_db": None, "suppression": None, "transients": []}
CLEAN = {"condition": "clean", "snr_db": None, "noise_source_ids": [], "noise_offsets": [], "dither_dbfs": -80.0}
NO_CODEC = {"codec": "none", "bitrate": None, "band": "wide"}
G722 = {"codec": "g722", "bitrate": 64.0, "band": "wide"}
NOISE_FILE = "datasets/raw/demand/OOFFICE/ch01.wav"
NOISE_N = 400_000


@pytest.fixture(autouse=True)
def raw(tmp_path, monkeypatch) -> Path:
    """The data root of every test here, so no test can reach the real data directory by forgetting a fixture."""
    root = tmp_path / "data"
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(root))
    noise = root / NOISE_FILE
    noise.parent.mkdir(parents=True)
    x = (0.1 * np.random.default_rng(7).standard_normal(NOISE_N)).astype(np.float32)
    sf.write(noise, x, 16000, subtype="PCM_16")
    return root


def noisy(snr_db: float = 5.0, offset_n: int = 1000) -> dict:
    return {
        "condition": "office",
        "snr_db": snr_db,
        "noise_source_ids": [NOISE_FILE],
        "noise_offsets": [offset_n],
        "dither_dbfs": None,
    }


def speech_file(x: np.ndarray, name: str = "c0") -> str:
    path = data_dir() / "datasets/raw/common-voice" / f"{name}.wav"
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
    clip = Clip(name, "common-voice", "de", "s", "train", speech_file(x, name), 0.0, seconds, "CC0", None)
    timing = Timing(name, "word", [tuple(s) for s in speech], [tuple(m) for m in mask], None, "test")
    return draw_sample(clip, timing, SLICE) | CHAIN_OFF | NO_CODEC | CLEAN | overrides


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("room", True),
        ("handset", {"highpass_hz": 100.0, "lowpass_hz": 3400.0, "peaks": []}),
        ("agc", {"target_dbfs": -20.0, "attack_ms": 5.0, "release_ms": 100.0}),
        ("clip_drive_db", 3.0),
        ("suppression", "wiener"),
        ("transients", [{"kind": "click", "offset_n": 0, "peak_db": 0.0}]),
        ("segments", [{"clip_id": "x", "start_n": 0, "end_n": 1, "rir_id": None, "lowpass_hz": 1000.0}]),
        ("band", "narrow_leg"),
        ("codec", "amr_nb"),
        ("codec", "amr_wb"),
        ("codec", "opus"),
        ("condition", "competing"),
        ("condition", "babble"),
        ("condition", "ssn"),
        ("condition", "city"),
        ("condition", "as_is"),
    ],
)
def test_a_leg_that_is_not_built_raises_instead_of_being_skipped(raw, field, value):
    with pytest.raises(NotImplementedError, match=field):
        construct_sample(make_row(**{field: value}), raw)


def test_a_clip_drive_of_zero_still_asks_for_the_leg(raw):
    with pytest.raises(NotImplementedError, match="clip_drive_db"):
        construct_sample(make_row(clip_drive_db=0.0), raw)


def test_a_noise_row_without_exactly_one_noise_is_refused(raw):
    with pytest.raises(ValueError, match="one noise"):
        construct_sample(make_row(**noisy()) | {"noise_source_ids": [], "noise_offsets": []}, raw)


def test_an_offset_off_the_millisecond_grid_is_refused(raw):
    with pytest.raises(ValueError, match="millisecond"):
        construct_sample(make_row(speech_offset_n=17, total_n=6 * 16000), raw)
