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

from openvad_train.dataset import codecs, construct
from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.codecs import DecodeError, MissingCodec
from openvad_train.dataset.config import SLICE, BuildConfig, dataset_root, work_dir
from openvad_train.dataset.construct import OutOfSpace, PlanMismatch, construct_sample, run_construct, write_shard
from openvad_train.dataset.cutting import Chunk
from openvad_train.dataset.mix import dither, loop_slice
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.plan import PLAN_SCHEMA, draw_sample
from openvad_train.dataset.schema import Clip, Timing
from openvad_train.fetch.common import cache_dir, data_dir, take_share_lock, write_manifest


GB = 1_000_000_000
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


@pytest.mark.parametrize("extra", [CLEAN, noisy()], ids=["clean", "noisy"])
def test_common_voice_waits_for_its_wada_snr_step(raw, extra):
    with pytest.raises(NotImplementedError, match="baseline_snr_db"):
        construct_sample(make_row(**extra, source="common-voice"), raw)


def test_a_clip_drive_of_zero_still_asks_for_the_leg(raw):
    with pytest.raises(NotImplementedError, match="clip_drive_db"):
        construct_sample(make_row(clip_drive_db=0.0), raw)


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
    real, calls = construct.construct_sample, []

    def fails_once(row, root):
        calls.append(row["sample_id"])
        if len(calls) == 1:
            raise DecodeError("a transient failure")
        return real(row, root)

    monkeypatch.setattr(construct, "construct_sample", fails_once)
    result = write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert (result.n_ok, result.n_failed) == (3, 0)
    assert calls[:2] == [rows[0]["sample_id"]] * 2
    assert (tmp_path / "s.failures.jsonl").read_text() == ""


def test_a_row_is_tried_twice_before_it_is_logged(rows_with_corrupt_clip, raw, tmp_path, monkeypatch):
    real, calls = construct.construct_sample, []

    def counting(row, root):
        calls.append(row["sample_id"])
        return real(row, root)

    monkeypatch.setattr(construct, "construct_sample", counting)
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

    monkeypatch.setattr(construct, "construct_sample", broken)
    with pytest.raises(type(error)):
        write_shard(rows, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert finished_looking(tmp_path) == []


def test_verification_refuses_labels_that_do_not_match_the_audio(rows, raw, tmp_path, monkeypatch):
    real = construct.construct_sample

    def one_frame_short(row, root):
        flac, labels = real(row, root)
        return flac, labels[:-1]

    monkeypatch.setattr(construct, "construct_sample", one_frame_short)
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


def test_a_row_asking_for_an_unbuilt_leg_aborts_through_the_workers(cfg, planned_rows):
    write_plan(cfg, [planned_rows[0] | {"room": True}, *planned_rows[1:]])
    with pytest.raises(NotImplementedError, match="room"):
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
