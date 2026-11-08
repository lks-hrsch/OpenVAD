# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import io
import tarfile

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from as_is_support import make_ami_tree, make_librivad_tree, make_voices_tree, write_noise
from mfa_source_support import make_sc_tree, make_vp_tree
from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.config import SLICE
from openvad_train.dataset.parquet_io import write_parquet
from openvad_train.dataset.plan import clip_rows
from openvad_train.dataset.schema import Timing
from openvad_train.dataset.sources import ami, librivad, speechcommands, voices, voxpopuli


def ramp(n: int) -> np.ndarray:
    return ((np.arange(n) % 2000 - 1000) / 2000).astype(np.float32)


def flac_of(x: np.ndarray, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def wav_on_disk(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, ramp(32000), 16000, subtype="PCM_16")
    return path


def wav_bytes(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def callhome_parquet(tmp_path):
    def row(x: np.ndarray, rate: int = 16000) -> dict:
        return {"audio": {"bytes": wav_bytes(x, rate), "path": None}}

    path = tmp_path / "data-00000-of-00001.parquet"
    frame = pl.DataFrame([row(ramp(16000)), row(np.ones(16000, np.float32) * 0.25), row(ramp(8000), 8000)])
    asyncio.run(write_parquet({path: frame.lazy()}))
    return path


def tar_with(tmp_path, members: dict[str, bytes]):
    path = tmp_path / "s.tar"
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(path, "r:") as tar:
        return path, {m.name: (m.offset_data, m.size) for m in tar.getmembers()}


def test_load_audio_reads_a_tar_member_by_offset(tmp_path):
    x, other = ramp(4000), ramp(9000)[::-1]
    path, where = tar_with(tmp_path, {"a.flac": flac_of(other), "b.flac": flac_of(x)})
    row = {
        "audio_format": "flac_tar",
        "audio_filepath": path.name,
        "byte_offset": where["b.flac"][0],
        "byte_length": where["b.flac"][1],
    }
    y = load_audio(row, tmp_path)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, x, atol=1 / 16384)


def test_load_audio_refuses_a_pointer_past_the_end_of_the_tar(tmp_path):
    path, where = tar_with(tmp_path, {"a.flac": flac_of(ramp(4000))})
    row = {
        "audio_format": "flac_tar",
        "audio_filepath": path.name,
        "byte_offset": path.stat().st_size - 10,
        "byte_length": where["a.flac"][1],
    }
    with pytest.raises(ValueError, match="bytes"):
        load_audio(row, tmp_path)


def test_load_audio_refuses_a_stored_rate_other_than_16k(tmp_path):
    path, where = tar_with(tmp_path, {"a.flac": flac_of(ramp(4000), 8000)})
    row = {
        "audio_format": "flac_tar",
        "audio_filepath": path.name,
        "byte_offset": where["a.flac"][0],
        "byte_length": where["a.flac"][1],
    }
    with pytest.raises(ValueError, match="8000"):
        load_audio(row, tmp_path)


def test_load_audio_reads_raw_file_with_offset(wav_on_disk):
    row = {"audio_format": "file", "audio_filepath": wav_on_disk.name, "offset": 1.0, "duration": 0.5}
    y = load_audio(row, wav_on_disk.parent)
    assert len(y) == 8000
    np.testing.assert_allclose(y, ramp(32000)[16000:24000], atol=1 / 16384)


def test_load_audio_reads_callhome_row(callhome_parquet):
    row = {
        "audio_format": "callhome_parquet",
        "audio_filepath": callhome_parquet.name,
        "row_in_file": 0,
        "offset": 0.0,
        "duration": 0.5,
    }
    assert len(load_audio(row, callhome_parquet.parent)) == 8000


def test_load_audio_takes_the_callhome_row_and_offset_it_names(callhome_parquet):
    row = {
        "audio_format": "callhome_parquet",
        "audio_filepath": callhome_parquet.name,
        "row_in_file": 1,
        "offset": 0.25,
        "duration": 0.5,
    }
    y = load_audio(row, callhome_parquet.parent)
    assert len(y) == 8000
    np.testing.assert_allclose(y, 0.25, atol=1e-3)
    row["row_in_file"] = 0
    np.testing.assert_allclose(load_audio(row, callhome_parquet.parent), ramp(16000)[4000:12000], atol=1 / 16384)


def test_load_audio_refuses_callhome_audio_at_another_rate(callhome_parquet):
    row = {
        "audio_format": "callhome_parquet",
        "audio_filepath": callhome_parquet.name,
        "row_in_file": 2,
        "offset": 0.0,
        "duration": 0.5,
    }
    with pytest.raises(ValueError, match="8000"):
        load_audio(row, callhome_parquet.parent)


def test_load_audio_defaults_to_the_data_directory(wav_on_disk, monkeypatch):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(wav_on_disk.parent))
    row = {"audio_format": "file", "audio_filepath": wav_on_disk.name, "offset": 0.0, "duration": 0.5}
    assert len(load_audio(row)) == 8000


def test_load_audio_refuses_an_unknown_format():
    with pytest.raises(ValueError, match="audio_format"):
        load_audio({"audio_format": "tape", "audio_filepath": "a"})


# ---------------------------------------------------------------------------------------------------- as-is rows


@pytest.fixture
def raw(tmp_path):
    """The data root: rows point at `datasets/raw/<source>/...` relative to it."""
    return tmp_path


def _words_to(seconds: float) -> list[tuple[int, int]]:
    return [(1000 + i * 800, 1500 + i * 800) for i in range(round(seconds * 1000 / 800))]


@pytest.fixture
def as_is_row_of(raw):
    """The last plan row of an as-is source's first suitable clip, so a cut clip is read at an offset."""
    raw_dir = raw / "datasets/raw"

    def build(source: str) -> dict:
        if source == "ami":
            tree = make_ami_tree(raw_dir / "ami")
            clip = next(c for c in ami.clips(tree))
            write_noise(raw / clip.path, 70.0, seed=3)
            clip = next(c for c in ami.clips(tree) if c.clip_id == clip.clip_id)
            timing = Timing(clip.clip_id, "word", _words_to(65), [], None, "t")
        elif source == "voxpopuli":
            tree = make_vp_tree(raw_dir / "voxpopuli")
            clip = next(c for c in voxpopuli.clips(tree) if c.clip_id == "de_s2_1")
            (raw / clip.path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(raw / clip.path, np.random.default_rng(4).uniform(-0.5, 0.5, 160 * 16000), 16000, format="OGG")
            timing = Timing(clip.clip_id, "word", [(500, 3000)], [], None, "t")
        elif source == "speech-commands":
            tree = make_sc_tree(raw_dir / "speech-commands")
            clip = next(c for c in speechcommands.clips(tree) if c.clip_id == "no/cccccccc_nohash_0")
            write_noise(raw / clip.path, 0.75, seed=5)
            timing = Timing(clip.clip_id, "word", [(100, 600)], [], None, "t")
        elif source == "voices":
            tree = make_voices_tree(raw_dir / "voices")
            clip = next(c for c in voices.clips(tree))
            (timing,) = voices.timings(tree, [clip])
        else:
            tree = make_librivad_tree(raw_dir / "librivad")
            clip = next(c for c in librivad.clips(tree) if "Concat" in c.path)
            (timing,) = librivad.timings(tree, [clip])
        return clip_rows(clip, timing, SLICE)[-1]

    return build


def source_slice(row: dict, raw) -> np.ndarray:
    audio, rate = sf.read(raw / row["audio_filepath"], dtype="float32", always_2d=True)
    assert rate == 16000
    start = round(row["offset"] * 16000)
    return audio.mean(axis=1)[start : start + round(row["duration"] * 16000)]


@pytest.mark.parametrize("source", ["ami", "voxpopuli", "speech-commands", "voices", "librivad"])
def test_as_is_row_reads_the_exact_source_samples(source, as_is_row_of, raw):
    row = as_is_row_of(source)
    assert row["audio_format"] == "file"
    assert row["shard"] is None
    assert row["treatment"] == "as_is"
    assert row["audio_filepath"].startswith(f"datasets/raw/{source}/")
    audio, expected = load_audio(row, raw), source_slice(row, raw)
    assert len(expected) == round(row["duration"] * 16000) > 0
    if source == "voxpopuli":
        # A long Ogg is read by seeking, which agrees with a whole decode to 2e-7 (`decode_16k`).
        assert np.allclose(audio, expected, atol=1e-5)
    else:
        assert np.array_equal(audio, expected)


def test_a_cut_ami_row_starts_inside_its_file(as_is_row_of):
    assert as_is_row_of("ami")["offset"] > 0
