# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gzip

import pytest

from fetch_support import clean_run, orphan_archive, plain_tar, tar_with
from openvad_train.fetch import common, voxpopuli


HEADER = "id_|paragraph_id|session_id|speaker_id|original_text|normed_text|decoded"
HEADER += "|start_time|end_time|cer|wer|vad|split|gender"


def _write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("\n".join([HEADER, *rows]) + "\n")
    return path


def test_voxpopuli_stats_per_split(tmp_path):
    rows = [
        "a|p|s|1|t|t|t|3600|5400|0|0|[]|train|f",  # a non-zero start: the duration is end minus start
        "b|p|s|1|t|t|t|0|1800|0|0|[]|train|f",  # the same speaker again
        "c|p|s|None|t|t|t|0|1800|0|0|[]|train|m",
        "d|p|s|None|t|t|t|0|1800|0|0|[]|train|m",
        "e|p|s|2|t|t|t|0|360|0|0|[]|dev|m",
        "f|p|s|3|t|t|t|0|36|9|9|[]|invalid|m",
    ]
    assert voxpopuli.voxpopuli_stats(_write_table(tmp_path / "asr_xx.tsv.gz", rows)) == {
        "dev": {"hours": 0.1, "speakers": 1, "unknown_speaker_hours": 0.0},
        "invalid": {"hours": 0.0, "speakers": 1, "unknown_speaker_hours": 0.0},
        "train": {"hours": 2.0, "speakers": 1, "unknown_speaker_hours": 1.0},
    }


def test_voxpopuli_with_a_truncated_tar_keeps_the_archive_and_no_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(voxpopuli, "VOXPOPULI_LANGS", ())
    monkeypatch.setattr(voxpopuli, "VOXPOPULI_YEARS", (2009,))

    def fake_download_ranges(url, dest, **_):
        dest.parent.mkdir(parents=True, exist_ok=True)
        plain_tar(dest, hole=True)
        return dest

    monkeypatch.setattr(voxpopuli, "download_ranges", fake_download_ranges)
    with pytest.raises(RuntimeError, match="ends at byte"):
        voxpopuli.fetch_voxpopuli()
    out = common.raw_dir("voxpopuli")
    assert (tmp_path / "cache/voxpopuli-original_2009.tar").exists()
    assert not (out / ".original_2009-extracted").exists()
    assert not (out / "manifest.json").exists()


def test_fetch_voxpopuli_resumes_after_a_year_marker_and_records_stats_and_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(voxpopuli, "VOXPOPULI_LANGS", ("xx",))
    monkeypatch.setattr(voxpopuli, "VOXPOPULI_YEARS", (2009, 2010))
    rows = ["a|p|s|1|t|t|t|0|3600|0|0|[]|train|f", "b|p|s|None|t|t|t|0|1800|0|0|[]|train|m"]
    monkeypatch.setattr(voxpopuli, "download", lambda url, dest, **_: _write_table(dest, rows))
    fetched = []

    def fake_download_ranges(url, dest, **_):
        fetched.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tar_with(dest, f"{dest.stem.split('_')[1]}/s.ogg", b"o")
        return dest

    monkeypatch.setattr(voxpopuli, "download_ranges", fake_download_ranges)
    out = common.raw_dir("voxpopuli")
    out.mkdir(parents=True)
    (out / ".original_2009-extracted").touch()  # a crash after the marker of 2009
    orphan_archive("voxpopuli-original_2009.tar")
    voxpopuli.fetch_voxpopuli()
    assert fetched == ["voxpopuli-original_2010.tar"]
    assert (out / "raw_audios/2010/s.ogg").exists()
    manifest = clean_run("voxpopuli")
    assert manifest["languages"]["xx"] == {"train": {"hours": 1.5, "speakers": 1, "unknown_speaker_hours": 0.5}}
    url = voxpopuli.VOXPOPULI_URL
    assert [x["url"] for x in manifest["sources"]] == [
        f"{url}/annotations/asr/asr_xx.tsv.gz",
        f"{url}/audios/original_2009.tar",
        f"{url}/audios/original_2010.tar",
    ]
