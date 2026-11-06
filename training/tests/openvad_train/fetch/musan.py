# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections import Counter

import pytest

from fetch_support import archive_with, clean_run
from openvad_train.fetch import common, musan


def test_musan_licence_table_matches_the_audit():
    # Hand-audited from the per-source LICENSE files; an edit that loses a row or relabels one must be deliberate.
    assert Counter(musan.musan_licences().values()) == {
        "Public Domain": 1349,
        "CC BY-SA 3.0": 256,
        "CC BY 3.0": 254,
        "CC BY": 81,
        "CC BY-ND 3.0": 32,
        "no statement": 14,
        "CC BY-SA 4.0": 13,
        "CC BY-SA 2.5": 6,
        "CC BY 4.0": 4,
        "CC BY-SA 1.0": 4,
        "CC BY-SA": 3,
    }


def _fake_musan(monkeypatch, tmp_path, wavs):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    members = {f"musan/{w}": b"x" for w in wavs} | {"musan/music/jamendo/LICENSE": b"..."}

    def fake_download_ranges(url, dest, *, md5):
        assert md5 == musan.MUSAN_MD5
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_with(dest, members)
        return dest

    monkeypatch.setattr(musan, "download_ranges", fake_download_ranges)


def test_fetch_musan_extracts_every_file_with_its_licence(monkeypatch, tmp_path):
    wavs = ["music/jamendo/music-jamendo-0009.wav", "music/jamendo/music-jamendo-0107.wav"]
    _fake_musan(monkeypatch, tmp_path, [*wavs, "noise/free-sound/noise-free-sound-0000.wav"])
    musan.fetch_musan()
    out = common.raw_dir("musan")
    manifest = clean_run("musan")
    assert sorted(manifest["licences"]) == sorted(p.stem for p in out.rglob("*.wav"))
    assert manifest["licences"]["music-jamendo-0009"] == "CC BY-ND 3.0"
    assert manifest["licences"]["music-jamendo-0107"] == "no statement"
    assert manifest["licences"]["noise-free-sound-0000"] == "Public Domain"
    assert (out / "musan/music/jamendo/LICENSE").exists()


def test_fetch_musan_rejects_a_file_without_licence_entry(monkeypatch, tmp_path):
    _fake_musan(monkeypatch, tmp_path, ["music/jamendo/music-jamendo-9999.wav"])
    with pytest.raises(ValueError, match="music-jamendo-9999"):
        musan.fetch_musan()
    assert not (tmp_path / "datasets/raw/musan/manifest.json").exists()
