# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import struct
import zipfile

from fetch_support import clean_run, wav_header
from openvad_train.fetch import common, wham


def test_wav_format_skips_chunks_before_fmt():
    header = wav_header(rate=16000, channels=2, bits=32, data=0, extra=b"JUNK" + struct.pack("<I", 3) + b"abc\0")
    assert wham.wav_format(header) == {"sample_rate": 16000, "channels": 2, "bits": 32}


def test_fetch_wham_records_format_and_splits_and_skips_finder_files(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    # 400 bytes per second, so 180 s per file is 0.05 h: large enough to survive the rounding to 0.01 h.
    samples = 180 * 400
    wav = wav_header(rate=100, channels=2, bits=16, data=samples) + bytes(samples)
    members = {f"wham_noise/{s}/{i}.wav": wav for s, n in (("tr", 2), ("cv", 1), ("tt", 1)) for i in range(n)}
    members |= {"wham_noise/metadata/noise_meta_tt.csv": b"utterance_id\n", "wham_noise/.DS_Store": b"x"}

    def fake_download_ranges(url, dest, *, md5=None):
        assert (url, md5) == (wham.WHAM_URL, wham.WHAM_MD5)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            for name, data in members.items():
                z.writestr(name, data)
        return dest

    monkeypatch.setattr(wham, "download_ranges", fake_download_ranges)
    wham.fetch_wham()
    manifest = clean_run("wham")
    assert manifest["format"] == {"sample_rate": 100, "channels": 2, "bits": 16}
    assert manifest["splits"] == {
        "tr": {"files": 2, "hours": 0.1},
        "cv": {"files": 1, "hours": 0.05},
        "tt": {"files": 1, "hours": 0.05},
    }
    out = common.raw_dir("wham")
    assert (out / "wham_noise/metadata/noise_meta_tt.csv").exists()
    assert not (out / "wham_noise/.DS_Store").exists()


def test_fetch_wham_does_not_extract_again_after_its_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    wav = wav_header(rate=16000, channels=1, bits=16, data=2) + b"\0\0"

    def fake_download_ranges(url, dest, *, md5=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            for s in wham.WHAM_SPLITS:
                z.writestr(f"wham_noise/{s}/new.wav", wav)
        return dest

    monkeypatch.setattr(wham, "download_ranges", fake_download_ranges)
    out = common.raw_dir("wham")
    for s in wham.WHAM_SPLITS:  # a crash after the marker: the tree is there
        (out / f"wham_noise/{s}").mkdir(parents=True)
        (out / f"wham_noise/{s}/old.wav").write_bytes(wav)
    (out / ".wham_noise-extracted").touch()
    wham.fetch_wham()
    assert not list(out.glob("wham_noise/*/new.wav"))
    clean_run("wham")
