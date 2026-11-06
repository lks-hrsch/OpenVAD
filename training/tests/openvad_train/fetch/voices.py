# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import pytest

from fetch_support import archive_with, clean_run, orphan_archive
from openvad_train.fetch import common, voices


def test_voices_with_a_bad_gzip_crc_keeps_the_archive_and_no_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))

    def fake_download_ranges(url, dest, **_):
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_with(dest, {f"f{i}": os.urandom(5000) for i in range(3)})
        raw = bytearray(dest.read_bytes())
        raw[-8] ^= 0xFF  # the CRC32 in the gzip trailer
        dest.write_bytes(raw)
        return dest

    monkeypatch.setattr(voices, "download_ranges", fake_download_ranges)
    with pytest.raises(RuntimeError, match="CRC check failed"):
        voices.fetch_voices()
    out = common.raw_dir("voices")
    assert (common.cache_dir() / voices.VOICES_ARCHIVE).exists()
    assert not (out / ".devkit-extracted").exists()
    assert common.read_manifest(out) is None


def test_voices_does_not_download_or_extract_again_after_its_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(voices, "download_ranges", lambda *_, **__: pytest.fail("must not download"))
    out = common.raw_dir("voices")
    out.mkdir(parents=True)
    (out / ".devkit-extracted").touch()
    orphan_archive(voices.VOICES_ARCHIVE)
    voices.fetch_voices()
    clean_run("voices")
