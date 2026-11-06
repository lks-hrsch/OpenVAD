# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from fetch_support import archive_with, clean_run
from openvad_train.fetch import common, mls


def test_fetch_mls_extracts_counts_and_deletes_the_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mls, "MLS_LANGS", {"polish": "0" * 32})
    fetched = []

    def fake_download_ranges(url, dest, *, md5=None):
        fetched.append((url.rsplit("/", 1)[1], md5))
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_with(
            dest,
            {
                "mls_polish/train/transcripts.txt": b"1_2_000001\tx\n1_2_000002\tx\n3_4_000001\tx\n",
                "mls_polish/dev/transcripts.txt": b"5_6_000001\tx\n",
                "mls_polish/test/transcripts.txt": b"",
            },
        )
        return dest

    monkeypatch.setattr(mls, "download_ranges", fake_download_ranges)
    mls.fetch_mls()
    assert fetched == [("mls_polish.tar.gz", "0" * 32)]
    manifest = clean_run("mls")
    assert manifest["sources"] == [{"url": f"{mls.MLS_URL}/mls_polish.tar.gz", "md5": "0" * 32}]
    assert manifest["languages"]["polish"]["splits"] == {
        "train": {"segments": 3, "speakers": 2},
        "dev": {"segments": 1, "speakers": 1},
        "test": {"segments": 0, "speakers": 0},
    }


def test_fetch_mls_resume_keeps_complete_files_and_replaces_partial_ones(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mls, "MLS_LANGS", {"polish": "0" * 32})

    def fake_download_ranges(url, dest, *, md5=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        members = {f"mls_polish/{s}/transcripts.txt": b"1_2_000001\tx\n" for s in mls.MLS_SPLITS}
        archive_with(dest, {**members, "mls_polish/train/audio/a.flac": b"complete audio"})
        return dest

    monkeypatch.setattr(mls, "download_ranges", fake_download_ranges)
    root = tmp_path / "datasets/raw/mls/mls_polish"
    (root / "train/audio").mkdir(parents=True)
    done = root / "train/transcripts.txt"
    done.write_bytes(b"1_2_000009\tx\n")  # same size as in the archive, so a rewrite would be visible
    partial = root / "train/audio/a.flac"
    partial.write_bytes(b"comp")
    mls.fetch_mls()
    assert done.read_bytes() == b"1_2_000009\tx\n"
    assert partial.read_bytes() == b"complete audio"
    assert (root / "dev/transcripts.txt").read_bytes() == b"1_2_000001\tx\n"


def test_marker_skip_deletes_an_orphaned_archive(monkeypatch, tmp_path):
    # A crash between the marker and the archive deletion left the archive in the cache.
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mls, "MLS_LANGS", {"polish": "0" * 32})
    monkeypatch.setattr(mls, "download_ranges", lambda *_, **__: pytest.fail("must not refetch"))
    out = common.raw_dir("mls")
    for split in mls.MLS_SPLITS:
        (out / "mls_polish" / split).mkdir(parents=True)
        (out / "mls_polish" / split / "transcripts.txt").write_text("1_2_000001\tx\n")
    (out / ".polish-extracted").touch()
    common.cache_dir().mkdir(parents=True)
    (common.cache_dir() / "mls_polish.tar.gz").write_bytes(b"orphan")
    mls.fetch_mls()
    assert not (common.cache_dir() / "mls_polish.tar.gz").exists()
    assert common.is_complete(out)
