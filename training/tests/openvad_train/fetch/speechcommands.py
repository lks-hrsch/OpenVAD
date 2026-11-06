# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from fetch_support import archive_with, clean_run, orphan_archive
from openvad_train.fetch import common, speechcommands


def test_sc_splits_use_the_official_lists(tmp_path):
    # Different counts per split, and clips != speakers, so swapped lists or a wrong speaker key fail.
    train = ("yes/aa_nohash_0.wav", "yes/aa_nohash_1.wav", "no/dd_nohash_0.wav")
    validation = ("yes/bb_nohash_0.wav", "no/bb_nohash_1.wav")
    testing = ("no/cc_nohash_1.wav",)
    for rel in (*train, *validation, *testing, "_background_noise_/x.wav"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    (tmp_path / "validation_list.txt").write_text("\n".join(validation) + "\n")
    (tmp_path / "testing_list.txt").write_text("\n".join(testing) + "\n")
    assert speechcommands.sc_splits(tmp_path) == {
        "train": {"clips": 3, "speakers": 2},
        "validation": {"clips": 2, "speakers": 1},
        "testing": {"clips": 1, "speakers": 1},
    }


MAIN, TEST_SET = speechcommands.SC_ARCHIVES


def _fake_download(monkeypatch):
    members = {
        MAIN: {
            "yes/aa_nohash_0.wav": b"w",
            "no/cc_nohash_1.wav": b"w",
            "validation_list.txt": b"",
            "testing_list.txt": b"no/cc_nohash_1.wav\n",
        },
        TEST_SET: {"yes/zz_nohash_0.wav": b"w"},
    }
    fetched = []

    def fake_download(url, dest, *, sha256=None):
        fetched.append((dest.name, sha256))
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_with(dest, members[dest.name])
        return dest

    monkeypatch.setattr(speechcommands, "download", fake_download)
    return fetched


def test_fetch_speechcommands_extracts_the_two_archives_into_separate_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    fetched = _fake_download(monkeypatch)
    speechcommands.fetch_speechcommands()
    assert fetched == list(speechcommands.SC_ARCHIVES.items())
    manifest = clean_run("speech-commands")
    assert manifest["splits"]["testing"]["clips"] == 1
    out = common.raw_dir("speech-commands")
    assert (out / "speech_commands_v0.02/yes/aa_nohash_0.wav").exists()
    assert (out / "speech_commands_test_set_v0.02/yes/zz_nohash_0.wav").exists()
    assert not (out / "speech_commands_v0.02/yes/zz_nohash_0.wav").exists()


def test_fetch_speechcommands_skips_an_archive_with_a_marker_and_removes_its_orphan(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    fetched = _fake_download(monkeypatch)
    out = common.raw_dir("speech-commands")
    out.mkdir(parents=True)
    (out / f".{TEST_SET}-extracted").touch()  # a crash after the marker of the test set
    orphan_archive(TEST_SET)
    speechcommands.fetch_speechcommands()
    assert [name for name, _ in fetched] == [MAIN]
    clean_run("speech-commands")
