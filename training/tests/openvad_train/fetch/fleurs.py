# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import hashlib

import pytest

from fetch_support import archive_with, clean_run, hf_server, orphan_archive
from openvad_train.fetch import common, fleurs


_FLEURS_NAMES = [f"{s}.tsv" for s in fleurs.FLEURS_SPLITS] + [f"audio/{s}.tar.gz" for s in fleurs.FLEURS_SPLITS]


def test_fleurs_files_requires_all_six_files():
    tree = [{"type": "file", "path": f"data/xx/{n}"} for n in _FLEURS_NAMES[:-1]]
    with pytest.raises(RuntimeError, match=r"xx.*5 of the 6"):
        fleurs.fleurs_files(tree, "xx")


def _fleurs_server(monkeypatch, tmp_path, langs, *, oid=None):
    """Serve `langs` over a fake Hugging Face; `oid` replaces the SHA-256 of every archive."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(fleurs, "FLEURS_LANGS", langs)
    monkeypatch.setattr(fleurs, "hf_token", lambda: "tok")
    base = f"https://huggingface.co/datasets/{fleurs.FLEURS_REPO}/resolve/{fleurs.FLEURS_REV}"
    listings, files = {}, {}
    for lang in langs:
        tree = []
        for s in fleurs.FLEURS_SPLITS:
            archive_with(tmp_path / "build.tar.gz", {f"{s}/{lang}-{s}.wav": b"w"})
            archive = (tmp_path / "build.tar.gz").read_bytes()
            # Two utterances of one hour each: 57,600,000 samples at 16 kHz.
            files[f"{base}/data/{lang}/{s}.tsv"] = f"1\t{s}.wav\tt\tt\tt\t{16000 * 3600}\tm\n".encode() * 2
            files[f"{base}/data/{lang}/audio/{s}.tar.gz"] = archive
            tree += [
                {"type": "file", "path": f"data/{lang}/{s}.tsv"},
                {
                    "type": "file",
                    "path": f"data/{lang}/audio/{s}.tar.gz",
                    "lfs": {"oid": oid or hashlib.sha256(archive).hexdigest()},
                },
            ]
        api = f"https://huggingface.co/api/datasets/{fleurs.FLEURS_REPO}/tree/{fleurs.FLEURS_REV}/data/{lang}?recursive=true"
        listings[api] = tree
    (tmp_path / "build.tar.gz").unlink()
    return hf_server(monkeypatch, listings, files)


def test_fetch_fleurs_sends_the_token_on_archive_requests_and_extracts_each_split_once(monkeypatch, tmp_path):
    served = _fleurs_server(monkeypatch, tmp_path, ("aa_bb", "cc_dd"))
    out = common.raw_dir("fleurs")
    # A crash after the marker of aa_bb train: its audio is on disk and its archive is still in the cache.
    (out / "aa_bb/audio/train").mkdir(parents=True)
    (out / "aa_bb/audio/train/old.wav").write_bytes(b"old")
    (out / ".aa_bb-train-extracted").touch()
    orphan_archive("fleurs-aa_bb-train.tar.gz")
    fleurs.fetch_fleurs()
    archives = [r for r in served if r.full_url.endswith(".tar.gz")]
    assert len(archives) == 5
    assert all(r.get_header("Authorization") == "Bearer tok" for r in archives)
    assert (out / "aa_bb/audio/train/old.wav").exists()
    assert not (out / "aa_bb/audio/train/aa_bb-train.wav").exists()
    assert (out / "cc_dd/audio/test/cc_dd-test.wav").exists()
    manifest = clean_run("fleurs")
    assert manifest["languages"]["aa_bb"]["train"] == {"utterances": 2, "hours": 2.0}
    assert len(manifest["sources"]) == 12


def test_fetch_fleurs_rejects_an_archive_that_fails_its_sha256(monkeypatch, tmp_path):
    _fleurs_server(monkeypatch, tmp_path, ("aa_bb",), oid="0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fleurs.fetch_fleurs()
    out = common.raw_dir("fleurs")
    assert not (out / "manifest.json").exists()
    assert not list(out.rglob("*.wav"))
