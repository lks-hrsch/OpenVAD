# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import hashlib

import pytest

from fetch_support import clean_run, hf_server
from openvad_train.fetch import callhome, common


def _callhome_server(monkeypatch, oid_of_eng):
    api = f"https://huggingface.co/api/datasets/{callhome.CALLHOME_REPO}/tree/{callhome.CALLHOME_REV}?recursive=true"
    base = f"https://huggingface.co/datasets/{callhome.CALLHOME_REPO}/resolve/{callhome.CALLHOME_REV}"
    data = {"eng/a.parquet": b"english", "deu/b.parquet": b"german"}
    tree = [
        {
            "type": "file",
            "path": p,
            "lfs": {"oid": oid_of_eng if p.startswith("eng") else hashlib.sha256(d).hexdigest()},
        }
        for p, d in data.items()
    ]
    tree += [
        {"type": "file", "path": "fra/c.parquet", "lfs": {"oid": "f" * 64}},
        {"type": "file", "path": "eng/README.md"},
    ]
    monkeypatch.setattr(common, "hf_token", lambda: "tok")
    return hf_server(monkeypatch, {api: tree}, {f"{base}/{p}": d for p, d in data.items()}), base


def test_fetch_callhome_sends_the_token_and_fetches_only_the_used_shards(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    sha_eng = hashlib.sha256(b"english").hexdigest()
    served, base = _callhome_server(monkeypatch, sha_eng)
    callhome.fetch_callhome()
    assert len(served) == 3  # the listing and the two parquet shards of the used languages
    assert all(r.get_header("Authorization") == "Bearer tok" for r in served)
    out = common.raw_dir("callhome")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.parquet")) == ["deu/b.parquet", "eng/a.parquet"]
    assert clean_run("callhome")["sources"] == [
        {"url": f"{base}/eng/a.parquet", "sha256": sha_eng},
        {"url": f"{base}/deu/b.parquet", "sha256": hashlib.sha256(b"german").hexdigest()},
    ]


def test_fetch_callhome_rejects_a_shard_that_fails_its_sha256(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    _callhome_server(monkeypatch, "0" * 64)
    with pytest.raises(ValueError, match=r"sha256 mismatch for a\.parquet"):
        callhome.fetch_callhome()
    assert not (common.raw_dir("callhome") / "manifest.json").exists()
    assert not (common.raw_dir("callhome") / "eng/a.parquet").exists()
