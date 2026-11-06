# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import zipfile

from fetch_support import clean_run, orphan_archive
from openvad_train.fetch import common, demand


def test_fetch_demand_extracts_each_zip_once_and_lists_48k_only_scenes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    files = [
        {"key": f"{k}.zip", "checksum": f"md5:{c * 32}", "links": {"self": f"https://x/{k}.zip"}}
        for k, c in (("A_16k", "1"), ("B_16k", "2"), ("SCAFE_48k", "3"), ("A_48k", "4"))
    ]
    monkeypatch.setattr(demand, "read_text", lambda _: json.dumps({"files": files}))
    fetched = []

    def fake_download(url, dest, *, md5=None):
        fetched.append((dest.name, md5))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr(f"{dest.name.split('_')[0]}/ch01.wav", b"new")
        return dest

    monkeypatch.setattr(demand, "download", fake_download)
    out = common.raw_dir("demand")
    # A crash after the marker of A_16k: its scene is on disk, and its archive is still in the cache.
    (out / "A").mkdir(parents=True)
    (out / "A/ch01.wav").write_bytes(b"old")
    (out / ".A_16k.zip-extracted").touch()
    orphan_archive("A_16k.zip")
    demand.fetch_demand()
    assert fetched == [("B_16k.zip", "2" * 32)]
    assert (out / "A/ch01.wav").read_bytes() == b"old"
    assert (out / "B/ch01.wav").read_bytes() == b"new"
    manifest = clean_run("demand")
    assert manifest["sources"] == [
        {"url": "https://x/A_16k.zip", "md5": "1" * 32},
        {"url": "https://x/B_16k.zip", "md5": "2" * 32},
    ]
    assert manifest["missing_16k"] == ["SCAFE"]
