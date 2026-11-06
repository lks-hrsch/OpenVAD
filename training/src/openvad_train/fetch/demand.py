# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""DEMAND environmental noise from Zenodo, the 16 kHz zips only, checked against the record's MD5s."""

import json
import zipfile

from openvad_train.fetch.common import cache_dir, download, raw_dir, read_text, write_manifest


DEMAND_RECORD = 1227121


def demand_entries(files: list[dict]) -> tuple[list[dict], list[str]]:
    """The 16 kHz zips, and the scenes published only at 48 kHz (SCAFE), which are skipped."""
    keys = {f["key"] for f in files}
    entries = sorted((f for f in files if f["key"].endswith("_16k.zip")), key=lambda f: f["key"])
    only_48k = (k for k in keys if k.endswith("_48k.zip") and k.replace("_48k", "_16k") not in keys)
    missing = sorted(k.removesuffix("_48k.zip") for k in only_48k)
    return entries, missing


def fetch_demand() -> None:
    out = raw_dir("demand")
    out.mkdir(parents=True, exist_ok=True)
    record = json.loads(read_text(f"https://zenodo.org/api/records/{DEMAND_RECORD}"))
    entries, missing = demand_entries(record["files"])
    sources = []
    markers = []
    for entry in entries:
        md5 = entry["checksum"].removeprefix("md5:")
        url = entry["links"]["self"]
        sources.append({"url": url, "md5": md5})
        markers.append(marker := out / f".{entry['key']}-extracted")
        archive = cache_dir() / entry["key"]
        if marker.exists():
            archive.unlink(missing_ok=True)  # left behind by a crash after the marker
            continue
        download(url, archive, md5=md5)
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        marker.touch()
        archive.unlink()
    info = {
        "dataset": "demand",
        # The record metadata says CC BY 4.0, its description CC BY-SA 3.0; the stricter one is assumed.
        "licence": "CC BY-SA 3.0 (record metadata says CC BY 4.0)",
        "use": "noise for augmentation, one shared pool (ADR 0023); also mixed into the Picovoice benchmark (ADR 0020)",
        "record": DEMAND_RECORD,
        "sources": sources,
        "missing_16k": missing,
    }
    write_manifest(out, info)
    for marker in markers:
        marker.unlink()
