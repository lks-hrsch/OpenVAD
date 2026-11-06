"""MUSAN (music, speech, noise) from OpenSLR 17, every file, with the committed per-file licence table."""

import csv
import tarfile
from pathlib import Path

from openvad_train.fetch.common import cache_dir, download_ranges, raw_dir, tar_filter, write_manifest


MUSAN_URL = "https://openslr.elda.org/resources/17/musan.tar.gz"
# OpenSLR publishes no checksum; this is the MD5 of the archive as first fetched (2026-11-01).
MUSAN_MD5 = "0c472d4fc0c5141eca47ad1ffeb2a7df"
# The licence of every file, read from the per-source LICENSE files; "no statement" where they give none.
MUSAN_LICENCES = Path(__file__).with_name("musan_licences.tsv")


def musan_licences() -> dict[str, str]:
    """Licence per file ID."""
    with MUSAN_LICENCES.open() as f:
        return {row["file_id"]: row["licence"] for row in csv.DictReader(f, delimiter="\t")}


def fetch_musan() -> None:
    out = raw_dir("musan")
    out.mkdir(parents=True, exist_ok=True)
    archive = cache_dir() / "musan.tar.gz"
    licences = musan_licences()
    with tarfile.open(download_ranges(MUSAN_URL, archive, md5=MUSAN_MD5)) as tar:
        ids = sorted(Path(m.name).stem for m in tar.getmembers() if m.name.endswith(".wav"))
        if missing := [i for i in ids if i not in licences]:
            raise ValueError(f"MUSAN: {len(missing)} files have no entry in {MUSAN_LICENCES.name}, first: {missing[0]}")
        tar.extractall(out, filter=tar_filter)
    info = {
        "dataset": "musan",
        "licence": "per file: public domain, CC BY, CC BY-SA, CC BY-ND or no statement (see musan/*/*/LICENSE)",
        "use": "noise and music for augmentation, one shared pool (ADR 0023)",
        "sources": [{"url": MUSAN_URL, "md5": MUSAN_MD5}],
        "licences": {i: licences[i] for i in ids},
    }
    write_manifest(out, info)
    archive.unlink()
