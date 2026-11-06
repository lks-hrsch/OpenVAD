"""VoxPopuli transcribed (ASR) data: the annotation tables and the source session audio; segments are cut later.

The VoxPopuli code is CC BY-NC 4.0 and not vendored; this reimplements the two downloads it needs.
"""

import collections
import csv
import gzip
import tarfile
from pathlib import Path

from openvad_train.fetch.common import (
    cache_dir,
    check_archive_end,
    download,
    download_ranges,
    raw_dir,
    tar_filter,
    write_manifest,
)


VOXPOPULI_URL = "https://dl.fbaipublicfiles.com/voxpopuli"
# Transcribed languages with an MFA acoustic model (Slovak, Slovene and Lithuanian have none).
VOXPOPULI_LANGS = ("en", "de", "fr", "es", "pl", "ro", "it", "cs", "hu", "nl", "hr", "fi", "et")
# The transcribed segments all come from the "original" session audio of these years.
VOXPOPULI_YEARS = tuple(range(2009, 2021))


def voxpopuli_stats(table: Path) -> dict:
    """Hours and speakers of the valid rows per official split (`invalid` rows do not match their text).

    A `speaker_id` of `None` means the speaker is unknown: those rows count towards no speaker and cannot be split
    by speaker, so ADR 0023 keeps them for train only. Their hours are in `hours` and in `unknown_speaker_hours`.
    """
    csv.field_size_limit(1 << 30)
    hours, unknown = collections.defaultdict(float), collections.defaultdict(float)
    speakers = collections.defaultdict(set)
    with gzip.open(table, "rt", newline="") as f:
        for row in csv.DictReader(f, delimiter="|"):
            duration = float(row["end_time"]) - float(row["start_time"])
            hours[row["split"]] += duration
            if row["speaker_id"] == "None":
                unknown[row["split"]] += duration
            else:
                speakers[row["split"]].add(row["speaker_id"])
    return {
        s: {
            "hours": round(hours[s] / 3600, 1),
            "speakers": len(speakers[s]),
            "unknown_speaker_hours": round(unknown[s] / 3600, 1),
        }
        for s in sorted(hours)
    }


def fetch_voxpopuli() -> None:
    out = raw_dir("voxpopuli")
    out.mkdir(parents=True, exist_ok=True)
    languages = {}
    sources = []
    for lang in VOXPOPULI_LANGS:
        url = f"{VOXPOPULI_URL}/annotations/asr/asr_{lang}.tsv.gz"
        sources.append({"url": url})
        table = download(url, out / "annotations" / f"asr_{lang}.tsv.gz")
        languages[lang] = voxpopuli_stats(table)
    markers = []
    for year in VOXPOPULI_YEARS:
        markers.append(marker := out / f".original_{year}-extracted")
        archive = cache_dir() / f"voxpopuli-original_{year}.tar"
        if marker.exists():
            archive.unlink(missing_ok=True)  # left behind by a crash after the marker
            continue
        # No checksums are published; the size and the tar end check the download (early ends and boundary holes).
        download_ranges(f"{VOXPOPULI_URL}/audios/original_{year}.tar", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(out / "raw_audios", filter=tar_filter)
            check_archive_end(tar, archive)
        marker.touch()
        archive.unlink()
    info = {
        "dataset": "voxpopuli",
        "licence": "CC0 (data); the code is CC BY-NC 4.0 and not used",
        "use": (
            "train, dev, test, unmodified (ADR 0023); our own split by speaker, the official splits share speakers; "
            "rows with speaker_id None train only"
        ),
        "sources": [*sources, *({"url": f"{VOXPOPULI_URL}/audios/original_{y}.tar"} for y in VOXPOPULI_YEARS)],
        "languages": languages,
    }
    write_manifest(out, info)
    for marker in markers:
        marker.unlink(missing_ok=True)
