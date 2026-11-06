"""Lugosch's Montreal Forced Aligner alignments for LibriSpeech, from Zenodo."""

import json
import zipfile

from openvad_train.fetch.common import cache_dir, download, raw_dir, read_text, write_manifest


ALIGNMENTS_RECORD = 2619474
ALIGNMENTS_FILE = "librispeech_alignments.zip"
# The 360 alignments label the VOiCES speakers from train-clean-360 (ADR 0023),
# though the LibriSpeech 360 audio is no longer fetched.
ALIGNMENT_SPLITS = ("train-clean-100", "train-clean-360", "dev-clean", "test-clean")


def alignment_members(names: list[str]) -> list[str]:
    """Zip members (files only) in a LibriSpeech split we use; the zip also holds the other splits."""
    return [n for n in names if not n.endswith("/") and n.split("/", 1)[0] in ALIGNMENT_SPLITS]


def fetch_alignments() -> None:
    out = raw_dir("librispeech-alignments")
    out.mkdir(parents=True, exist_ok=True)
    record = json.loads(read_text(f"https://zenodo.org/api/records/{ALIGNMENTS_RECORD}"))
    entry = next(f for f in record["files"] if f["key"] == ALIGNMENTS_FILE)
    md5 = entry["checksum"].removeprefix("md5:")
    url = entry["links"]["self"]
    archive = download(url, cache_dir() / ALIGNMENTS_FILE, md5=md5)
    with zipfile.ZipFile(archive) as z:
        z.extractall(out, members=alignment_members(z.namelist()))
    write_manifest(
        out,
        {
            "dataset": "librispeech-alignments",
            "licence": "CC BY 4.0",
            "use": "source alignments of LibriSpeech, labels for VOiCES (ADR 0023) "
            "and the test-clean benchmark (ADR 0020)",
            "record": ALIGNMENTS_RECORD,
            "sources": [{"url": url, "md5": md5}],
            "splits": list(ALIGNMENT_SPLITS),
        },
    )
    archive.unlink()
