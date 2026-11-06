"""Multilingual LibriSpeech (OpenSLR 94) as FLAC, the languages with a pretrained MFA model (English is not used)."""

import tarfile
from pathlib import Path

from openvad_train.fetch.common import cache_dir, download_ranges, raw_dir, tar_filter, write_manifest


MLS_URL = "https://dl.fbaipublicfiles.com/mls"
# MD5s from https://dl.fbaipublicfiles.com/mls/md5sum.txt.
MLS_LANGS = {
    "german": "91ac982bf63869307f1b8950dfb7c776",
    "french": "4172e807697259bff9ad63661aecabf6",
    "spanish": "6c34698dd522dde28fdc43309e9cc1ac",
    "italian": "dc77f5805aecc7182aa20786032e5dc1",
    "dutch": "6b171a16baff0108efd320b1ad65b9d1",
    "polish": "ce1a1278006cc373c9d1cb6dbfc03d47",
    "portuguese": "12d54613fae75ae5fb1d55836408f3ee",
}
MLS_SPLITS = ("train", "dev", "test")


def mls_counts(lang_dir: Path) -> dict[str, dict[str, int]]:
    """Segments and speakers per split, from `<split>/transcripts.txt` (IDs are `<speaker>_<book>_<segment>`)."""
    counts = {}
    for split in MLS_SPLITS:
        ids = [line.split("\t", 1)[0] for line in (lang_dir / split / "transcripts.txt").read_text().splitlines()]
        counts[split] = {"segments": len(ids), "speakers": len({i.split("_", 1)[0] for i in ids})}
    return counts


def fetch_mls() -> None:
    out = raw_dir("mls")
    out.mkdir(parents=True, exist_ok=True)
    languages = {}
    markers = []
    for lang, md5 in MLS_LANGS.items():
        # The archives are 6 to 123 GB, so each is extracted and deleted before the next one is fetched.
        markers.append(marker := out / f".{lang}-extracted")
        archive = cache_dir() / f"mls_{lang}.tar.gz"
        if not marker.exists():
            with tarfile.open(download_ranges(f"{MLS_URL}/mls_{lang}.tar.gz", archive, md5=md5)) as tar:
                # A file already on disk at full size is skipped, so a rerun after a kill resumes.
                for member in tar:
                    dest = out / member.name
                    if not (member.isfile() and dest.is_file() and dest.stat().st_size == member.size):
                        tar.extract(member, out, filter=tar_filter)
            marker.touch()
        # Also after a skip: a crash between the marker and the deletion leaves the archive behind.
        archive.unlink(missing_ok=True)
        languages[lang] = {"md5": md5, "splits": mls_counts(out / f"mls_{lang}")}
    info = {
        "dataset": "mls",
        "licence": "CC BY 4.0",
        "use": "train (augmented and telephone channel, ADR 0023), dev, test: the official speaker-disjoint splits",
        "sources": [{"url": f"{MLS_URL}/mls_{lang}.tar.gz", "md5": md5} for lang, md5 in MLS_LANGS.items()],
        "languages": languages,
    }
    write_manifest(out, info)
    # Markers stay until the manifest exists; after that a rerun must extract again, not bless a damaged tree.
    for marker in markers:
        marker.unlink(missing_ok=True)
