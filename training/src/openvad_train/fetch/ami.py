"""AMI Meeting Corpus: individual headset (IHM) audio of the full-corpus-ASR partition, plus the manual annotations."""

import re
import tarfile
import zipfile
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


AMI_URL = "https://groups.inf.ed.ac.uk/ami"
AMI_ANNOTATIONS = "ami_public_manual_1.6.2.zip"
# OpenSLR 16 serves the same IHM wavs as one archive; Edinburgh answers 403 to per-file requests from Python.
AMI_HEADSET_URL = "https://openslr.elda.org/resources/16/headset.tar.gz"
# The full-corpus-ASR partition (https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml), by session; speaker-disjoint.
AMI_PARTITION = {
    "train": [
        "ES2002",
        "ES2005",
        "ES2006",
        "ES2007",
        "ES2008",
        "ES2009",
        "ES2010",
        "ES2012",
        "ES2013",
        "ES2015",
        "ES2016",
        "IS1000",
        "IS1001",
        "IS1002",
        "IS1003",
        "IS1004",
        "IS1005",
        "IS1006",
        "IS1007",
        "TS3005",
        "TS3008",
        "TS3009",
        "TS3010",
        "TS3011",
        "TS3012",
        "EN2001",
        "EN2003",
        "EN2004",
        "EN2005",
        "EN2006",
        "EN2009",
        "IN1001",
        "IN1002",
        "IN1005",
        "IN1007",
        "IN1008",
        "IN1009",
        "IN1012",
        "IN1013",
        "IN1014",
        "IN1016",
        "ES2014",
        "TS3007",
        "ES2003",
        "TS3006",
    ],
    "dev": ["ES2011", "IS1008", "TS3004", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"],
    "test": ["ES2004", "IS1009", "TS3003", "EN2002"],
}


def ami_headsets(meetings_xml: str) -> dict[str, list[int]]:
    """Headset channels per meeting of the partition, from `corpusResources/meetings.xml`."""
    sessions = {s for ss in AMI_PARTITION.values() for s in ss}
    out = {}
    for attrs, body in re.findall(r"<meeting ([^>]*)>(.*?)</meeting>", meetings_xml, re.S):
        found = re.search(r'observation="([^"]+)"', attrs)
        if found and found.group(1)[:6] in sessions:
            out[found.group(1)] = sorted(int(c) for c in re.findall(r'channel="(\d+)"', body))
    return out


def fetch_ami() -> None:
    out = raw_dir("ami")
    annotations_url = f"{AMI_URL}/AMICorpusAnnotations/{AMI_ANNOTATIONS}"
    zip_path = download(annotations_url, out / AMI_ANNOTATIONS)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out / "annotations")
    headsets = ami_headsets((out / "annotations/corpusResources/meetings.xml").read_text(encoding="latin-1"))
    # Members are named <meeting>/audio/<meeting>.Headset-<ch>.wav; the archive is streamed once.
    wanted = {
        f"{m}/audio/{m}.Headset-{ch}.wav": f"audio/{m}/{m}.Headset-{ch}.wav"
        for m, chs in headsets.items()
        for ch in chs
    }
    marker = out / ".headset-extracted"
    archive = cache_dir() / "ami-headset.tar.gz"
    if not marker.exists():
        download_ranges(AMI_HEADSET_URL, archive)
        extracted = 0
        # Random access ("r:gz"), not a stream, so that the gzip CRC can be checked at the end.
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name in wanted:
                    member.name = wanted[member.name]
                    tar.extract(member, out, filter=tar_filter)
                    extracted += 1
            check_archive_end(tar, archive)
        if not extracted:
            raise RuntimeError(f"AMI: no wanted headset in {archive}; the member names may differ, the archive is kept")
        marker.touch()
    # Also after a skip: a crash between the marker and the deletion leaves the archive behind.
    archive.unlink(missing_ok=True)
    missing = [Path(dest).name for dest in wanted.values() if not (out / dest).exists()]
    splits = {s: sorted(m for m in headsets if m[:6] in ss) for s, ss in AMI_PARTITION.items()}
    info = {
        "dataset": "ami",
        "licence": "CC BY 4.0 (Edinburgh release); the OpenSLR 16 page states CC BY-NC-SA 2.0 (modified)",
        "use": "train, dev, test, unmodified (ADR 0023): the full-corpus-ASR partition, individual headsets",
        "sources": [{"url": annotations_url}, {"url": AMI_HEADSET_URL}],
        "meetings": splits,
        "missing": missing,
    }
    write_manifest(out, info)
    marker.unlink(missing_ok=True)
