"""VOiCES (re-recorded LibriSpeech in four rooms), the devkit only, from the public S3 bucket."""

import tarfile

from openvad_train.fetch.common import (
    cache_dir,
    check_archive_end,
    download_ranges,
    raw_dir,
    tar_filter,
    write_manifest,
)


VOICES_URL = "https://lab41openaudiocorpus.s3.amazonaws.com"
# The devkit (29.5 GB, mics 1 and 5, 2 utterances per speaker) extracts to VOiCES_devkit/. The full release
# (448.8 GB, 999,168 wavs) re-records the same 3,903 utterances 256 times, so it is not fetched.
VOICES_ARCHIVE = "VOiCES_devkit.tar.gz"


def fetch_voices() -> None:
    out = raw_dir("voices")
    out.mkdir(parents=True, exist_ok=True)
    url = f"{VOICES_URL}/{VOICES_ARCHIVE}"
    archive = cache_dir() / VOICES_ARCHIVE
    marker = out / ".devkit-extracted"
    if not marker.exists():
        # No checksums are published; the size, the gzip CRC and the tar end check the download.
        with tarfile.open(download_ranges(url, archive)) as tar:
            tar.extractall(out, filter=tar_filter)
            check_archive_end(tar, archive)
        marker.touch()
    info = {
        "dataset": "voices",
        "licence": "CC BY 4.0",
        "use": (
            "train, dev, test, unmodified (ADR 0023); the 12 dev-clean speakers left out, "
            "the 64 train-clean-100 speakers train only"
        ),
        "sources": [{"url": url}],
    }
    write_manifest(out, info)
    for path in (archive, marker):
        path.unlink(missing_ok=True)
