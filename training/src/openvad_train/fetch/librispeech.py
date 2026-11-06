# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""LibriSpeech from the OpenSLR tarballs, checked against the published MD5s."""

import tarfile

from openvad_train.fetch.common import cache_dir, download, raw_dir, read_text, tar_filter, write_manifest


LIBRISPEECH_MIRROR = "https://openslr.elda.org/resources/12"
LIBRISPEECH_MD5SUM = "https://www.openslr.org/resources/12/md5sum.txt"
LIBRISPEECH_SPLITS = ("train-clean-100", "dev-clean", "test-clean")


def parse_md5sum(text: str) -> dict[str, str]:
    return {name: digest for digest, name in (line.split(maxsplit=1) for line in text.splitlines() if line.strip())}


def fetch_librispeech() -> None:
    out = raw_dir("librispeech")
    out.mkdir(parents=True, exist_ok=True)
    md5s = parse_md5sum(read_text(LIBRISPEECH_MD5SUM))
    sources = []
    archives = []
    markers = []
    for split in LIBRISPEECH_SPLITS:
        name = f"{split}.tar.gz"
        url = f"{LIBRISPEECH_MIRROR}/{name}"
        sources.append({"url": url, "md5": md5s[name]})
        archives.append(cache_dir() / name)
        # The marker lets a rerun skip a split that is already extracted (hours on a network share); it lives in
        # the tree, so deleting the tree to start over also forgets the extraction.
        markers.append(marker := out / f".{split}-extracted")
        if marker.exists():
            continue
        with tarfile.open(download(url, archives[-1], md5=md5s[name])) as tar:
            tar.extractall(out, filter=tar_filter)
        marker.touch()
    info = {
        "dataset": "librispeech",
        "licence": "CC BY 4.0",
        "use": "source of LibriVAD and Picovoice only; test-clean is benchmark audio (ADR 0020), never used directly "
        "in dataset creation (ADR 0023)",
        "sources": sources,
        "splits": list(LIBRISPEECH_SPLITS),
    }
    write_manifest(out, info)
    # Archives and markers stay until the manifest exists, so an interrupted run neither re-downloads nor re-extracts.
    for path in (*archives, *markers):
        path.unlink(missing_ok=True)
