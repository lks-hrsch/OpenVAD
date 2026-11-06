# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Google Speech Commands v0.02: one-second words, with the official speaker-disjoint split lists."""

import tarfile
from pathlib import Path

from openvad_train.fetch.common import cache_dir, download, raw_dir, tar_filter, write_manifest


SC_URL = "http://download.tensorflow.org/data"
# SHA-256s from tensorflow/datasets `speech_commands/checksums.tsv`.
SC_ARCHIVES = {
    "speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
    "speech_commands_test_set_v0.02.tar.gz": "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df",
}


def sc_splits(root: Path) -> dict[str, dict[str, int]]:
    """Clips and speakers per official split: `validation_list.txt` and `testing_list.txt`, the rest is train.

    File names are `<word>/<speaker hash>_nohash_<n>.wav`; `_background_noise_` holds long noise files, not words.
    """
    held = {s: set((root / f"{s}_list.txt").read_text().split()) for s in ("validation", "testing")}
    clips = {"train": [], "validation": [], "testing": []}
    for wav in root.glob("*/*.wav"):
        rel = wav.relative_to(root).as_posix()
        if rel.startswith("_background_noise_/"):
            continue
        split = next((s for s, names in held.items() if rel in names), "train")
        clips[split].append(wav.name.split("_nohash_", 1)[0])
    return {s: {"clips": len(v), "speakers": len(set(v))} for s, v in clips.items()}


def fetch_speechcommands() -> None:
    out = raw_dir("speech-commands")
    out.mkdir(parents=True, exist_ok=True)
    markers = []
    for name, sha256 in SC_ARCHIVES.items():
        markers.append(marker := out / f".{name}-extracted")
        archive = cache_dir() / name
        if marker.exists():
            archive.unlink(missing_ok=True)  # left behind by a crash after the marker
            continue
        download(f"{SC_URL}/{name}", archive, sha256=sha256)
        target = out / name.removesuffix(".tar.gz")
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter=tar_filter)
        marker.touch()
        archive.unlink()
    info = {
        "dataset": "speech-commands",
        "licence": "CC BY 4.0",
        "use": "train, dev, test, unmodified (ADR 0023): the official speaker-disjoint split lists",
        "sources": [{"url": f"{SC_URL}/{n}", "sha256": s} for n, s in SC_ARCHIVES.items()],
        "splits": sc_splits(out / "speech_commands_v0.02"),
    }
    write_manifest(out, info)
    for marker in markers:
        marker.unlink(missing_ok=True)
