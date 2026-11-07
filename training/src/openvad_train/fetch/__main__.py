# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Fetch one public dataset into $OPENVAD_DATA_DIR/datasets/raw."""

import argparse
import fcntl
from collections.abc import Callable
from pathlib import Path

from openvad_train.fetch.alignments import fetch_alignments
from openvad_train.fetch.ami import fetch_ami
from openvad_train.fetch.ava import fetch_ava
from openvad_train.fetch.callhome import fetch_callhome
from openvad_train.fetch.common import (
    MANIFEST,
    cache_dir,
    count_tree,
    is_complete,
    raw_dir,
    read_manifest,
    take_share_lock,
)
from openvad_train.fetch.commonvoice import fetch_commonvoice
from openvad_train.fetch.demand import fetch_demand
from openvad_train.fetch.fleurs import fetch_fleurs
from openvad_train.fetch.librispeech import fetch_librispeech
from openvad_train.fetch.librivad import fetch_librivad, fetch_librivad_small
from openvad_train.fetch.mls import fetch_mls
from openvad_train.fetch.musan import fetch_musan
from openvad_train.fetch.picovoice import fetch_picovoice
from openvad_train.fetch.simsamu import fetch_simsamu
from openvad_train.fetch.speechcommands import fetch_speechcommands
from openvad_train.fetch.swc import fetch_swc
from openvad_train.fetch.voices import fetch_voices
from openvad_train.fetch.voxpopuli import fetch_voxpopuli
from openvad_train.fetch.wham import fetch_wham


FETCHERS: dict[str, Callable[[], None]] = {
    "librispeech": fetch_librispeech,
    "alignments": fetch_alignments,
    "ava": fetch_ava,
    "librivad": fetch_librivad,
    "demand": fetch_demand,
    "voices": fetch_voices,
    "callhome": fetch_callhome,
    "swc": fetch_swc,
    "musan": fetch_musan,
    "librivad-small": fetch_librivad_small,
    "picovoice": fetch_picovoice,
    "common-voice": fetch_commonvoice,
    "simsamu": fetch_simsamu,
    "wham": fetch_wham,
    "mls": fetch_mls,
    "fleurs": fetch_fleurs,
    "voxpopuli": fetch_voxpopuli,
    "ami": fetch_ami,
    "speech-commands": fetch_speechcommands,
}
# Directory under datasets/raw where it differs from the fetcher name.
RAW_NAMES = {"alignments": "librispeech-alignments", "ava": "ava-speech"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(FETCHERS))
    name: str = parser.parse_args(argv).dataset
    cache_dir().mkdir(parents=True, exist_ok=True)
    # Two runs of one dataset would zero-fill each other's pieces; the flock is released when the file closes.
    # The flock is local and per user because it is unreliable on the SMB share that holds the data root.
    local = Path.home() / ".cache/openvad"
    local.mkdir(parents=True, exist_ok=True)
    with (local / f"fetch-{name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{name}: another fetch of this dataset is already running") from None
        # Two machines mount the share and flock does not reach across SMB, so a lock file on the share covers them.
        share_lock = take_share_lock(name)
        try:
            _fetch(name)
        finally:
            share_lock.unlink(missing_ok=True)


def _fetch(name: str) -> None:
    root = raw_dir(RAW_NAMES.get(name, name))
    manifest = read_manifest(root)
    if manifest is not None:
        if is_complete(root):
            # Trees finished by older code keep the in-tree markers that fetchers now delete after the manifest.
            for marker in (*root.glob(".*-extracted"), root / ".setup-done", root / ".generate-done"):
                marker.unlink(missing_ok=True)
            print(f"{name}: already complete at {root}")
            return
        files, size = count_tree(root)
        raise RuntimeError(
            f"{name}: {root} does not match its manifest (expected {manifest.get('files')} files, "
            f"{manifest.get('bytes')} bytes; found {files} files, {size} bytes); "
            f"delete {root / MANIFEST} to refetch"
        )
    FETCHERS[name]()
    print(f"{name}: fetched to {root}")


if __name__ == "__main__":
    main()
