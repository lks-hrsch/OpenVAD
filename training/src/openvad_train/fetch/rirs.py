# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Room impulse responses for the room step of the channel chain: BUT ReverbDB and OpenSLR 28 (RIRS_NOISES).

Only the RIR wavs are kept. BUT ReverbDB's 60 s room-noise recordings are most of its 8.7 GB, and the simulated RIRs
and the noises of RIRS_NOISES are not wanted. The fetcher writes `index.tsv`, one line per RIR with the room it was
measured in, so the dataset stage never walks the tree.
"""

import re
import tarfile
import zipfile
from pathlib import Path

from openvad_train.fetch.common import cache_dir, download_ranges, raw_dir, tar_filter, write_manifest


# BUT Speech@FIT release 19_06, RIRs and room noise only, linked from
# https://speech.fit.vut.cz/software/but-speech-fit-reverb-database. The publisher gives no checksum, so the SHA-256
# is that of the file downloaded on 2026-11-08 and only detects a later change of the file.
BUT_URL = "https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz"
BUT_BYTES = 9_308_593_693
BUT_SHA256 = "d0d14cdd626b4b05401027c5a774b4f24369f4ea38f57480c6fc6e04e9f3f5b5"
# OpenSLR resource 28 (https://www.openslr.org/28/), which publishes no checksum either; the same rule. Many of its
# files have several channels (microphone arrays); the first is used.
OPENSLR_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
OPENSLR_BYTES = 1_311_166_223
OPENSLR_SHA256 = "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"
OPENSLR_MD5 = "e6f48e257286e05de56413b4779d8ffb"

# ./PLACE/MIC_SETUP/SPK_SETUP/MIC_ID/RIR/<name>.wav; PLACE is the room (read_me.txt of the release).
_BUT_RIR = re.compile(r"(?:\./)?[^/]+/[^/]+/[^/]+/[^/]+/RIR/[^/]+\.wav")
_REAL = "RIRS_NOISES/real_rirs_isotropic_noises/"
# The 325 real RIRs of RIRS_NOISES come from three databases (AIR, the 2014 REVERB challenge, RWCP) and carry no
# room column, so the room is read off the file name: the position, angle and microphone numbers are dropped.
_OPENSLR_ROOMS = (
    ("air", re.compile(r"air_type1_air_(?:binaural|phone)_(?:bt_)?(.+?)(?:\d?_(?:hhp|hfrp)|(?:_\d+)+)\.wav")),
    ("rvb", re.compile(r"RVB2014_type\d_rir_(\w+?room\d)_(?:near|far)_\w+\.wav")),
    ("rwcp", re.compile(r"RWCP_type\d_rir_(?:circle|cirline)_([a-z0-9]+)_imp\w*\.wav")),
    ("rwcp", re.compile(r"RWCP_(type4)_rir_\w+\.wav")),
)


def openslr_room(name: str) -> str | None:
    """The room of a file of `real_rirs_isotropic_noises`, None for a noise or a list.

    A RIR of no known pattern is an error, so a changed upstream cannot slip a room in unnoticed.
    """
    if "_noise_" in name or not name.endswith(".wav"):
        return None
    for database, pattern in _OPENSLR_ROOMS:
        if m := pattern.fullmatch(name):
            return f"{database}-{m[1]}"
    raise ValueError(f"{name}: cannot tell which room this RIR was measured in")


def _download(url: str, dest: Path, size: int, *, md5: str | None = None, sha256: str | None = None) -> Path:
    """`download_ranges`, and the size pinned here, since the publishers give no checksum to check it against."""
    path = download_ranges(url, dest, md5=md5, sha256=sha256)
    if (got := path.stat().st_size) != size:
        raise ValueError(f"{path.name}: {got} bytes, expected {size}; delete it to refetch")
    return path


def _extract_but(archive: Path, out: Path) -> None:
    # Streamed, because the archive is 9 GB and only the RIR directories are wanted.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and _BUT_RIR.fullmatch(member.name):
                tar.extract(member, out, filter=tar_filter)


def _extract_openslr(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        keep = ["RIRS_NOISES/README"] + [
            n for n in z.namelist() if n.startswith(_REAL) and openslr_room(n[len(_REAL) :])
        ]
        z.extractall(out, members=keep)


def rir_index(out: Path) -> list[tuple[str, str]]:
    """(path under `out`, room) of every RIR in the tree, sorted. Rooms are prefixed with the set they belong to."""
    rows = [
        (p.relative_to(out).as_posix(), f"but-{p.relative_to(out / 'but').parts[0]}")
        for p in sorted((out / "but").rglob("*.wav"))
    ]
    rows += [
        (p.relative_to(out).as_posix(), f"openslr28-{openslr_room(p.name)}")
        for p in sorted((out / "openslr28" / _REAL).glob("*.wav"))
    ]
    return rows


def fetch_rirs() -> None:
    out = raw_dir("rirs")
    out.mkdir(parents=True, exist_ok=True)
    cache = cache_dir()
    but_archive = _download(BUT_URL, cache / "but_reverbdb.tgz", BUT_BYTES, sha256=BUT_SHA256)
    openslr_archive = _download(
        OPENSLR_URL, cache / "rirs_noises.zip", OPENSLR_BYTES, md5=OPENSLR_MD5, sha256=OPENSLR_SHA256
    )
    # A marker after each extraction: a crash between the two must not unpack the 9 GB archive again.
    for archive, extract, subdir in (
        (but_archive, _extract_but, "but"),
        (openslr_archive, _extract_openslr, "openslr28"),
    ):
        marker = out / f".{subdir}-extracted"
        if not marker.exists():
            extract(archive, out / subdir)
            marker.touch()
    index = rir_index(out)
    (out / "index.tsv").write_text("path\troom\n" + "".join(f"{path}\t{room}\n" for path, room in index))
    rooms: dict[str, set[str]] = {"but": set(), "openslr28": set()}
    for _, room in index:
        rooms[room.split("-", 1)[0]].add(room)
    files = {name: sum(room.startswith(f"{name}-") for _, room in index) for name in rooms}
    write_manifest(
        out,
        {
            "dataset": "rirs",
            "licence": "BUT ReverbDB: CC BY 4.0; OpenSLR 28 RIRS_NOISES: Apache-2.0",
            "licences": {"but": "CC BY 4.0", "openslr28": "Apache-2.0"},
            "use": "room impulse responses of the room step (ADR 0021); rooms split 80/10/10 by hash of the room",
            "sources": [
                {"url": BUT_URL, "sha256": BUT_SHA256},
                {"url": OPENSLR_URL, "md5": OPENSLR_MD5, "sha256": OPENSLR_SHA256},
            ],
            "sets": {name: {"rirs": files[name], "rooms": len(rooms[name])} for name in rooms},
        },
    )
    for path in (but_archive, openslr_archive, *out.glob(".*-extracted")):
        path.unlink()
