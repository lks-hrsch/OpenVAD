# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Room impulse responses for the room step of the channel chain: BUT ReverbDB and OpenSLR 28 (RIRS_NOISES).

Only the RIR wavs are kept. BUT ReverbDB's 60 s room-noise recordings are most of its 8.7 GB, and the simulated RIRs
and the noises of RIRS_NOISES are not wanted. The fetcher writes `index.tsv`, one line per RIR with the room it was
measured in, so the dataset stage never walks the tree.
"""

import csv
import fnmatch
import re
import tarfile
import zipfile
from collections import Counter
from functools import cache
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
# physical room: the committed table maps each file name pattern to a room, with its source on every line. The
# room-id of the archive's own `rir_list` is not used: it is a Kaldi grouping, and it splits RWCP's E1A to E1C.
OPENSLR_ROOMS = Path(__file__).with_name("openslr28_rooms.tsv")
# What a complete extraction holds, from the archives pinned above: RIRs per room, rooms prefixed with their set.
EXPECTED_ROOMS: dict[str, int] = {
    "but-Hotel_SkalskyDvur_ConferenceRoom2": 124,
    "but-Hotel_SkalskyDvur_Room112": 155,
    "but-VUT_FIT_C236": 310,
    "but-VUT_FIT_D105": 434,
    "but-VUT_FIT_E112": 62,
    "but-VUT_FIT_L207": 465,
    "but-VUT_FIT_L212": 186,
    "but-VUT_FIT_L227": 434,
    "but-VUT_FIT_Q301": 155,
    "openslr28-air-aula-carolina": 11,
    "openslr28-air-bathroom": 2,
    "openslr28-air-booth": 6,
    "openslr28-air-corridor": 3,
    "openslr28-air-kitchen": 2,
    "openslr28-air-lecture": 16,
    "openslr28-air-meeting": 12,
    "openslr28-air-office": 9,
    "openslr28-air-stairway": 46,
    "openslr28-rvb2014": 36,
    "openslr28-rwcp-anechoic": 41,
    "openslr28-rwcp-conference": 15,
    "openslr28-rwcp-echo": 67,
    "openslr28-rwcp-tatami-large": 14,
    "openslr28-rwcp-tatami-small": 7,
    "openslr28-rwcp-type4": 38,
}


@cache
def _room_patterns() -> list[tuple[str, str]]:
    with OPENSLR_ROOMS.open() as f:
        return [(row["pattern"], row["room"]) for row in csv.DictReader(f, delimiter="\t")]


def openslr_room(name: str) -> str | None:
    """The room of a file of `real_rirs_isotropic_noises`, None for a noise or a list.

    A RIR that matches no pattern or more than one is an error, so a changed upstream cannot slip a room in.
    """
    if "_noise_" in name or not name.endswith(".wav"):
        return None
    rooms = {room for pattern, room in _room_patterns() if fnmatch.fnmatchcase(name, pattern)}
    if len(rooms) != 1:
        raise ValueError(f"{name}: expected exactly one room in {OPENSLR_ROOMS.name}, got {sorted(rooms)}")
    return rooms.pop()


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


def _check_complete(index: list[tuple[str, str]]) -> None:
    """The index holds exactly the RIRs and rooms of the pinned archives, else no manifest is written.

    A short extraction would otherwise be recorded as the set, and the room split, which ranks the rooms present,
    would move for every sample.
    """
    found = Counter(room for _, room in index)
    if found != EXPECTED_ROOMS:
        wrong = sorted(
            f"{room}: {found[room]} of {EXPECTED_ROOMS.get(room, 0)}"
            for room in found | EXPECTED_ROOMS
            if found[room] != EXPECTED_ROOMS.get(room, 0)
        )
        raise RuntimeError(
            f"the RIR tree is not the pinned set (found of expected): {'; '.join(wrong)}; delete it to refetch"
        )


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
    _check_complete(index)
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
