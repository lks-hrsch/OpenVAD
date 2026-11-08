# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io
import tarfile
import zipfile
from itertools import chain

import pytest

from fetch_support import clean_run, wav_header
from openvad_train.dataset.channel import RirPool
from openvad_train.fetch import __main__ as cli
from openvad_train.fetch import common, rirs


WAV = wav_header(rate=16000, channels=1, bits=16, data=4) + bytes(4)
REAL = rirs._REAL


@pytest.mark.parametrize(
    ("name", "room"),
    [
        ("air_type1_air_binaural_aula_carolina_1_3_135_3.wav", "air-aula_carolina"),
        ("air_type1_air_binaural_booth_0_1.wav", "air-booth"),
        ("air_type1_air_binaural_stairway_1_2_105.wav", "air-stairway"),
        ("air_type1_air_phone_stairway1_hfrp.wav", "air-stairway"),
        ("air_type1_air_phone_bt_corridor_hhp.wav", "air-corridor"),
        ("air_type1_air_phone_lecture_hhp.wav", "air-lecture"),
        ("RVB2014_type1_rir_mediumroom2_far_anglb.wav", "rvb-mediumroom2"),
        ("RVB2014_type2_rir_simroom3_near_angla.wav", "rvb-simroom3"),
        ("RWCP_type1_rir_circle_ane_imp000.wav", "rwcp-ane"),
        ("RWCP_type2_rir_cirline_e2b_imp070.wav", "rwcp-e2b"),
        ("RWCP_type3_rir_cirline_ofc_imp_rev.wav", "rwcp-ofc"),
        ("RWCP_type4_rir_m10l.wav", "rwcp-type4"),
    ],
)
def test_openslr_room_is_read_off_the_name(name, room):
    assert rirs.openslr_room(name) == room


@pytest.mark.parametrize(
    "name", ["RWCP_type5_noise_cirline_ofc_ambient1.wav", "RVB2014_type1_noise_simroom1_1.wav", "rir_list"]
)
def test_openslr_noises_and_lists_have_no_room(name):
    assert rirs.openslr_room(name) is None


def test_openslr_rir_of_no_known_pattern_is_an_error():
    with pytest.raises(ValueError, match="which room"):
        rirs.openslr_room("new_database_rir_1.wav")


def archive_bytes(but: dict[str, bytes], openslr: dict[str, bytes]) -> dict[str, bytes]:
    tgz, zipped = io.BytesIO(), io.BytesIO()
    with tarfile.open(fileobj=tgz, mode="w:gz") as tar:
        for name, data in but.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(zipped, "w") as z:
        for name, data in openslr.items():
            z.writestr(name, data)
    return {rirs.BUT_URL: tgz.getvalue(), rirs.OPENSLR_URL: zipped.getvalue()}


def fake_download(
    monkeypatch, but_members: dict[str, bytes], openslr_members: dict[str, bytes], pin_sizes: bool = True
):
    archives = archive_bytes(but_members, openslr_members)
    served = []
    if pin_sizes:  # the real archives are gigabytes
        monkeypatch.setattr(rirs, "BUT_BYTES", len(archives[rirs.BUT_URL]))
        monkeypatch.setattr(rirs, "OPENSLR_BYTES", len(archives[rirs.OPENSLR_URL]))

    def fake_download_ranges(url, dest, *, md5=None, sha256=None):
        served.append((url, sha256))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archives[url])
        return dest

    monkeypatch.setattr(rirs, "download_ranges", fake_download_ranges)
    return served


BUT = {
    "./VUT_FIT_L207/MicID01/SpkID01_20171223_S/01/RIR/IR_sweep.v00.wav": WAV,
    "./VUT_FIT_L207/MicID01/SpkID01_20171223_S/02/RIR/IR_sweep.v00.wav": WAV,
    "./VUT_FIT_L207/MicID01/SpkID01_20171223_S/01/silence/silence_60sec.v00.wav": WAV,
    "./VUT_FIT_L207/MicID01/SpkID01_20171223_S/01/mic_meta.txt": b"meta",
    "./VUT_FIT_Q301/MicID02/SpkID01_20171223_S/05/RIR/IR_sweep.v01.wav": WAV,
}
OPENSLR = {
    "RIRS_NOISES/README": b"readme",
    f"{REAL}air_type1_air_binaural_booth_0_1.wav": WAV,
    f"{REAL}RWCP_type1_rir_circle_ane_imp000.wav": WAV,
    f"{REAL}RWCP_type1_rir_circle_ane_imp010.wav": WAV,
    f"{REAL}RWCP_type5_noise_cirline_ofc_ambient1.wav": WAV,
    f"{REAL}rir_list": b"list",
    "RIRS_NOISES/simulated_rirs/smallroom/Room001/Room001-00001.wav": WAV,
    "RIRS_NOISES/pointsource_noises/noise-free-sound-0000.wav": WAV,
}


@pytest.fixture
def fetched(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    served = fake_download(monkeypatch, BUT, OPENSLR)
    rirs.fetch_rirs()
    return served


def test_an_archive_of_another_size_than_the_pinned_one_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    fake_download(monkeypatch, BUT, OPENSLR, pin_sizes=False)
    with pytest.raises(ValueError, match="expected 9308593693"):
        rirs.fetch_rirs()


def test_fetch_downloads_both_pinned_archives(fetched):
    assert fetched == [(rirs.BUT_URL, rirs.BUT_SHA256), (rirs.OPENSLR_URL, rirs.OPENSLR_SHA256)]
    assert rirs.BUT_URL.startswith("https://")
    assert rirs.OPENSLR_URL.startswith("https://")
    for digest in (rirs.BUT_SHA256, rirs.OPENSLR_SHA256):
        assert len(digest) == 64
        int(digest, 16)


def test_fetch_keeps_the_rirs_and_nothing_else(fetched):
    out = common.raw_dir("rirs")
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert names == [
        "but/VUT_FIT_L207/MicID01/SpkID01_20171223_S/01/RIR/IR_sweep.v00.wav",
        "but/VUT_FIT_L207/MicID01/SpkID01_20171223_S/02/RIR/IR_sweep.v00.wav",
        "but/VUT_FIT_Q301/MicID02/SpkID01_20171223_S/05/RIR/IR_sweep.v01.wav",
        "index.tsv",
        "manifest.json",
        "openslr28/RIRS_NOISES/README",
        f"openslr28/{REAL}RWCP_type1_rir_circle_ane_imp000.wav",
        f"openslr28/{REAL}RWCP_type1_rir_circle_ane_imp010.wav",
        f"openslr28/{REAL}air_type1_air_binaural_booth_0_1.wav",
    ]


def test_fetch_records_licences_sets_and_a_complete_manifest(fetched):
    manifest = clean_run("rirs")
    assert manifest["licences"] == {"but": "CC BY 4.0", "openslr28": "Apache-2.0"}
    assert "CC BY 4.0" in manifest["licence"]
    assert "Apache-2.0" in manifest["licence"]
    assert manifest["sets"] == {"but": {"rirs": 3, "rooms": 2}, "openslr28": {"rirs": 3, "rooms": 2}}
    assert [s["url"] for s in manifest["sources"]] == [rirs.BUT_URL, rirs.OPENSLR_URL]


def test_the_index_lists_every_rir_with_its_room_and_feeds_the_pool(fetched):
    out = common.raw_dir("rirs")
    lines = (out / "index.tsv").read_text().splitlines()
    assert lines[0] == "path\troom"
    assert {line.split("\t")[1] for line in lines[1:]} == {
        "but-VUT_FIT_L207",
        "but-VUT_FIT_Q301",
        "openslr28-air-booth",
        "openslr28-rwcp-ane",
    }
    assert all((out / line.split("\t")[0]).is_file() for line in lines[1:])
    pool = RirPool(out)
    in_pool = chain.from_iterable(pool.rooms(s) for s in ("train", "validation", "test"))
    assert sorted(in_pool) == sorted({line.split("\t")[1] for line in lines[1:]})


def test_fetch_does_not_extract_again_after_its_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    fake_download(monkeypatch, BUT, OPENSLR)
    out = common.raw_dir("rirs")
    old = out / "but/VUT_FIT_OLD/m/s/1/RIR/old.wav"
    old.parent.mkdir(parents=True)
    old.write_bytes(WAV)
    (out / ".but-extracted").touch()
    rirs.fetch_rirs()
    assert not list(out.glob("but/VUT_FIT_L207"))
    assert (out / "openslr28" / REAL / "RWCP_type1_rir_circle_ane_imp000.wav").exists()
    assert "but-VUT_FIT_OLD" in (out / "index.tsv").read_text()
    clean_run("rirs")


def test_a_new_openslr_rir_name_stops_the_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    fake_download(monkeypatch, BUT, OPENSLR | {f"{REAL}mystery_rir_1.wav": WAV})
    with pytest.raises(ValueError, match="which room"):
        rirs.fetch_rirs()


def test_rirs_is_a_registered_fetcher():
    assert cli.FETCHERS["rirs"] is rirs.fetch_rirs
