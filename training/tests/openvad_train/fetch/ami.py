import zipfile

import pytest

from fetch_support import archive_with, clean_run, orphan_archive
from openvad_train.fetch import ami, common


def test_ami_headsets_keep_partition_meetings_and_their_channels():
    xml = (
        '<meeting observation="ES2002a" duration="1"><speaker channel="0"/><speaker channel="1"/></meeting>'
        '<meeting observation="EN2001a" duration="1"><speaker channel="4"/><speaker channel="0"/></meeting>'
        '<meeting observation="IB4005" duration="1"><speaker channel="0"/></meeting>'
    )
    assert ami.ami_headsets(xml) == {"ES2002a": [0, 1], "EN2001a": [0, 4]}


def test_ami_partition_sessions_do_not_overlap():
    sets = [set(v) for v in ami.AMI_PARTITION.values()]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def _fake_ami(monkeypatch, tmp_path, meetings: dict[str, int], members: dict[str, bytes], *, damaged=False):
    """Fake both downloads, with `meetings` as meeting: headset count; returns the headset archive requests."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    xml = "".join(
        f'<meeting observation="{m}" duration="1">'
        + "".join(f'<speaker channel="{c}"/>' for c in range(n))
        + "</meeting>"
        for m, n in meetings.items()
    )
    fetched = []

    def fake_download(url, dest, **_):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("corpusResources/meetings.xml", xml)
        return dest

    def fake_download_ranges(url, dest, **_):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive_with(dest, members)
        if damaged:  # flip a byte of the gzip CRC in the trailer
            raw = bytearray(dest.read_bytes())
            raw[-8] ^= 0xFF
            dest.write_bytes(raw)
        return dest

    monkeypatch.setattr(ami, "download", fake_download)
    monkeypatch.setattr(ami, "download_ranges", fake_download_ranges)
    return fetched


def test_ami_extracts_only_partition_headsets_and_lists_missing(monkeypatch, tmp_path):
    # ES2002a.Headset-1 is absent; IB4005 is outside the partition; Array1-01 is not a headset.
    names = [
        "ES2002a/audio/ES2002a.Headset-0.wav",
        "EN2001a/audio/EN2001a.Headset-0.wav",
        "ES2004a/audio/ES2004a.Headset-0.wav",
        "IB4005/audio/IB4005.Headset-0.wav",
        "ES2002a/audio/ES2002a.Array1-01.wav",
    ]
    meetings = {"ES2002a": 2, "EN2001a": 1, "ES2004a": 1}
    fetched = _fake_ami(monkeypatch, tmp_path, meetings, {n: n.encode() for n in names})
    ami.fetch_ami()
    out = common.raw_dir("ami")
    assert fetched == [ami.AMI_HEADSET_URL]
    assert sorted(p.relative_to(out / "audio").as_posix() for p in (out / "audio").rglob("*.wav")) == [
        "EN2001a/EN2001a.Headset-0.wav",
        "ES2002a/ES2002a.Headset-0.wav",
        "ES2004a/ES2004a.Headset-0.wav",
    ]
    assert (out / "audio/ES2002a/ES2002a.Headset-0.wav").read_bytes() == b"ES2002a/audio/ES2002a.Headset-0.wav"
    manifest = clean_run("ami")
    assert "CC BY-NC-SA 2.0 (modified)" in manifest["licence"]
    assert manifest["missing"] == ["ES2002a.Headset-1.wav"]
    assert manifest["meetings"] == {"train": ["EN2001a", "ES2002a"], "dev": [], "test": ["ES2004a"]}


def test_ami_keeps_the_archive_when_no_member_matches(monkeypatch, tmp_path):
    # A prefix on every member name, as a different archive layout would have.
    members = {"amicorpus/ES2002a/audio/ES2002a.Headset-0.wav": b"x"}
    _fake_ami(monkeypatch, tmp_path, {"ES2002a": 1}, members)
    with pytest.raises(RuntimeError, match="no wanted headset"):
        ami.fetch_ami()
    out = common.raw_dir("ami")
    assert (common.cache_dir() / "ami-headset.tar.gz").exists()
    assert not (out / ".headset-extracted").exists()
    assert not (out / "manifest.json").exists()


def test_ami_rejects_an_archive_with_a_bad_crc(monkeypatch, tmp_path):
    # OpenSLR 16 publishes no checksum, so the gzip CRC is the only check of the bytes.
    members = {"ES2002a/audio/ES2002a.Headset-0.wav": b"x"}
    _fake_ami(monkeypatch, tmp_path, {"ES2002a": 1}, members, damaged=True)
    with pytest.raises(RuntimeError, match="delete it to refetch"):
        ami.fetch_ami()
    out = common.raw_dir("ami")
    assert not (out / ".headset-extracted").exists()
    assert not (out / "manifest.json").exists()


def test_ami_marker_skips_the_archive_and_deletes_an_orphan(monkeypatch, tmp_path):
    # A crash after the marker: the audio is on disk and the archive is still in the cache.
    fetched = _fake_ami(monkeypatch, tmp_path, {"ES2002a": 1}, {})
    out = common.raw_dir("ami")
    (out / "audio/ES2002a").mkdir(parents=True)
    (out / "audio/ES2002a/ES2002a.Headset-0.wav").write_bytes(b"old")
    (out / ".headset-extracted").touch()
    orphan_archive("ami-headset.tar.gz")
    ami.fetch_ami()
    assert fetched == []
    assert clean_run("ami")["missing"] == []
