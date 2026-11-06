"""Helpers shared by the tests of several fetch modules."""

import io
import json
import struct
import tarfile
import urllib.error
import urllib.request
from email.message import Message

from openvad_train.fetch import common


def assert_manifest_shape(m: dict) -> None:
    """The keys every fetcher's manifest has, and a non-empty `sources` list of `{url, ...}` dicts."""
    assert {"dataset", "licence", "use", "sources", "files", "bytes", "retrieved"} <= m.keys()
    assert isinstance(m["use"], str)
    assert isinstance(m["sources"], list)
    assert m["sources"]
    assert all(isinstance(x, dict) and isinstance(x.get("url"), str) for x in m["sources"])
    assert all(set(x) <= {"url", "md5", "sha256"} for x in m["sources"])


class FakeResp:
    def __init__(self, chunks, *, status=200, length=0, headers=None):
        self.chunks, self.status, self.length = list(chunks), status, length
        self.headers = headers or Message()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _=-1):
        return self.chunks.pop(0) if self.chunks else b""


def http_error(code, content_range=None):
    headers = Message()
    if content_range is not None:
        headers["Content-Range"] = content_range
    return urllib.error.HTTPError("https://x/f", code, "err", headers, None)


def tar_with(path, name, data=b"x", mtime=0):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = mtime
    with tarfile.open(path, "w") as tar:
        tar.addfile(info, io.BytesIO(data))


def tar_bytes(name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo(f"LibriSpeech/{name}/f.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def archive_with(path, members: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def wav_header(*, rate: int, channels: int, bits: int, data: int, extra: bytes = b"") -> bytes:
    block = channels * bits // 8
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, channels, rate, rate * block, block, bits)
    body = b"WAVE" + extra + fmt + struct.pack("<4sI", b"data", data)
    return struct.pack("<4sI", b"RIFF", len(body) + data) + body


def plain_tar(path, *, hole=False):
    # Three 16 KiB members, so a hole over the second header leaves more than the 20 KiB of end padding unread.
    with tarfile.open(path, "w") as tar:
        for i in range(3):
            info = tarfile.TarInfo(f"g{i}")
            info.size = 16384
            tar.addfile(info, io.BytesIO(b"y" * info.size))
    if hole:
        raw = bytearray(path.read_bytes())
        second = 512 + 16384
        raw[second : second + 1024] = bytes(1024)
        path.write_bytes(raw)


def clean_run(name):
    """The manifest of a finished fetch, after checking it is complete and leaves no marker or cache file."""
    out = common.raw_dir(name)
    manifest = common.read_manifest(out)
    assert manifest is not None
    assert_manifest_shape(manifest)
    assert common.is_complete(out)
    assert not list(out.glob(".*"))
    assert not list(common.cache_dir().glob("*"))
    return manifest


def orphan_archive(name):
    """A cache archive that a crash after the marker left behind."""
    common.cache_dir().mkdir(parents=True, exist_ok=True)
    (common.cache_dir() / name).write_bytes(b"orphan")


def hf_server(monkeypatch, listings: dict[str, list], files: dict[str, bytes]):
    """Fake urlopen for the Hugging Face API (by listing URL) and resolve URLs; returns the requests it served."""
    served = []

    def fake_urlopen(req, **_):
        served.append(req)
        if req.full_url in listings:
            return FakeResp([json.dumps(listings[req.full_url]).encode()])
        return FakeResp([files[req.full_url]])

    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    return served
