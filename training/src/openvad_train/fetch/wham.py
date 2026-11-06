"""WHAM! noise (no WSJ0 mixtures) from the official S3 download, with its tr, cv and tt splits."""

import struct
import zipfile

from openvad_train.fetch.common import cache_dir, download_ranges, raw_dir, write_manifest


# The download linked from http://wham.whisper.ai/ (also in lhotse's recipes/wham.py); the MD5 is published there.
WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
WHAM_MD5 = "d5af15645d521d3920e01954c6cd7594"
WHAM_SPLITS = ("tr", "cv", "tt")


def wav_format(header: bytes) -> dict[str, int]:
    """Sample rate, channels and bits per sample from the `fmt ` chunk of a RIFF/WAVE header."""
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE header")
    pos = 12
    while pos + 8 <= len(header):
        chunk, size = struct.unpack_from("<4sI", header, pos)
        if chunk == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", header, pos + 8)
            return {"sample_rate": rate, "channels": channels, "bits": bits}
        pos += 8 + size + size % 2
    raise ValueError("no fmt chunk in the header")


def wham_splits(z: zipfile.ZipFile) -> tuple[dict[str, int], dict[str, dict]]:
    """The audio format (the same in every split, else an error) and the files and hours per split."""
    formats, splits = set(), {}
    for split in WHAM_SPLITS:
        prefix = f"wham_noise/{split}/"
        wavs = [i for i in z.infolist() if i.filename.startswith(prefix) and i.filename.endswith(".wav")]
        with z.open(wavs[0]) as f:
            fmt = wav_format(f.read(256))
        formats.add(tuple(fmt.items()))
        bytes_per_s = fmt["sample_rate"] * fmt["channels"] * fmt["bits"] // 8
        # Headers are a few dozen bytes per file, well below the 0.01 h shown.
        splits[split] = {"files": len(wavs), "hours": round(sum(i.file_size for i in wavs) / bytes_per_s / 3600, 2)}
    if len(formats) != 1:
        raise ValueError(f"WHAM!: the splits differ in audio format: {sorted(formats)}")
    return dict(formats.pop()), splits


def fetch_wham() -> None:
    out = raw_dir("wham")
    out.mkdir(parents=True, exist_ok=True)
    archive = cache_dir() / "wham_noise.zip"
    marker = out / ".wham_noise-extracted"
    with zipfile.ZipFile(download_ranges(WHAM_URL, archive, md5=WHAM_MD5)) as z:
        fmt, splits = wham_splits(z)
        if not marker.exists():
            # The zip carries macOS Finder files (.DS_Store, ._*); they are not data.
            keep = [n for n in z.namelist() if not any(part.startswith(".") for part in n.split("/"))]
            z.extractall(out, members=keep)
            marker.touch()
    info = {
        "dataset": "wham",
        "licence": "CC BY-NC 4.0",
        "use": "tr train noise and cv dev noise (ADR 0023); tt test only, never trained on",
        "sources": [{"url": WHAM_URL, "md5": WHAM_MD5}],
        "format": fmt,
        "splits": splits,
    }
    write_manifest(out, info)
    for path in (archive, marker):
        path.unlink()
