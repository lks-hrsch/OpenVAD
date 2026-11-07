# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Paths, resumable downloads and manifests shared by the dataset fetchers."""

import concurrent.futures
import datetime
import gzip
import hashlib
import json
import os
import re
import socket
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


CHUNK = 1 << 20
MANIFEST = "manifest.json"
TIMEOUT_S = 60
# Range downloads: some mirrors cap each connection (S3 0.55 MB/s), not the line.
PIECE = 64 * CHUNK
RANGE_JOBS = 8
PIECE_TRIES = 3
PIECE_BACKOFF_S = 5  # doubles per retry


class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    # urllib copies request headers onto a redirect, which would send a bearer token to the CDN host.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_parts, new_parts = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(new.full_url)
            # A plain-http hop on the same host would send the token in clear text.
            if new_parts.netloc != old_parts.netloc or new_parts.scheme != "https":
                new.remove_header("Authorization")
        return new


urllib.request.install_opener(urllib.request.build_opener(_DropAuthOnCrossHostRedirect))


def redact(url: str) -> str:
    """The URL without its query string, which may hold a signature."""
    return url.split("?", 1)[0]


def tar_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    # No mtime: setting it costs one SMB flush (about 127 ms) per extracted file on the share.
    return tarfile.data_filter(member, path).replace(mtime=None, deep=False)  # pyright: ignore[reportArgumentType]


def check_archive_end(tar: tarfile.TarFile, archive: Path) -> None:
    """Fail if an extracted archive is damaged or ended early; `tarfile` checks neither on its own.

    A gzip archive is read to its end, which makes `gzip` check the CRC and length. A plain tar stops silently at
    a zero block, so more than the end blocks and record padding left after it means a hole; the 20480-byte limit
    is two records of 10240 bytes, the record size of Python's `tarfile` and GNU tar by default.
    This finds an early end and a hole at a member boundary, not a hole inside a member's data.
    """
    if isinstance(tar.fileobj, gzip.GzipFile):
        try:
            while tar.fileobj.read(CHUNK):
                pass
        except (gzip.BadGzipFile, EOFError) as e:
            raise RuntimeError(f"{archive}: {e}; delete it to refetch") from e
    elif (size := archive.stat().st_size) - tar.offset > 20480:
        raise RuntimeError(f"{archive}: the tar ends at byte {tar.offset} of {size}; delete it to refetch")


def data_dir() -> Path:
    return Path(os.environ.get("OPENVAD_DATA_DIR", "data")).absolute()


def raw_dir(name: str) -> Path:
    return data_dir() / "datasets/raw" / name


def cache_dir() -> Path:
    return data_dir() / "cache"


def take_share_lock(name: str) -> Path:
    """Create the lock file on the share; take over a stale one left by a dead process on this host."""
    path = cache_dir() / f"{name}.fetching"
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = re.fullmatch(r"host=(\S+) pid=(\d+)\s*", path.read_text(errors="replace"))
            # An unparsable file counts as held by another host, so it is never deleted silently.
            host, pid = (held[1], int(held[2])) if held else ("unknown", None)
            if host == socket.gethostname() and pid is not None:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    path.unlink(missing_ok=True)
                    continue
                except PermissionError:
                    pass  # the process exists under another user
            raise SystemExit(
                f"{name}: held by host {host}, pid {pid if pid is not None else 'unknown'}; "
                f"delete {path} if no fetch is running there"
            ) from None
        with os.fdopen(fd, "w") as f:
            f.write(f"host={socket.gethostname()} pid={os.getpid()}")
        return path


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _check(path: Path, md5: str | None, sha256: str | None) -> None:
    for algo, expected in (("md5", md5), ("sha256", sha256)):
        if expected is not None and (actual := file_hash(path, algo)) != expected.lower():
            raise ValueError(f"{algo} mismatch for {path.name}: expected {expected}, got {actual}")


def download(
    url: str, dest: Path, *, md5: str | None = None, sha256: str | None = None, headers: dict[str, str] | None = None
) -> Path:
    """Download `url` to `dest`, resuming a `.part` file, and verify the given hashes."""
    if dest.exists():
        try:
            _check(dest, md5, sha256)
            return dest
        except ValueError:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={**(headers or {}), **({"Range": f"bytes={have}-"} if have else {})})
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as e:
        if e.code != 416 or not have:
            raise
        # The server has no bytes past the end of the part: it is complete, or stale and restarted.
        # Without a hash, the 416's own size (`bytes */N`) is the only proof; a longer stale part must not pass.
        try:
            if md5 is None and sha256 is None and e.headers.get("Content-Range") != f"bytes */{have}":
                raise ValueError(f"{part.name} size does not match the server")
            _check(part, md5, sha256)
        except ValueError:
            part.unlink()
            return download(url, dest, md5=md5, sha256=sha256, headers=headers)
        part.rename(dest)
        return dest
    with resp:
        resumed = getattr(resp, "status", 200) == 206
        with part.open("ab" if resumed else "wb") as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())  # the rename below must not outlive the bytes
        # http.client returns b"" on a clean server close before Content-Length; keep the part to resume.
        if getattr(resp, "length", None):
            raise ConnectionError(f"{redact(url)}: connection closed with {resp.length} bytes missing; rerun to resume")
    try:
        _check(part, md5, sha256)
    except ValueError:
        part.unlink()
        raise
    part.rename(dest)
    return dest


def _fetch_piece(url: str, part: Path, start: int, end: int, headers: dict[str, str] | None = None) -> None:
    for attempt in range(PIECE_TRIES):
        try:
            req = urllib.request.Request(url, headers={**(headers or {}), "Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, part.open("r+b") as f:
                if resp.status != 206:
                    raise ConnectionError(f"{redact(url)}: server ignored the range request (status {resp.status})")
                f.seek(start)
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                if f.tell() != end:
                    raise ConnectionError(f"{redact(url)}: piece {start} ended at byte {f.tell()} of {end}")
                # Log a piece only once its bytes are on disk, or a crash leaves zeros that the log calls done.
                f.flush()
                os.fsync(f.fileno())
            return
        except OSError as e:
            # A 401/403 is an expired or revoked link: retrying cannot help, and the caller needs to see it.
            if attempt == PIECE_TRIES - 1 or (isinstance(e, urllib.error.HTTPError) and e.code in (401, 403)):
                raise
            time.sleep(PIECE_BACKOFF_S * 2**attempt)


def download_ranges(
    url: str,
    dest: Path,
    *,
    md5: str | None = None,
    sha256: str | None = None,
    headers: dict[str, str] | None = None,
    jobs: int = RANGE_JOBS,
) -> Path:
    """Download `url` to `dest` in pieces over `jobs` connections; a rerun fetches only the missing pieces.

    Without a hash only the size is checked (some mirrors publish no checksums); the archive format checks the rest.
    """
    if dest.exists():
        try:
            _check(dest, md5, sha256)
        except ValueError as e:
            raise ValueError(f"{e}; delete it to refetch") from e
        return dest
    # A ranged GET, not HEAD: a presigned URL signs the method, so HEAD is rejected.
    req = urllib.request.Request(url, headers={**(headers or {}), "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        content_range = resp.headers["Content-Range"]
        size = int(content_range.rsplit("/", 1)[1]) if content_range else int(resp.headers["Content-Length"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    # Start offsets of the finished pieces, one per line, so a rerun can resume.
    log = dest.with_suffix(dest.suffix + ".pieces")
    if not part.exists() or part.stat().st_size != size:
        log.unlink(missing_ok=True)
        with part.open("wb") as f:
            f.truncate(size)
    done = {int(line) for line in log.read_text().split()} if log.exists() else set()
    todo = [start for start in range(0, size, PIECE) if start not in done]
    with log.open("a") as log_file, concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(_fetch_piece, url, part, s, min(s + PIECE, size), headers): s for s in todo}
        failed = []
        denied = None
        try:
            for future in concurrent.futures.as_completed(futures):
                if (error := future.exception()) is not None:
                    failed.append(f"{futures[future]}: {error}")
                    if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                        denied = error
                    continue
                log_file.write(f"{futures[future]}\n")
                log_file.flush()
        except KeyboardInterrupt:
            pool.shutdown(cancel_futures=True)
            raise
    if denied is not None:
        raise denied
    if failed:
        raise ConnectionError(f"{redact(url)}: {len(failed)} pieces failed, rerun to resume; first: {failed[0]}")
    try:
        _check(part, md5, sha256)
    except ValueError:
        part.unlink()
        log.unlink()
        raise
    part.rename(dest)
    log.unlink()
    return dest


def count_tree(root: Path) -> tuple[int, int]:
    # Dotfiles (.DS_Store from Finder on SMB) and __pycache__ are not data.
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p != root / MANIFEST
        and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)
    ]
    return len(files), sum(p.stat().st_size for p in files)


def write_manifest(root: Path, info: dict) -> None:
    files, size = count_tree(root)
    manifest = {**info, "files": files, "bytes": size, "retrieved": datetime.date.today().isoformat()}
    # Write aside and rename, so a kill mid-write cannot leave a truncated manifest (a dotfile, which count_tree skips).
    tmp = root / f".{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, root / MANIFEST)


def read_manifest(root: Path) -> dict | None:
    """The manifest, `{}` if unreadable, None if there is none."""
    try:
        return json.loads((root / MANIFEST).read_text())
    except (FileNotFoundError, NotADirectoryError):
        return None
    except json.JSONDecodeError:
        return {}


def is_complete(root: Path) -> bool:
    m = read_manifest(root)
    return m is not None and (m.get("files"), m.get("bytes")) == count_tree(root)


def hf_token() -> str | None:
    """The Hugging Face token from `HF_TOKEN` or `hf auth login`, if any; it raises the rate limits."""
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return os.environ.get("HF_TOKEN") or (token_file.read_text().strip() if token_file.exists() else None)


def hf_tree(repo: str, rev: str, path: str = "", headers: dict[str, str] | None = None) -> list[dict]:
    """The recursive file listing of a Hugging Face dataset repo at a revision, optionally of one directory."""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}{'/' + path if path else ''}?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=TIMEOUT_S) as resp:
        if 'rel="next"' in (resp.headers.get("Link") or ""):
            raise RuntimeError(f"{redact(url)}: the file listing is paginated, which the fetchers do not support")
        return json.loads(resp.read().decode())


def hf_source(base: str, entry: dict) -> dict:
    """The manifest `sources` entry of a Hugging Face tree entry: its URL, and the SHA-256 if it is in LFS."""
    oid = entry.get("lfs", {}).get("oid")
    return {"url": f"{base}/{entry['path']}", **({"sha256": oid} if oid else {})}


def read_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
        return resp.read().decode()
