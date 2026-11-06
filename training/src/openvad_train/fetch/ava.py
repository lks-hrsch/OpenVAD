"""AVA-Speech labels, and the labelled audio span cut from the AVA S3 mirror with ffmpeg."""

import concurrent.futures
import csv
import shutil
import subprocess
from pathlib import Path

from openvad_train.fetch.common import TIMEOUT_S, download, file_hash, raw_dir, write_manifest


# ffmpeg and ffprobe take the network read/write timeout in microseconds.
RW_TIMEOUT_US = str(TIMEOUT_S * 1_000_000)

AVA_LABELS = "https://research.google.com/ava/download/ava_speech_labels_v1.csv"
AVA_FILE_LIST = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"
AVA_LABELS_SHA256 = "9936403f9abcb9e768d1b6889ceeadb6d6797d70c39cf9cf1062cdc2f3577377"
AVA_FILE_LIST_SHA256 = "3cc1cd1262ce727ef367c692a5505ee643bcf32dc930be90d663514fb52bbec2"
AVA_VIDEOS = "https://s3.amazonaws.com/ava-dataset/trainval"
# The labels cover the 900 s to 1800 s span of each video.
AVA_START_S = 900
AVA_DURATION_S = 900
AVA_JOBS = 4


def ava_video_ids(labels_csv: str) -> list[str]:
    return sorted({row[0] for row in csv.reader(labels_csv.splitlines()) if row})


def ava_mirror_map(ids: list[str], file_list: str) -> tuple[dict[str, str], list[str]]:
    by_id = {name.rpartition(".")[0]: name for name in file_list.split()}
    return {i: by_id[i] for i in ids if i in by_id}, sorted(i for i in ids if i not in by_id)


def ava_ffmpeg_cmd(url: str, out: Path) -> list[str]:
    """Cut the labelled span as FLAC, keeping the source sample rate and channels."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-rw_timeout", RW_TIMEOUT_US]
    cmd = [*cmd, "-ss", str(AVA_START_S), "-i", url]
    return [*cmd, "-t", str(AVA_DURATION_S), "-vn", "-c:a", "flac", str(out)]


def require_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("ffmpeg not found on PATH (need both ffmpeg and ffprobe); install it and retry")


class ShortOutputError(Exception):
    """An AVA cut that is shorter than the labelled span."""


def _cause(exc: Exception) -> str:
    if isinstance(exc, ShortOutputError):
        return str(exc)
    if isinstance(exc, subprocess.CalledProcessError):
        return f"{exc.cmd[0]} exited {exc.returncode}: {' '.join((exc.stderr or '').split())}"
    return repr(exc)


def _require_full(path: Path) -> None:
    """ffmpeg exits 0 when the HTTP input ends early, so check the local file's duration."""
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    text = subprocess.run(probe, check=True, capture_output=True, text=True).stdout.strip()
    seconds = float(text) if text not in ("", "N/A") else 0.0
    if seconds < AVA_DURATION_S - 1:
        raise ShortOutputError(f"short output: {seconds:.1f} s")


def _ava_cut(video_id: str, name: str, audio: Path) -> tuple[str, float]:
    """Cut one video; return its audio sha256 and the container start_time."""
    url = f"{AVA_VIDEOS}/{name}"
    probe = ["ffprobe", "-v", "error", "-rw_timeout", RW_TIMEOUT_US]
    probe = [*probe, "-show_entries", "format=start_time", "-of", "csv=p=0", url]
    start = subprocess.run(probe, check=True, capture_output=True, text=True).stdout.strip()
    final = audio / f"{video_id}.flac"
    if final.exists():
        _require_full(final)  # never deleted: a short finished file is reported, not removed
    else:
        part = audio / f"{video_id}.part.flac"
        try:
            subprocess.run(ava_ffmpeg_cmd(url, part), check=True, capture_output=True, text=True)
            _require_full(part)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.rename(final)
    return file_hash(final, "sha256"), float(start) if start not in ("", "N/A") else 0.0


def fetch_ava() -> None:
    require_ffmpeg()
    out = raw_dir("ava-speech")
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    # Leftovers of a failed run would be counted into the manifest.
    for part in audio.glob("*.part.flac"):
        part.unlink()
    labels = download(AVA_LABELS, out / Path(AVA_LABELS).name, sha256=AVA_LABELS_SHA256)
    file_list = download(AVA_FILE_LIST, out / Path(AVA_FILE_LIST).name, sha256=AVA_FILE_LIST_SHA256)
    found, missing = ava_mirror_map(ava_video_ids(labels.read_text()), file_list.read_text())
    (out / "missing.txt").write_text("".join(f"{i}\n" for i in missing))
    with concurrent.futures.ThreadPoolExecutor(AVA_JOBS) as pool:
        futures = {i: pool.submit(_ava_cut, i, name, audio) for i, name in found.items()}
        try:
            concurrent.futures.wait(futures.values())
        except KeyboardInterrupt:
            pool.shutdown(cancel_futures=True)
            raise
    results: dict[str, tuple[str, float]] = {}
    failed: dict[str, str] = {}
    for video_id, future in futures.items():
        try:
            results[video_id] = future.result()
        except (subprocess.CalledProcessError, OSError, ShortOutputError) as exc:
            failed[video_id] = _cause(exc)
    if failed:
        lines = "\n".join(f"  {i}: {cause}" for i, cause in failed.items())
        raise RuntimeError(f"AVA-Speech: ffmpeg/ffprobe failed for {len(failed)} videos:\n{lines}")
    ffmpeg_version = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    info = {
        "dataset": "ava-speech",
        "use": "benchmark only (ADR 0020)",
        "ffmpeg": ffmpeg_version.stdout.splitlines()[0],
        "licence": "labels CC BY 4.0; audio copyright of the YouTube uploaders, not redistributable",
        "offset_s": AVA_START_S,
        "offset_note": f"t_audio = t_label - {AVA_START_S}",
        "videos": len(found),
        "missing": len(missing),
        "sources": [
            {"url": AVA_LABELS, "sha256": file_hash(labels, "sha256")},
            {"url": AVA_FILE_LIST, "sha256": file_hash(file_list, "sha256")},
        ],
        "audio": {
            i: {"sha256": sha, **({"start_time": start} if start else {})} for i, (sha, start) in results.items()
        },
    }
    write_manifest(out, info)
