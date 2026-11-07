# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `construct`: each plan row becomes stored audio and labels, a pure function of the row."""

import argparse
import asyncio
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import soundfile as sf

from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.codecs import DecodeError, decode_16k, delay_ms, encoder_of, preflight, roundtrip
from openvad_train.dataset.config import DEMAND_CATEGORIES, FULL, SLICE, BuildConfig, dataset_root, work_dir
from openvad_train.dataset.cutting import rebase
from openvad_train.dataset.labels import SAMPLES_PER_FRAME, SAMPLES_PER_MS, apply_gap_rule, rasterise
from openvad_train.dataset.mix import SAMPLE_RATE, dither, loop_slice, mix_at_snr, place, protect_peak
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.plan import PLAN_SCHEMA
from openvad_train.fetch.common import (
    cache_dir,
    data_dir,
    file_hash,
    read_manifest,
    take_share_lock,
    write_manifest,
)


log = logging.getLogger(__name__)


_SPACE_MARGIN_BYTES = 50_000_000_000
_MAX_FAILED_SHARE = 0.05
# Words quieter than this are digital silence with labels on it: a real SWC "word" of 3.39 s sat at -83.6 dBFS.
_MIN_WORD_DBFS = -70.0
_CONDITIONS_BUILT = {"clean", *DEMAND_CATEGORIES}
_CODECS_BUILT = {"none", "g722"}


class OutOfSpace(Exception):  # noqa: N818 (the name is the plan's interface)
    """The share has too little free space for the next shard; a re-run continues."""


class PlanMismatch(Exception):  # noqa: N818 (the name is the plan's interface)
    """The plan, its version or the ffmpeg differ from what the shards already built came from."""


def _legs_not_built(row: dict) -> list[str]:
    """The fields of `row` that ask for a leg construction does not have yet: Task 18 and Task 19 add them."""
    asked = {
        "room": row["room"],
        "handset": row["handset"] is not None,
        "agc": row["agc"] is not None,
        "clip_drive_db": row["clip_drive_db"] is not None,
        "suppression": row["suppression"] is not None,
        "transients": bool(row["transients"]),
        "segments": bool(row["segments"]),
        "band": row["band"] != "wide",
        "codec": row["codec"] not in _CODECS_BUILT,
        "condition": row["condition"] not in _CONDITIONS_BUILT,
    }
    return [field for field, wanted in asked.items() if wanted]


def _read_noise(path: Path, offset_n: int, n: int) -> np.ndarray:
    """What `loop_slice` gives on the whole file, reading only the part it needs unless the slice wraps around."""
    with sf.SoundFile(path) as f:
        if f.samplerate != SAMPLE_RATE:
            raise ValueError(f"{path}: noise must be 16 kHz, got {f.samplerate}")
        if offset_n + n <= f.frames:
            f.seek(offset_n)
            return f.read(n, dtype="float32")
        return loop_slice(f.read(dtype="float32"), offset_n, n)


def _noise_of(row: dict, raw: Path) -> np.ndarray:
    files, offsets = row["noise_source_ids"], row["noise_offsets"]
    if len(files) != 1 or len(offsets) != 1:
        raise ValueError(f"{row['sample_id']}: a {row['condition']} row needs one noise, got {files} at {offsets}")
    return _read_noise(raw / files[0], offsets[0], row["total_n"])


def codec_delay_ms(row: dict) -> int:
    """The whole milliseconds by which the row's stored audio lags its input, which its labels are shifted by."""
    return 0 if row["codec"] == "none" else round(delay_ms(row["codec"]))


def construct_sample(row: dict, raw: Path) -> tuple[bytes, np.ndarray]:
    """16-bit 16 kHz mono FLAC and one label per 10 ms frame for a plan row.

    `raw` is the root the row's paths are relative to, `$OPENVAD_DATA_DIR` in a build. The only draw made here is
    the dither of a clean row: its `total_n` normals from `default_rng(row["seed"])`. Bytes then depend only on the
    row, its source files and the ffmpeg version.

    Labels come from the audio produced, never from file metadata. `DecodeError` means this clip cannot be built: a
    damaged file, a decode further than one 10 ms frame from `source_n`, or words under -70 dBFS RMS. A decode can
    run long only by `decode_16k` breaking its contract, because it stops at `source_n`. `NotImplementedError` names
    the fields of a row asking for a leg that does not exist yet, and nothing is skipped silently.
    """
    missing = _legs_not_built(row)
    if missing:
        raise NotImplementedError(f"{row['sample_id']} asks for legs not built yet: {', '.join(missing)}")
    offset_n, total_n, source_n = row["speech_offset_n"], row["total_n"], row["source_n"]
    if offset_n % SAMPLES_PER_MS:
        raise ValueError(f"{row['sample_id']}: padding offset {offset_n} is off the millisecond grid labels use")

    path = str(raw / row["source_path"])
    speech = decode_16k(path, row["source_offset_n"] / SAMPLE_RATE, source_n / SAMPLE_RATE)
    if abs(len(speech) - source_n) > SAMPLES_PER_FRAME:
        raise DecodeError(f"{path}: decoded {len(speech)} samples for source_n {source_n}, so its labels would drift")
    # Words the audio ends before are no speech: an MP3's overstated length leaves up to a frame of them.
    end_ms = -(-len(speech) // SAMPLES_PER_MS)
    speech_ms, mask_ms = rebase(row["speech_ms"], 0, end_ms), rebase(row["mask_ms"], 0, end_ms)

    x = place(speech, total_n, offset_n)
    # Audio decisions use the word union at G = 0, so a change of G leaves the audio as it is.
    active = np.zeros(total_n, bool)
    for start, end in apply_gap_rule(speech_ms, 0):
        active[offset_n + start * SAMPLES_PER_MS : offset_n + end * SAMPLES_PER_MS] = True
    if not active.any() or np.sqrt(np.mean(np.square(x[active], dtype=np.float64))) < 10 ** (_MIN_WORD_DBFS / 20):
        raise DecodeError(f"{path}: silent under the words of {row['sample_id']}: under {_MIN_WORD_DBFS} dBFS")
    if row["condition"] == "clean":
        x = x + dither(total_n, row["dither_dbfs"], np.random.default_rng(row["seed"]))
    else:
        noise = _noise_of(row, raw)
        if not noise[active].any():
            raise DecodeError(f"{row['noise_source_ids'][0]}: silent under the words of {row['sample_id']}")
        x = mix_at_snr(x, noise, active, row["snr_db"])
    x = protect_peak(x)
    if row["codec"] != "none":
        x = roundtrip(x, row["codec"], round(row["bitrate"] * 1000))  # the plan's bitrates are in kbit/s

    shift_ms = offset_n // SAMPLES_PER_MS + codec_delay_ms(row)
    labels = rasterise(apply_gap_rule(speech_ms, row["gap_ms"]), mask_ms, len(x), shift_ms)
    flac = io.BytesIO()
    sf.write(flac, x, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return flac.getvalue(), labels


class ShardResult(NamedTuple):
    n_ok: int
    n_failed: int
    sha256: str


def _construct_with_a_retry(row: dict, raw: Path) -> tuple[bytes, np.ndarray]:
    """`construct_sample`, tried once more on `DecodeError` so a transient failure does not drop the row for good.

    After the second failure the source must still give one byte: an `OSError` here means the share is failing, not
    the clip, and aborts the run. ffmpeg reports both the same way.
    """
    try:
        return construct_sample(row, raw)
    except DecodeError as e:
        log.info("%s: retrying after %s", row["sample_id"], e)
    try:
        return construct_sample(row, raw)
    except DecodeError:
        with (raw / row["source_path"]).open("rb") as f:
            f.read(1)
        raise


def _write_tar(rows: list[dict], tar_path: Path, failures_path: Path, raw: Path) -> tuple[list[dict], int]:
    """The uncompressed tar of `rows`, and the built rows with their labels. A clip that cannot be built goes to the
    failures file.
    """
    built, n_failed = [], 0
    with tarfile.open(tar_path, "w") as tar, failures_path.open("w") as failures:
        for row in rows:
            try:
                flac, labels = _construct_with_a_retry(row, raw)
            except DecodeError as e:
                log.warning("%s: %s", row["sample_id"], e)
                failure = {"sample_id": row["sample_id"], "source_path": row["source_path"], "error": str(e)}
                failures.write(json.dumps(failure) + "\n")
                n_failed += 1
                continue
            member = tarfile.TarInfo(f"{row['sample_id']}.flac")
            member.size = len(flac)
            tar.addfile(member, io.BytesIO(flac))
            built.append(row | {"labels": labels.tolist(), "codec_delay_ms": codec_delay_ms(row)})
    return built, n_failed


def _verify_stage(tar_path: Path, fragment_path: Path, n_ok: int) -> None:
    """Read the stage back the way training will: every pointer through `load_audio`, against its labels."""
    fragment = asyncio.run(read_parquet(fragment_path))
    if fragment.height != n_ok:
        raise RuntimeError(f"{fragment_path} holds {fragment.height} rows, expected {n_ok}")
    for row in fragment.iter_rows(named=True):
        audio = load_audio(row | {"audio_filepath": tar_path.name}, tar_path.parent)
        if len(audio) != row["total_n"] or round(len(audio) / 160) != len(row["labels"]):
            raise RuntimeError(
                f"{row['sample_id']}: {len(audio)} samples and {len(row['labels'])} labels, total_n {row['total_n']}"
            )


def _copy_to_share(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)
    if (copied := dst.stat().st_size) != (size := src.stat().st_size):
        raise OSError(f"{dst}: copied {copied} of {size} bytes")
    # On the server before the rename, so a crash cannot leave the final name on data still in a client cache.
    with dst.open("rb+") as f:
        os.fsync(f.fileno())


def _publish_shard(moves: list[tuple[Path, Path]]) -> None:
    """Copy each stage file to a `.tmp` beside its destination, then rename them in order; the last is the fragment,
    so a fragment means a finished shard. A kill at any point leaves only `.tmp` names, or a tar with no fragment.
    """
    # An old fragment must not vouch for a tar about to be replaced.
    moves[-1][1].unlink(missing_ok=True)
    tmps = [dst.with_name(dst.name + ".tmp") for _, dst in moves]
    try:
        for (src, _), tmp in zip(moves, tmps, strict=True):
            _copy_to_share(src, tmp)
        for (_, dst), tmp in zip(moves, tmps, strict=True):
            os.replace(tmp, dst)
    finally:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)


def write_shard(rows: list[dict], tar_path: Path, fragment_path: Path, raw: Path) -> ShardResult:
    """Build one shard in a local temporary directory, verify it, and publish it to `tar_path` and `fragment_path`,
    with `<tar>.sha256` and `<fragment stem>.failures.jsonl` beside them. `raw` is the root of `construct_sample`,
    and `tar_path` must lie under it because the fragment's pointers are relative to it.

    A shard with no row built or over `_MAX_FAILED_SHARE` of its rows failed raises without publishing, so a re-run
    rebuilds it: that many failures point at the setup, and a published shard would never be looked at again.
    """
    audio_filepath = tar_path.relative_to(raw).as_posix()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openvad-shard-") as stage_name:
        stage = Path(stage_name)
        stage_tar, stage_fragment = stage / "shard.tar", stage / "fragment.parquet"
        stage_failures, stage_sidecar = stage / "failures.jsonl", stage / "sha256"
        built, n_failed = _write_tar(rows, stage_tar, stage_failures, raw)
        if not built or n_failed / len(rows) > _MAX_FAILED_SHARE:
            first = stage_failures.read_text().partition("\n")[0]
            raise RuntimeError(f"{tar_path}: {n_failed} of {len(rows)} rows failed, not published; first: {first}")
        # Offsets come from parsing the tar, so a pointer holds what any reader of this tar will find there.
        with tarfile.open(stage_tar, "r:") as tar:
            members = tar.getmembers()
        pointed = [
            row | {"audio_filepath": audio_filepath, "byte_offset": m.offset_data, "byte_length": m.size}
            for row, m in zip(built, members, strict=True)
        ]
        frame = pl.DataFrame(pointed, schema=PLAN_SCHEMA)
        asyncio.run(write_parquet({stage_fragment: frame.lazy()}))
        _verify_stage(stage_tar, stage_fragment, len(built))
        sha256 = file_hash(stage_tar, "sha256")
        stage_sidecar.write_text(f"{sha256}  {tar_path.name}\n")
        _publish_shard(
            [
                (stage_tar, tar_path),
                (stage_sidecar, tar_path.with_name(tar_path.name + ".sha256")),
                (stage_failures, fragment_path.with_suffix(".failures.jsonl")),
                (stage_fragment, fragment_path),
            ]
        )
    return ShardResult(len(built), n_failed, sha256)


def _is_finished(tar_path: Path, fragment_path: Path) -> bool:
    return fragment_path.exists() and tar_path.exists() and tar_path.with_name(tar_path.name + ".sha256").exists()


def _check_build_state(cfg: BuildConfig, plan_manifest: dict, plan_path: Path, ffmpeg: dict) -> None:
    """Refuse a run that would mix shards of two plans or two ffmpegs, whose outputs are bytes of different builds."""
    for key, value in (("plan_version", cfg.plan_version), ("global_seed", cfg.global_seed)):
        if plan_manifest.get(key) != value:
            raise PlanMismatch(
                f"the plan has {key} {plan_manifest.get(key)!r}, the configuration {value!r}: rerun plan"
            )
    state = {
        "stage": "construct",
        "config": cfg.name,
        "plan_version": cfg.plan_version,
        "global_seed": cfg.global_seed,
        "plan_sha256": file_hash(plan_path, "sha256"),
        "ffmpeg": ffmpeg,
    }
    fragments = work_dir(cfg) / "fragments"
    built_from = read_manifest(fragments)
    if built_from is None:
        if next(fragments.rglob("*.parquet"), None) is not None:
            raise PlanMismatch(f"{fragments} holds shards and no manifest: delete its shards too to rebuild")
        fragments.mkdir(parents=True, exist_ok=True)
        write_manifest(fragments, state)
        return
    changed = [key for key, value in state.items() if built_from.get(key) != value]
    if changed:
        raise PlanMismatch(
            f"{', '.join(changed)} changed since {fragments} was started: delete its shards and manifest to rebuild"
        )


def _report_shards(done: set[Future], shards: dict[Future, str]) -> None:
    """Log finished shards; `result()` re-raises what a worker raised, which aborts the run."""
    for future in done:
        result = future.result()
        log.info("%s: %d built, %d failed, sha256 %s", shards.pop(future), result.n_ok, result.n_failed, result.sha256)


def _build_shards(cfg: BuildConfig, plan: pl.DataFrame, workers: int) -> None:
    """Submit unfinished shards in the plan's rank order. On a failure nothing new starts and the shards running
    finish, because leaving the pool's context waits for them. A stop for space leaves a clean prefix, but with more
    than one worker a shard after a failed one can finish and leave a hole, which a re-run fills: it goes by what is
    finished, not by position.
    """
    raw, data, fragments = data_dir(), dataset_root(cfg) / "data", work_dir(cfg) / "fragments"
    data.mkdir(parents=True, exist_ok=True)
    running: dict[Future, str] = {}
    # Spawn, because Polars' thread pool does not survive a fork.
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        for (shard,), frame in plan.group_by("shard", maintain_order=True):
            tar_path, fragment_path = data / f"{shard}.tar", fragments / f"{shard}.parquet"
            if _is_finished(tar_path, fragment_path):
                continue
            if len(running) >= workers:
                _report_shards(wait(running, return_when=FIRST_COMPLETED).done, running)
            estimate = float(frame["duration"].sum()) / 3600 * cfg.flac_mb_per_hour * 1e6
            free = shutil.disk_usage(data).free
            if free < estimate + _SPACE_MARGIN_BYTES:
                raise OutOfSpace(
                    f"{data}: {free / 1e9:.1f} GB free, {shard} needs its {estimate / 1e9:.1f} GB plus 50 GB"
                )
            running[pool.submit(write_shard, frame.to_dicts(), tar_path, fragment_path, raw)] = shard
        _report_shards(wait(running).done, running)


def _record_failures(cfg: BuildConfig, plan: pl.DataFrame) -> None:
    """Total the failures files of every shard, so a re-run that built nothing still reports the rows left out."""
    fragments = work_dir(cfg) / "fragments"
    shards = plan["shard"].unique()
    n_failed = sum(len((fragments / f"{shard}.failures.jsonl").read_text().splitlines()) for shard in shards)
    write_manifest(fragments, (read_manifest(fragments) or {}) | {"n_failed": n_failed})
    if n_failed:
        log.warning("%d rows failed and were left out; see the .failures.jsonl files under %s", n_failed, fragments)


def run_construct(cfg: BuildConfig, workers: int = 11) -> None:
    """Build every unfinished shard of the plan's `augmented` rows; a re-run continues where one stopped.

    Raises `SystemExit` when another run holds the lock, `OutOfSpace` and `PlanMismatch` as named, and whatever a
    worker raised for a broken setup. A clip that cannot be built does not stop the run, within the limit of
    `write_shard`; the total left out goes in `fragments/manifest.json` as `n_failed` and in a warning.
    """
    cache_dir().mkdir(parents=True, exist_ok=True)
    lock = take_share_lock(f"construct-{cfg.name}")
    try:
        plan_dir = work_dir(cfg) / "plan"
        manifest = read_manifest(plan_dir)
        if manifest is None:
            raise FileNotFoundError(f"{plan_dir} has no manifest: run the plan stage first")
        # shortcut: the parent holds the whole plan, about 26 GB at full scale by Task 6's estimate; read it per shard
        # if the workstation cannot hold that.
        plan = asyncio.run(read_parquet(plan_dir / "plan.parquet")).filter(pl.col("treatment") == "augmented")
        ffmpeg = preflight({encoder_of(c) for c in plan["codec"].unique() if c != "none"})
        _check_build_state(cfg, manifest, plan_dir / "plan.parquet", ffmpeg)
        _build_shards(cfg, plan, workers)
        _record_failures(cfg, plan)
    finally:
        lock.unlink(missing_ok=True)


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset construct", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    parser.add_argument("--workers", type=int, default=11, help="worker processes, one per shard")
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_construct({"slice": SLICE, "full": FULL}[ns.config], ns.workers)
