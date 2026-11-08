# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `publish`: finished shards and as-is rows become the split files Hugging Face loads, with pointers only."""

import argparse
import asyncio
import hashlib
import logging
import math
import subprocess
from pathlib import Path

import polars as pl

from openvad_rs import half_band_taps
from openvad_train.dataset.codecs import _delay_table
from openvad_train.dataset.config import FULL, SLICE, BuildConfig, dataset_root, work_dir
from openvad_train.dataset.construct import PlanMismatch
from openvad_train.dataset.labels import apply_gap_rule, rasterise
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.plan import PLAN_SCHEMA
from openvad_train.dataset.sources.librivad import VAL_SOURCE
from openvad_train.fetch.common import cache_dir, file_hash, raw_dir, read_manifest, take_share_lock, write_manifest
from openvad_train.fetch.musan import musan_licences


log = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
# Plan-only columns: what construction read, none of it needed to point at audio. The split is the file's name.
PLAN_ONLY = (
    "clip_id",
    "shard",
    "source_path",
    "source_offset_n",
    "source_n",
    "speech_offset_n",
    "total_n",
    "speech_ms",
    "mask_ms",
    "split",
)
INDEX_COLUMNS = ["sample_id", "split", "in_small", "source", "treatment", "audio_filepath"]
_LABEL_ROWS = 10_000
# LibriVAD `medium` stays out of Small (ADR 0017).
_NOT_IN_SMALL = "librivad"

_README = """---
pretty_name: OpenVAD v1
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---

# OpenVAD v1

Pointers, frame labels and metadata for voice activity detection in noisy telephone audio.
No audio is decoded here: `audio_filepath` is relative to `$OPENVAD_DATA_DIR`, and `audio_format` says how to read it
(`flac_tar` with `byte_offset` and `byte_length`, `file` with `offset` and `duration`, or `callhome_parquet` with
`row_in_file`). Labels are one per 10 ms frame: 0 non-speech, 1 speech, 2 masked.

Small is `ds["train"].filter(lambda r: r["in_small"])`, once `in_small` is set. Licences are per row, in `licence`.
Build provenance is in `manifest.json`.
"""


def _git_state(repo: Path) -> tuple[str | None, bool | None]:
    """`HEAD` of the repository around `repo` and whether the working tree differs from it (ignored files do not
    count), so a manifest never names a commit that could not have written it. Both are None without git.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def _as_is_with_labels(rows: pl.DataFrame) -> pl.DataFrame:
    """Labels of the as-is rows, which `construct` never sees: the plan's intervals at its G, from the start of the
    source with no shift. They are built `_LABEL_ROWS` rows at a time: the Python lists they pass through take about
    8 bytes a label against the 1 byte the column holds, which at hundreds of thousands of rows would be tens of GB.
    """
    chunks = []
    for start in range(0, rows.height, _LABEL_ROWS):
        labels = [
            rasterise(
                apply_gap_rule([(s, e) for s, e in r["speech_ms"]], r["gap_ms"]),
                [(s, e) for s, e in r["mask_ms"]],
                r["total_n"],
            ).tolist()
            for r in rows.slice(start, _LABEL_ROWS).iter_rows(named=True)
        ]
        chunks.append(pl.Series("labels", labels, dtype=PLAN_SCHEMA["labels"]))
    return rows.with_columns(
        labels=pl.concat(chunks) if chunks else pl.Series("labels", [], dtype=PLAN_SCHEMA["labels"])
    )


async def _read_all(paths: list[Path]) -> list[pl.DataFrame]:
    return await asyncio.gather(*(read_parquet(p) for p in paths))


def run_publish(cfg: BuildConfig) -> None:
    """Write `train.parquet`, `validation.parquet`, `test.parquet`, `index.parquet`, `README.md` and `manifest.json`
    under the dataset root, from the plan's as-is rows and the fragment of every finished shard.

    A shard is finished when its fragment exists, as in `construct`; `*.tmp` files, failures files and the fragments
    manifest are not fragments. A tar without a fragment is an orphan: it is not published, and it is reported in the
    manifest and as a warning. The build is `interim` while a planned shard has no fragment. Raw manifests the build
    reads must exist, because their hashes go in the manifest.
    """
    cache_dir().mkdir(parents=True, exist_ok=True)
    # The construct lock, because the fragments are what construct writes.
    lock = take_share_lock(f"construct-{cfg.name}")
    try:
        _publish(cfg)
    finally:
        lock.unlink(missing_ok=True)


def _check_fragments_belong_to_the_plan(
    cfg: BuildConfig, plan_manifest: dict, plan_path: Path, fragments: Path, finished: set[str], planned: set[str]
) -> None:
    """Raise `PlanMismatch` before anything is written if the fragments came from another plan: a re-plan leaves old
    shards whose rows and seeds the plan no longer has, and publishing them would read as complete.
    """
    for key, value in (("plan_version", cfg.plan_version), ("global_seed", cfg.global_seed)):
        if plan_manifest.get(key) != value:
            raise PlanMismatch(f"the plan has {key} {plan_manifest.get(key)!r}, the configuration {value!r}")
    if not finished:
        return
    built_from = read_manifest(fragments)
    if built_from is None:
        raise PlanMismatch(f"{fragments} holds shards and no manifest: rerun construct")
    expected = {
        **{k: plan_manifest[k] for k in ("plan_version", "global_seed")},
        "plan_sha256": file_hash(plan_path, "sha256"),
    }
    if changed := [key for key, value in expected.items() if built_from.get(key) != value]:
        raise PlanMismatch(
            f"{', '.join(changed)} differ between the plan and the shards in {fragments}: rerun construct"
        )
    if outside := sorted(finished - planned):
        raise PlanMismatch(f"fragments for shards the plan does not have: {', '.join(outside)}")


def _publish(cfg: BuildConfig) -> None:
    root, work = dataset_root(cfg), work_dir(cfg)
    plan_manifest = read_manifest(work / "plan")
    if plan_manifest is None:
        raise FileNotFoundError(f"{work / 'plan'} has no manifest: run the plan stage first")
    plan = asyncio.run(read_parquet(work / "plan/plan.parquet"))

    fragments = work / "fragments"
    fragment_paths = sorted(fragments.rglob("*.parquet"))
    finished = {p.relative_to(fragments).with_suffix("").as_posix() for p in fragment_paths}
    planned = set(plan.filter(pl.col("treatment") == "augmented")["shard"].unique())
    _check_fragments_belong_to_the_plan(cfg, plan_manifest, work / "plan/plan.parquet", fragments, finished, planned)
    missing = sorted(planned - finished)
    data = root / "data"
    orphans = sorted(p.relative_to(data).with_suffix("").as_posix() for p in data.rglob("*.tar"))
    orphans = [shard for shard in orphans if shard not in finished]
    if orphans:
        log.warning("%d tars have no fragment and are not published: %s", len(orphans), ", ".join(orphans))
    if missing:
        log.warning("%d of %d planned shards are missing: the build is interim", len(missing), len(planned))

    # shortcut: every row is held in memory, as in `construct`; stream by split if the full build does not fit.
    built = asyncio.run(_read_all(fragment_paths))
    as_is = _as_is_with_labels(plan.filter(pl.col("treatment") == "as_is"))
    rows = pl.concat([*built, as_is]) if built else as_is
    if rows.is_empty():
        # Writing would replace a good set with empty files, which `verify` would then have to catch.
        raise ValueError(f"{fragments} has no fragment and the plan no as-is row: nothing to publish")
    if rows["sample_id"].n_unique() != rows.height:
        raise ValueError("a sample id is repeated across the fragments and the as-is rows")
    rows = rows.sort("rank", "sample_id")

    sources = set(rows["source"]) | set(cfg.noise)
    # `librivad` rows can read the validation split of `librivad-val`, whose own manifest would otherwise go unrecorded.
    if rows["source_path"].str.starts_with(f"datasets/raw/{VAL_SOURCE}/").any():
        sources.add(VAL_SOURCE)
    if _rir_ids(rows):
        sources.add("rirs")
    talkers = _talker_paths(rows)
    sources |= {path.split("/")[2] for path in talkers}
    sources = sorted(sources)
    manifest_paths = {name: raw_dir(name) / "manifest.json" for name in sources}
    if absent := [name for name, path in manifest_paths.items() if not path.exists()]:
        raise FileNotFoundError(f"no raw manifest for {', '.join(absent)}: fetch them first")

    written = {
        root / f"{split}.parquet": rows.filter(pl.col("split") == split).drop(PLAN_ONLY).lazy() for split in SPLITS
    }
    written[root / "index.parquet"] = rows.select(INDEX_COLUMNS).lazy()
    root.mkdir(parents=True, exist_ok=True)
    # A kill between the Parquet files and the new manifest must not leave the old "complete" over new rows.
    (root / "manifest.json").unlink(missing_ok=True)
    asyncio.run(write_parquet(written))
    (root / "README.md").write_text(_README)

    fragments_manifest = read_manifest(fragments) or {}
    commit, dirty = _git_state(Path(__file__).parent)
    write_manifest(
        root,
        {
            "stage": "publish",
            "config": cfg.name,
            "state": "interim" if missing else "complete",
            "plan_version": cfg.plan_version,
            "global_seed": cfg.global_seed,
            "commit": commit,
            "commit_dirty": dirty,
            "ffmpeg": fragments_manifest.get("ffmpeg"),
            # Little-endian float32 bytes, so the hash does not depend on the host's byte order.
            "resampler_sha256": hashlib.sha256(half_band_taps().astype("<f4").tobytes()).hexdigest(),
            "delays": _delay_table(),
            "licences": {
                s: sorted(rows.filter(pl.col("source") == s)["licence"].unique()) for s in sorted(set(rows["source"]))
            }
            | _noise_licences(rows)
            | _rir_licences(rows)
            | _talker_licences(work, talkers),
            "raw_sources": {name: file_hash(path, "sha256") for name, path in manifest_paths.items()},
            "samples": {split: rows.filter(pl.col("split") == split).height for split in SPLITS},
            "planned_shards": len(planned),
            "missing_shards": missing,
            "orphan_tars": orphans,
            "n_failed": fragments_manifest.get("n_failed"),
        },
    )
    log.info("published %d samples to %s (%s)", rows.height, root, "interim" if missing else "complete")


# The conditions whose `noise_source_ids` are speech from reserved speakers, not a noise file.
_TALKER_CONDITIONS = ("babble", "ssn")


def _rir_ids(rows: pl.DataFrame) -> set[str]:
    """Every RIR a row reads: its room's, and those of its competing-talker segments."""
    with_segments = rows.filter(pl.col("segments").list.len() > 0)["segments"]
    return {*rows["rir_id"].drop_nulls(), *(s["rir_id"] for segments in with_segments for s in segments)} - {None}


def _noise_licences(rows: pl.DataFrame) -> dict[str, list[str]]:
    """`noise/<set>` to its licences, for every noise set a row's noise files come from (DEMAND, WHAM! and MUSAN).

    The set is the directory under `datasets/raw/` that the path points into. DEMAND and WHAM! have one licence in
    their raw manifest; MUSAN's differ per file, so its licences are those of the files the rows use.
    """
    ids = rows.filter(~pl.col("condition").is_in(_TALKER_CONDITIONS))["noise_source_ids"].explode().drop_nulls()
    paths = sorted({i.partition("#")[0] for i in ids})
    licences: dict[str, list[str]] = {}
    for name in sorted({path.split("/")[2] for path in paths}):
        if name == "musan":
            table = musan_licences()
            used = {table[path.rpartition("/")[2].removesuffix(".wav")] for path in paths if path.split("/")[2] == name}
            licences[f"noise/{name}"] = sorted(used)
        elif licence := (read_manifest(raw_dir(name)) or {}).get("licence"):
            licences[f"noise/{name}"] = [licence]
        else:
            raise ValueError(f"the {name} manifest records no licence")
    return licences


def _talker_paths(rows: pl.DataFrame) -> set[str]:
    """The files of the reserved speakers that babble, speech-shaped noise and competing talkers read."""
    noise = rows.filter(pl.col("condition").is_in(_TALKER_CONDITIONS))["noise_source_ids"].explode().drop_nulls()
    segments = rows.filter(pl.col("segments").list.len() > 0)["segments"]
    return {*noise, *(s["source_path"] for segs in segments for s in segs)}


def _talker_licences(work: Path, paths: set[str]) -> dict[str, list[str]]:
    """`talkers/<source>` to the licences of the reserved clips whose speech the rows use, from the index stage.

    A source's licence can differ by clip (SWC by folder), so it is looked up per clip, not per source.
    """
    licences: dict[str, list[str]] = {}
    for name in sorted({path.split("/")[2] for path in paths}):
        index = asyncio.run(read_parquet(work / "index" / f"{name}.parquet", ["path", "licence"]))
        used = {p for p in paths if p.split("/")[2] == name}
        found = index.filter(pl.col("path").is_in(used))
        if missing := used - set(found["path"]):
            raise ValueError(f"{len(missing)} talker clips are not in the {name} index, first: {sorted(missing)[0]}")
        licences[f"talkers/{name}"] = sorted(found["licence"].unique())
    return licences


def _rir_licences(rows: pl.DataFrame) -> dict[str, list[str]]:
    """`rirs/<set>` to its licence, for every RIR set a row reads (BUT ReverbDB is CC BY 4.0 and needs credit).

    The set is the directory under `datasets/raw/rirs/` that `rir_id` points into; the licences are the fetcher's.
    """
    ids = _rir_ids(rows)
    if not ids:
        return {}
    licences = (read_manifest(raw_dir("rirs")) or {}).get("licences", {})
    sets = sorted({rir_id.split("/")[3] for rir_id in ids})
    if absent := [s for s in sets if s not in licences]:
        raise ValueError(f"the rirs manifest records no licence for {', '.join(absent)}")
    return {f"rirs/{s}": [licences[s]] for s in sets}


def set_small(root: Path, hours: float) -> None:
    """Set `in_small` in `train.parquet` and `index.parquet`: the lowest ranks of everything except LibriVAD, up to
    `hours` of audio, a sample that would pass it not taken. Every call recomputes both files from scratch.

    Both files are written aside before either is replaced, so a failure while writing leaves both as they were.
    A kill between the two renames can still leave them disagreeing; `verify` finds that, and running this again
    repairs it.
    """
    if not (math.isfinite(hours) and hours > 0):
        raise ValueError(f"hours must be positive and finite, got {hours}")
    # shortcut: both files are held in memory; scan lazily if the full train file does not fit.
    train = asyncio.run(read_parquet(root / "train.parquet"))
    index = asyncio.run(read_parquet(root / "index.parquet"))
    eligible = train.filter(pl.col("source") != _NOT_IN_SMALL).sort("rank", "sample_id")
    chosen = eligible.filter(pl.col("duration").cum_sum() <= hours * 3600)["sample_id"]
    in_small = pl.col("sample_id").is_in(chosen.implode())
    asyncio.run(
        write_parquet(
            {
                root / "train.parquet": train.lazy().with_columns(in_small=in_small),
                root / "index.parquet": index.lazy().with_columns(
                    in_small=pl.when(pl.col("split") == "train").then(in_small)
                ),
            }
        )
    )


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset publish", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_publish({"slice": SLICE, "full": FULL}[ns.config])
