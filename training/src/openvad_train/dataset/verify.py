# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `verify`: check the published files and read every pointer back, failing on any that does not resolve."""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import soundfile as sf

from openvad_train.dataset.audio import load_audio
from openvad_train.dataset.config import FULL, SLICE, BuildConfig, dataset_root, work_dir
from openvad_train.dataset.parquet_io import read_parquet
from openvad_train.dataset.publish import INDEX_COLUMNS, SPLITS
from openvad_train.fetch.common import cache_dir, data_dir, file_hash, raw_dir, read_manifest, take_share_lock


log = logging.getLogger(__name__)

# Rows per source of as-is files decoded in full (rate and label length); every as-is pointer is only checked to exist.
_AS_IS_SAMPLE = 200
_MAX_REPORTED = 20


class VerifyError(Exception):
    """The published set would not load as described; the message lists what failed."""


@dataclass(frozen=True)
class VerifyReport:
    state: str
    samples: dict[str, int]
    tars_hashed: int
    constructed_decoded: int
    as_is_decoded: int
    # Per source and split: samples, mean `speech_ratio` column, mean speech share of the labels, condition shares.
    groups: list[dict]
    # Sources with no speaker ID in any row: their speaker disjointness cannot be checked (ADR 0023).
    speakers_unchecked: list[str]


def _fragment_of(root: Path, work: Path, tar: Path) -> Path:
    return work / "fragments" / tar.relative_to(root / "data").with_suffix(".parquet")


def _check_tars(root: Path, work: Path, tars: set[str], problems: list[str]) -> set[str]:
    """Hash every tar against its sidecar. A tar whose hash differs has its fragment removed: a same-size copy with
    one flipped byte would otherwise read as a finished shard, and `construct` could never repair it. Returns the
    tars that cannot be trusted.
    """
    bad: set[str] = set()
    for name in sorted(tars):
        tar = data_dir() / name
        sidecar = tar.with_name(tar.name + ".sha256")
        if not tar.is_file() or not sidecar.is_file():
            problems.append(f"{name}: the tar or its sha256 sidecar is missing")
            bad.add(name)
        elif file_hash(tar, "sha256") != sidecar.read_text().split()[0]:
            fragment = _fragment_of(root, work, tar)
            fragment.unlink(missing_ok=True)
            log.warning("%s fails its sha256: removed %s, so construct rebuilds the shard", name, fragment)
            problems.append(f"{name}: sha256 differs from its sidecar; its fragment is removed, rerun construct")
            bad.add(name)
    return bad


def _check_constructed(rows: pl.DataFrame, problems: list[str]) -> int:
    """Every constructed row through `load_audio`, against its duration and labels."""
    # shortcut: serial, one open per row; parallelise per tar if the full build is too slow.
    for row in rows.iter_rows(named=True):
        try:
            audio = load_audio(row)
        except Exception as e:
            problems.append(f"{row['sample_id']}: {type(e).__name__}: {e}")
            continue
        if len(audio) != round(row["duration"] * 16000):
            problems.append(f"{row['sample_id']}: {len(audio)} samples for a duration of {row['duration']} s")
        elif round(len(audio) / 160) != len(row["labels"]):
            problems.append(f"{row['sample_id']}: {len(audio)} samples and {len(row['labels'])} labels")
    return rows.height


def _check_as_is(rows: pl.DataFrame, problems: list[str]) -> int:
    """Every as-is path exists; a seeded sample of each source is decoded, at 16 kHz, with a label for each frame."""
    absent = {path for path in rows["audio_filepath"].unique().sort() if not (data_dir() / path).exists()}
    problems.extend(f"{path}: an as-is row points at a file that does not exist" for path in sorted(absent))
    # Per source: a single sample over all of them would miss a small source, and nothing else checks its files.
    sample = pl.concat([g.sample(min(_AS_IS_SAMPLE, g.height), seed=0) for g in rows.partition_by("source")])
    for row in sample.iter_rows(named=True):
        if row["audio_filepath"] in absent:
            continue
        try:
            if (
                row["audio_format"] == "file"
                and (rate := sf.info(data_dir() / row["audio_filepath"]).samplerate) != 16000
            ):
                problems.append(f"{row['sample_id']}: raw file is {rate} Hz, not 16 kHz")
                continue
            audio = load_audio(row)
        except Exception as e:
            problems.append(f"{row['sample_id']}: {type(e).__name__}: {e}")
            continue
        if round(len(audio) / 160) != len(row["labels"]):
            problems.append(f"{row['sample_id']}: {len(audio)} samples and {len(row['labels'])} labels")
    return sample.height


def _check_index(rows: pl.DataFrame, index: pl.DataFrame, problems: list[str]) -> None:
    if index["sample_id"].n_unique() != index.height:
        problems.append("index.parquet repeats a sample id")
    if rows["sample_id"].n_unique() != rows.height:
        problems.append("a sample id is in more than one split, or twice in one")
    key = ["sample_id"]
    if not rows.select(INDEX_COLUMNS).sort(key).equals(index.select(INDEX_COLUMNS).sort(key)):
        problems.append("index.parquet and the split files disagree on a sample, its split, tier, source or pointer")


def _check_speakers(rows: pl.DataFrame, problems: list[str]) -> list[str]:
    known = rows.filter(pl.col("speaker").is_not_null())
    shared = known.group_by("source", "speaker").agg(splits=pl.col("split").n_unique()).filter(pl.col("splits") > 1)
    problems += [
        f"{r['source']}: speaker {r['speaker']!r} is in {r['splits']} splits" for r in shared.iter_rows(named=True)
    ]
    return sorted(set(rows["source"]) - set(known["source"]))


def _summarise(rows: pl.DataFrame) -> list[dict]:
    speech_share = pl.col("labels").list.eval((pl.element() == 1).cast(pl.Float64)).list.mean()
    groups = (
        rows.group_by("source", "split")
        .agg(samples=pl.len(), speech_ratio=pl.col("speech_ratio").mean(), label_speech_ratio=speech_share.mean())
        .sort("source", "split")
    )
    shares = (
        rows.group_by("source", "split", "condition")
        .len()
        .with_columns(share=pl.col("len") / pl.col("len").sum().over("source", "split"))
    )
    return [
        g
        | {
            "condition_shares": dict(
                shares.filter((pl.col("source") == g["source"]) & (pl.col("split") == g["split"]))
                .sort("condition")
                .select("condition", "share")
                .iter_rows()
            )
        }
        for g in groups.iter_rows(named=True)
    ]


def run_verify(cfg: BuildConfig) -> VerifyReport:
    """Check the published set, and raise `VerifyError` listing every failure.

    Checks: the raw manifests against the hashes `publish` recorded; every tar's sha256 against its sidecar (a failing
    one has its fragment removed, see `_check_tars`); every constructed row decoded and its length and labels checked;
    every as-is path exists and a sample of them decodes at 16 kHz; one index row per sample, agreeing with the split
    files; and no speaker in two splits, for the sources that have speaker IDs. The report gives speech ratios and
    condition shares per source and split.

    An interim build may have no validation or test sample and still pass, with its `missing_shards` in the manifest;
    `load_dataset` refuses a split without data, and this does not call it. A build with no train sample fails.
    """
    root, work = dataset_root(cfg), work_dir(cfg)
    manifest = read_manifest(root)
    if not manifest:
        raise VerifyError(f"{root} has no readable manifest: run publish first")
    for name in (*SPLITS, "index"):
        if not (root / f"{name}.parquet").is_file():
            raise VerifyError(f"{root / f'{name}.parquet'} is missing: run publish first")
    cache_dir().mkdir(parents=True, exist_ok=True)
    # The construct lock, because a running construct writes the shards this reads and may rebuild them.
    lock = take_share_lock(f"construct-{cfg.name}")
    try:
        frames = {split: asyncio.run(read_parquet(root / f"{split}.parquet")) for split in SPLITS}
        index = asyncio.run(read_parquet(root / "index.parquet"))
        rows = pl.concat([f.with_columns(split=pl.lit(s)) for s, f in frames.items()])
        problems: list[str] = []
        if not frames["train"].height:
            problems.append("train.parquet has no samples")
        for split, frame in frames.items():
            if manifest["samples"][split] != frame.height:
                problems.append(
                    f"manifest.json says {split} has {manifest['samples'][split]} samples, the file has {frame.height}"
                )
        for name, recorded in manifest["raw_sources"].items():
            if file_hash(raw_dir(name) / "manifest.json", "sha256") != recorded:
                problems.append(f"{name}: the raw manifest changed since publish, so the set is stale")
        constructed = rows.filter(pl.col("treatment") == "augmented")
        tars = set(constructed.filter(pl.col("audio_format") == "flac_tar")["audio_filepath"])
        bad = _check_tars(root, work, tars, problems)
        n_constructed = _check_constructed(constructed.filter(~pl.col("audio_filepath").is_in(bad)), problems)
        as_is = rows.filter(pl.col("treatment") == "as_is")
        n_as_is = _check_as_is(as_is, problems) if as_is.height else 0
        _check_index(rows, index, problems)
        unchecked = _check_speakers(rows, problems)
        if (top := rows.select(pl.col("labels").explode().max()).item()) is not None and top > 2:
            problems.append("a label is not 0, 1 or 2")
    finally:
        lock.unlink(missing_ok=True)
    if problems:
        shown = problems[:_MAX_REPORTED] + (
            [f"... and {len(problems) - _MAX_REPORTED} more"] if len(problems) > _MAX_REPORTED else []
        )
        raise VerifyError(f"{len(problems)} problems in {root}:\n" + "\n".join(shown))
    return VerifyReport(
        manifest["state"],
        {split: f.height for split, f in frames.items()},
        len(tars),
        n_constructed,
        n_as_is,
        _summarise(rows),
        unchecked,
    )


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset verify", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = run_verify({"slice": SLICE, "full": FULL}[ns.config])
    except VerifyError as e:
        raise SystemExit(str(e)) from e
    print(f"verified: state {report.state}, samples {report.samples}")
    print(
        f"{report.tars_hashed} tars hashed, {report.constructed_decoded} constructed and "
        f"{report.as_is_decoded} as-is rows decoded"
    )
    for g in report.groups:
        shares = ", ".join(f"{c} {s:.2f}" for c, s in g["condition_shares"].items())
        print(
            f"{g['source']} {g['split']}: {g['samples']} samples, speech ratio {g['speech_ratio']:.3f} "
            f"(labels {g['label_speech_ratio']:.3f}); {shares}"
        )
    if report.speakers_unchecked:
        print(f"speaker disjointness not checkable (no speaker IDs): {', '.join(report.speakers_unchecked)}")
