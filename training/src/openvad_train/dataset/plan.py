# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `plan`: one row per sample with every random draw resolved, so construction is a pure function of a row."""

import argparse
import asyncio
import dataclasses
import logging
from collections import Counter, defaultdict
from types import ModuleType

import numpy as np
import polars as pl

from openvad_train.dataset.config import DEMAND_CATEGORIES, FULL, SLICE, BuildConfig, work_dir
from openvad_train.dataset.cutting import Chunk, cut_long, rebase
from openvad_train.dataset.index import SOURCES
from openvad_train.dataset.labels import apply_gap_rule, rasterise, speech_ratio
from openvad_train.dataset.mix import padded_length
from openvad_train.dataset.noise import NoisePool
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.schema import Clip, Timing, rank, sample_seed
from openvad_train.fetch.common import data_dir, raw_dir, write_manifest


log = logging.getLogger(__name__)

SAMPLES_PER_MS = 16
SAMPLE_RATE = 16_000

AUGMENTED_SOURCES = {"common-voice", "mls", "fleurs", "swc"}
_CUT_SOURCES = {"mls", "swc"}
# The modes the codecs have, so a drawn bitrate is one ffmpeg can encode.
_CODEC_MODES = {
    "amr_wb": (6.6, 8.85, 12.65, 14.25, 15.85, 18.25, 19.85, 23.05, 23.85),
    "amr_nb": (4.75, 5.15, 5.9, 6.7, 7.4, 7.95, 10.2, 12.2),
}
# Competing talker, babble and SSN need reserved clips and city needs WHAM!, which a later task adds to the draws.
_CONDITIONS_WITH_A_DRAW = {"clean", *DEMAND_CATEGORIES}

_INTERVALS = pl.List(pl.List(pl.Int64))
_PLAN_SCHEMA = {
    "sample_id": pl.String,
    "clip_id": pl.String,
    "split": pl.String,
    "rank": pl.UInt64,
    "seed": pl.UInt64,
    "shard": pl.String,
    "source": pl.String,
    "language": pl.String,
    "speaker": pl.String,
    "licence": pl.String,
    "treatment": pl.String,
    "aligner": pl.String,
    "audio_filepath": pl.String,
    "audio_format": pl.String,
    "byte_offset": pl.Int64,
    "byte_length": pl.Int64,
    "offset": pl.Float64,
    "duration": pl.Float64,
    "row_in_file": pl.Int64,
    "labels": pl.List(pl.UInt8),
    "band": pl.String,
    "in_small": pl.Boolean,
    "speech_ratio": pl.Float64,
    "condition": pl.String,
    "noise_source_ids": pl.List(pl.String),
    "noise_offsets": pl.List(pl.Int64),
    "snr_db": pl.Float64,
    "tir_db": pl.Float64,
    "baseline_snr_db": pl.Float64,
    "room": pl.Boolean,
    "rir_id": pl.String,
    "handset": pl.Struct(
        {
            "highpass_hz": pl.Float64,
            "lowpass_hz": pl.Float64,
            "peaks": pl.List(pl.Struct({"freq_hz": pl.Float64, "gain_db": pl.Float64})),
        }
    ),
    "agc": pl.Struct({"target_dbfs": pl.Float64, "attack_ms": pl.Float64, "release_ms": pl.Float64}),
    "clip_drive_db": pl.Float64,
    "suppression": pl.String,
    "codec": pl.String,
    "bitrate": pl.Float64,
    "codec_delay_ms": pl.Int64,
    "transients": pl.List(pl.Struct({"kind": pl.String, "offset_n": pl.Int64, "peak_db": pl.Float64})),
    "gap_ms": pl.Int64,
    "dither_dbfs": pl.Float64,
    "segments": pl.List(
        pl.Struct(
            {
                "clip_id": pl.String,
                "start_n": pl.Int64,
                "end_n": pl.Int64,
                "rir_id": pl.String,
                "lowpass_hz": pl.Float64,
            }
        )
    ),
    "source_path": pl.String,
    "source_offset_n": pl.Int64,
    "source_n": pl.Int64,
    "speech_offset_n": pl.Int64,
    "total_n": pl.Int64,
    "speech_ms": _INTERVALS,
    "mask_ms": _INTERVALS,
}


def _draw_bitrate(codec: str, cfg: BuildConfig, rng: np.random.Generator) -> float | None:
    if codec not in cfg.codec_bitrate_kbps:
        return None
    low, high = cfg.codec_bitrate_kbps[codec]
    modes = [m for m in _CODEC_MODES.get(codec, ()) if low <= m <= high]
    return float(rng.choice(modes)) if modes else float(rng.uniform(low, high))


def _draw_handset(cfg: BuildConfig, rng: np.random.Generator) -> dict:
    highpass, lowpass = float(rng.uniform(*cfg.handset_highpass_hz)), float(rng.uniform(*cfg.handset_lowpass_hz))
    return {
        "highpass_hz": highpass,
        "lowpass_hz": lowpass,
        "peaks": [
            {
                "freq_hz": float(np.exp(rng.uniform(np.log(highpass), np.log(lowpass)))),
                "gain_db": float(rng.uniform(-cfg.handset_peak_db, cfg.handset_peak_db)),
            }
            for _ in range(2)
        ],
    }


def _draw_mix_and_chain(cfg: BuildConfig, total_n: int, rng: np.random.Generator) -> dict:
    condition = str(rng.choice(list(cfg.condition_shares), p=list(cfg.condition_shares.values())))
    clean = condition == "clean"
    snr_db = None if clean or condition == "competing" else float(rng.choice(cfg.snr_grid_db))
    dither_dbfs = float(rng.uniform(*cfg.dither_dbfs)) if clean else None
    transients = (
        [
            {
                "kind": str(rng.choice(("click", "dtmf"))),
                "offset_n": int(rng.integers(total_n)),
                "peak_db": float(rng.uniform(*cfg.transient_peak_db)),
            }
        ]
        if rng.random() < cfg.transient_p
        else []
    )
    room = bool(rng.random() < cfg.room_p)
    handset = _draw_handset(cfg, rng) if rng.random() < cfg.handset_p else None
    agc = (
        {
            "target_dbfs": float(rng.uniform(*cfg.agc_target_dbfs)),
            "attack_ms": float(rng.uniform(*cfg.agc_attack_ms)),
            "release_ms": float(rng.uniform(*cfg.agc_release_ms)),
        }
        if rng.random() < cfg.agc_p
        else None
    )
    clip_drive_db = float(rng.uniform(*cfg.clip_drive_db)) if rng.random() < cfg.clip_p else None
    suppression = str(rng.choice(("wiener", "spectral_subtraction"))) if rng.random() < cfg.suppression_p else None
    codec = str(rng.choice(list(cfg.codec_shares), p=list(cfg.codec_shares.values())))
    return {
        "condition": condition,
        "snr_db": snr_db,
        "dither_dbfs": dither_dbfs,
        "transients": transients,
        "room": room,
        "handset": handset,
        "agc": agc,
        "clip_drive_db": clip_drive_db,
        "suppression": suppression,
        "codec": codec,
        "bitrate": _draw_bitrate(codec, cfg, rng),
        "band": "narrow_leg" if codec == "amr_nb" else "wide",
    }


def draw_sample(
    clip: Clip, timing: Timing, cfg: BuildConfig, pool: NoisePool | None = None, chunk: Chunk | None = None
) -> dict:
    """The plan row of `clip`, or of its `chunk`; its draws come only from the seed of its sample id.

    The row's columns are the spec's split columns plus what construction reads: `source_path`, `source_offset_n`
    and `source_n` say which audio is the foreground, `speech_offset_n` where it goes inside `total_n`, and
    `speech_ms` and `mask_ms` hold the intervals counted from the foreground's start.
    Without `pool` a noise condition is named but has no noise source, for tests that look only at the draws.
    """
    sample_id = f"{clip.source}/{clip.clip_id}" + ("" if chunk is None else f"@{chunk.start_ms}")
    seed = sample_seed(cfg.global_seed, cfg.plan_version, sample_id)
    start_ms, end_ms = (0, round(clip.duration_s * 1000)) if chunk is None else chunk
    source_n = round(clip.duration_s * SAMPLE_RATE) if chunk is None else (end_ms - start_ms) * SAMPLES_PER_MS
    speech_ms, mask_ms = rebase(timing.speech_ms, start_ms, end_ms), rebase(timing.mask_ms, start_ms, end_ms)
    row = {
        "sample_id": sample_id,
        "clip_id": clip.clip_id,
        "split": clip.split,
        "rank": rank(sample_id),
        "seed": seed,
        "source": clip.source,
        "language": clip.language,
        "speaker": clip.speaker,
        "licence": clip.licence,
        "aligner": timing.provenance,
        "gap_ms": cfg.gap_ms,
        "byte_offset": None,
        "byte_length": None,
        "row_in_file": None,
        "labels": None,
        "in_small": None,
        "shard": None,
        "tir_db": None,
        "baseline_snr_db": None,
        "rir_id": None,
        "codec_delay_ms": None,
        "segments": [],
        "source_path": clip.path,
        "source_offset_n": round(clip.offset_s * SAMPLE_RATE) + start_ms * SAMPLES_PER_MS,
        "source_n": source_n,
        "speech_ms": speech_ms,
        "mask_ms": mask_ms,
        "noise_source_ids": [],
        "noise_offsets": [],
    }
    if clip.source not in AUGMENTED_SOURCES:
        return row | {
            "treatment": "as_is",
            "audio_filepath": clip.path,
            "audio_format": "file",
            "offset": clip.offset_s,
            "duration": clip.duration_s,
            "speech_ratio": speech_ratio(speech_ms, end_ms - start_ms),
            "speech_offset_n": 0,
            "total_n": source_n,
            "condition": "as_is",
            "snr_db": None,
            "dither_dbfs": None,
            "transients": [],
            "room": False,
            "handset": None,
            "agc": None,
            "clip_drive_db": None,
            "suppression": None,
            "codec": "none",
            "bitrate": None,
            "band": "wide",
        }

    # Padding uses the word union at G = 0, so a change of G leaves the audio as it is.
    speech_s = sum(e - s for s, e in apply_gap_rule(speech_ms, 0)) / 1000
    if chunk is None:
        total_s = padded_length(speech_s, cfg.speech_fraction, max(0.0, clip.duration_s - speech_s))
    else:
        total_s = float(np.clip(speech_s / cfg.speech_fraction, *cfg.chunk_total_s))
    total_n = max(round(total_s * SAMPLE_RATE), source_n)

    rng = np.random.default_rng(seed)
    # Later tasks add their draws after these, so the values drawn so far stay. A whole-ms offset keeps labels exact.
    speech_offset_n = SAMPLES_PER_MS * int(rng.integers((total_n - source_n) // SAMPLES_PER_MS + 1))
    drawn = _draw_mix_and_chain(cfg, total_n, rng)
    if pool is not None and drawn["condition"] in DEMAND_CATEGORIES:
        noise = pool.draw(drawn["condition"], clip.split, rng)
        row["noise_source_ids"] = [noise.file.relative_to(data_dir()).as_posix()]
        row["noise_offsets"] = [noise.offset_n]
    return (
        row
        | {
            "treatment": "augmented",
            "audio_filepath": None,
            "audio_format": "flac_tar",
            "offset": 0.0,
            "duration": total_n / SAMPLE_RATE,
            "speech_ratio": speech_ratio(speech_ms, total_n // SAMPLES_PER_MS),
            "speech_offset_n": speech_offset_n,
            "total_n": total_n,
        }
        | drawn
    )


def _has_speech_frame(timing: Timing, start_ms: int, end_ms: int) -> bool:
    speech, mask = rebase(timing.speech_ms, start_ms, end_ms), rebase(timing.mask_ms, start_ms, end_ms)
    return bool((rasterise(speech, mask, (end_ms - start_ms) * SAMPLES_PER_MS) == 1).any())


def clip_rows(clip: Clip, timing: Timing, cfg: BuildConfig, pool: NoisePool | None = None) -> list[dict]:
    """The rows of one clip: none if it is reserved, and an augmented sample only if a frame of it is speech."""
    if clip.reserved:
        return []
    if clip.source not in AUGMENTED_SOURCES:
        return [draw_sample(clip, timing, cfg, pool)]
    duration_ms = round(clip.duration_s * 1000)
    if clip.source not in _CUT_SOURCES:
        return [draw_sample(clip, timing, cfg, pool)] if _has_speech_frame(timing, 0, duration_ms) else []
    # Inside the clip: the adapter's timeline may round a millisecond past it.
    inside = dataclasses.replace(
        timing, speech_ms=rebase(timing.speech_ms, 0, duration_ms), mask_ms=rebase(timing.mask_ms, 0, duration_ms)
    )
    # The default G of `cut_long`, never cfg.gap_ms: ablating G must change labels and leave the audio identical.
    chunks = cut_long(inside, cfg.max_chunk_speech_s, max_span_s=cfg.chunk_total_s[1])
    chunks = [Chunk(c.start_ms, min(c.end_ms, duration_ms)) for c in chunks]
    return [draw_sample(clip, inside, cfg, pool, c) for c in chunks if _has_speech_frame(inside, *c)]


def assign_shards(rows: list[dict], bytes_per_hour: float, target_bytes: int = 1_000_000_000) -> None:
    """Sort `rows` by rank and give each augmented one a `shard`, `<split>/<number>`, the tar's path under `data/`.

    Shards fill in rank order within a split, so an early stop leaves a clean prefix. As-is rows have none.
    """
    rows.sort(key=lambda r: r["rank"])
    number: dict[str, int] = defaultdict(int)
    filled: dict[str, float] = defaultdict(float)
    for row in rows:
        if row["treatment"] != "augmented":
            continue
        split, size = row["split"], row["duration"] / 3600 * bytes_per_hour
        if filled[split] and filled[split] + size > target_bytes:
            number[split] += 1
            filled[split] = 0.0
        filled[split] += size
        row["shard"] = f"{split}/{number[split]:05d}"


def _parse_limit(limit: str | None) -> tuple[int | None, float | None]:
    """`--limit` as (samples, hours); at most one is set."""
    if limit is None:
        return None, None
    hours = limit.endswith("h")
    try:
        value = float(limit[:-1]) if hours else int(limit)
    except ValueError:
        raise ValueError(f"limit must be a sample count or hours such as 1000h, got {limit!r}") from None
    if not value > 0:  # also refuses NaN
        raise ValueError(f"limit must be positive, got {limit!r}")
    return (None, float(value)) if hours else (int(value), None)


def _lowest_ranks(rows: list[dict], samples: int | None, hours: float | None) -> list[dict]:
    """The prefix of rank-sorted `rows` within the limit; a sample that would pass the hours is not taken."""
    if samples is not None:
        return rows[:samples]
    if hours is None:
        return rows
    kept, total_s = [], 0.0
    for row in rows:
        total_s += row["duration"]
        if total_s > hours * 3600:
            break
        kept.append(row)
    return kept


def _timings(cfg: BuildConfig, name: str, module: ModuleType, clips: list[Clip]) -> dict[str, Timing]:
    if hasattr(module, "timings"):
        return {t.clip_id: t for t in module.timings(raw_dir(name), clips)}
    words = work_dir(cfg) / "words" / name
    if not words.is_dir():
        raise FileNotFoundError(f"{name} is labelled by MFA and {words} does not exist: run the align stage first")
    raise NotImplementedError(f"reading the MFA word tables under {words} arrives with the align stage")


def _print_summary(df: pl.DataFrame, bytes_per_hour: float) -> list[dict]:
    """Hours and estimated constructed bytes per source, split and condition; as-is sources store nothing."""
    built_s = pl.col("duration").filter(pl.col("treatment") == "augmented").sum()
    summary = (
        df.group_by("source", "split", "condition")
        .agg(samples=pl.len(), hours=pl.col("duration").sum() / 3600, bytes=built_s / 3600 * bytes_per_hour)
        .sort("source", "split", "condition")
        .to_dicts()
    )
    for s in summary:
        print(
            f"{s['source']} {s['split']} {s['condition']}: "
            f"{s['samples']} samples, {s['hours']:.2f} h, {s['bytes'] / 1e9:.2f} GB"
        )
    hours, size = sum(s["hours"] for s in summary), sum(s["bytes"] for s in summary)
    print(f"total: {df.height} samples, {hours:.2f} h, {size / 1e9:.2f} GB")
    return summary


def run_plan(cfg: BuildConfig, limit: str | None = None) -> None:
    """Write `work/<name>/plan/plan.parquet` in rank order and the stage's `manifest.json`, from the index stage's
    tables. `limit`, a sample count or hours such as `1000h`, keeps the lowest ranks.
    """
    samples, hours = _parse_limit(limit)
    missing = sorted(c for c, share in cfg.condition_shares.items() if share > 0 and c not in _CONDITIONS_WITH_A_DRAW)
    if missing:
        raise NotImplementedError(f"conditions without a draw yet: {', '.join(missing)}")
    unknown = [name for name in cfg.sources if name not in SOURCES]
    if unknown:
        raise ValueError(f"no index adapter for source {unknown[0]!r}; have {sorted(SOURCES)}")

    pool = NoisePool(raw_dir("demand"))
    rows: list[dict] = []
    per_source: dict[str, dict] = {}
    for name in cfg.sources:
        fields = [f.name for f in dataclasses.fields(Clip)]
        index = asyncio.run(read_parquet(work_dir(cfg) / "index" / f"{name}.parquet", fields))
        clips = [Clip(**row) for row in index.iter_rows(named=True)]
        timings = _timings(cfg, name, SOURCES[name], clips)
        counts = Counter(reserved=sum(c.reserved for c in clips))
        for clip in clips:
            sample_rows = clip_rows(clip, timings[clip.clip_id], cfg, pool)
            rows += sample_rows
            counts["samples"] += len(sample_rows)
            counts["without_sample"] += not sample_rows and not clip.reserved
        per_source[name] = {"clips": len(clips), **{k: counts[k] for k in ("reserved", "without_sample", "samples")}}
        log.info("%s: %s", name, per_source[name])

    # Sorted before the limit too: `assign_shards` only sees what the limit keeps.
    rows.sort(key=lambda r: r["rank"])
    rows = _lowest_ranks(rows, samples, hours)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("a sample id is repeated: rows would overwrite each other's rank and seed")
    bytes_per_hour = cfg.flac_mb_per_hour * 1e6
    assign_shards(rows, bytes_per_hour)
    df = pl.DataFrame(rows, schema=_PLAN_SCHEMA)
    out = work_dir(cfg) / "plan"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({out / "plan.parquet": df.lazy()}))
    write_manifest(
        out,
        {
            "stage": "plan",
            "config": cfg.name,
            "plan_version": cfg.plan_version,
            "global_seed": cfg.global_seed,
            "limit": limit,
            "samples": df.height,
            "per_source": per_source,
            "summary": _print_summary(df, bytes_per_hour),
        },
    )


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset plan", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    parser.add_argument("--limit", help="keep the lowest ranks: a sample count, or hours such as 1000h")
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_plan({"slice": SLICE, "full": FULL}[ns.config], ns.limit)
