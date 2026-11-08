# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A synthetic planned build for the publish and verify tests: tone files, a plan of five constructed rows over
four shards, and optionally as-is rows. `construct` and `publish` then run on it as on real data.
"""

import asyncio
import dataclasses
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from openvad_train.dataset.config import SLICE, BuildConfig, work_dir
from openvad_train.dataset.parquet_io import write_parquet
from openvad_train.dataset.plan import PLAN_SCHEMA, draw_sample
from openvad_train.dataset.schema import Clip, Timing
from openvad_train.fetch.common import write_manifest
from rir_support import synthetic_rir


Split = Literal["train", "validation", "test"]
Kind = Literal["word", "turn", "frame"]
CHAIN_OFF = {"room": False, "handset": None, "agc": None, "clip_drive_db": None, "suppression": None, "transients": []}
CLEAN = {"condition": "clean", "snr_db": None, "noise_source_ids": [], "noise_offsets": [], "dither_dbfs": -80.0}
NOISE_FILE = "datasets/raw/demand/OOFFICE/ch01.wav"
RIR_FILE = "datasets/raw/rirs/but/room/rir.wav"
OFFICE = {"condition": "office", "snr_db": 5.0, "noise_source_ids": [NOISE_FILE], "noise_offsets": [1000]}


def _tone(seconds: float) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * 440 * np.arange(round(seconds * 16000)) / 16000)).astype(np.float32)


def _row(
    root: Path,
    source: str,
    name: str,
    split: Split,
    speaker: str | None,
    seconds: float,
    speech,
    kind: Kind = "word",
    raw: str | None = None,
    **override,
):
    path = root / "datasets/raw" / (raw or source) / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _tone(seconds), 16000, subtype="PCM_16")
    (path.parent / "manifest.json").write_text(f'{{"dataset": "{source}"}}')
    clip = Clip(name, source, "de", speaker, split, path.relative_to(root).as_posix(), 0.0, seconds, "CC0", None)
    row = draw_sample(clip, Timing(name, kind, list(speech), [], None, "test"), SLICE)
    if row["treatment"] == "augmented":
        row = row | CHAIN_OFF | CLEAN | {"codec": "none", "bitrate": None, "band": "wide"} | override
    return row


def write_plan(cfg: BuildConfig, rows: list[dict]) -> None:
    out = work_dir(cfg) / "plan"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({out / "plan.parquet": pl.DataFrame(rows, schema=PLAN_SCHEMA).lazy()}))
    write_manifest(out, {"stage": "plan", "plan_version": cfg.plan_version, "global_seed": cfg.global_seed})


def make_build(
    root: Path, *, as_is: bool = False, val: bool = False, room: bool = False, name: str = "test-build"
) -> BuildConfig:
    """Write the raw files and the plan under `root`, which must be `$OPENVAD_DATA_DIR`.

    `room` puts a RIR, with a raw manifest, under `datasets/raw/rirs` and gives the first row a room.
    """
    noise = root / NOISE_FILE
    noise.parent.mkdir(parents=True)
    sf.write(
        noise, (0.1 * np.random.default_rng(7).standard_normal(400_000)).astype(np.float32), 16000, subtype="PCM_16"
    )
    (root / "datasets/raw/demand/manifest.json").write_text('{"dataset": "demand"}')
    if room:
        (root / RIR_FILE).parent.mkdir(parents=True)
        sf.write(root / RIR_FILE, synthetic_rir(), 16000, subtype="FLOAT")
        (root / "datasets/raw/rirs/manifest.json").write_text('{"dataset": "rirs"}')
    rows = [
        _row(
            root, "swc", "c0", "train", "t1", 3.0, [(500, 2500)], **({"room": True, "rir_id": RIR_FILE} if room else {})
        )
        | {"shard": "train/00000"},
        _row(root, "swc", "c1", "train", "t2", 2.0, [(100, 1800)], **OFFICE) | {"shard": "train/00000"},
        _row(root, "swc", "c2", "train", "t1", 3.0, [(0, 3000)]) | {"shard": "train/00001"},
        _row(root, "swc", "c3", "validation", "v1", 2.0, [(200, 1500)]) | {"shard": "validation/00000"},
        _row(root, "swc", "c4", "test", "x1", 2.0, [(200, 1500)]) | {"shard": "test/00000"},
    ]
    if as_is:
        rows += [
            _row(root, "ami", "a0", "train", "ma", 4.0, [(500, 1500), (2000, 3000)]),
            _row(root, "ami", "a1", "validation", "mb", 2.0, [(100, 900)]),
            # A gap of 100 ms, under G: frame labels keep it open.
            _row(root, "librivad", "l0", "train", None, 3.0, [(300, 1000), (1100, 2700)], kind="frame"),
            _row(root, "librivad", "l1", "test", None, 3.0, [(300, 2700)], kind="frame"),
        ]
    if val:
        # A `librivad` row read from the validation tree, which has its own raw manifest.
        rows.append(
            _row(root, "librivad", "l2", "validation", None, 3.0, [(300, 2700)], kind="frame", raw="librivad-val")
        )
    cfg = dataclasses.replace(SLICE, name=name)
    write_plan(cfg, rows)
    return cfg


@pytest.fixture
def free_space(monkeypatch) -> None:
    """The test machine's real disk must not decide whether a shard is built."""
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=10**16))
