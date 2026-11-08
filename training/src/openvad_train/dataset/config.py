# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Build configuration: every tunable of docs/project/spec/dataset-construction.md, with two presets."""

from dataclasses import dataclass, field
from pathlib import Path

from openvad_train.fetch.common import data_dir


# ADR 0023: no speaker contributes more than this to a source.
SPEAKER_CAP_HOURS = 30.0

# 9 LibriVAD noise categories plus clean and competing talker.
_NOISE_CATEGORIES = ("babble", "ssn", "city", "domestic", "nature", "office", "public", "street", "transport")
DEMAND_CATEGORIES = ("domestic", "nature", "office", "public", "street", "transport")


def _full_shares() -> dict[str, float]:
    return {"clean": 0.15, "competing": 0.15} | {c: 0.7 / len(_NOISE_CATEGORIES) for c in _NOISE_CATEGORIES}


def _slice_shares() -> dict[str, float]:
    # The slice has DEMAND only: clean plus its six scenes.
    return {"clean": 0.15} | {c: 0.85 / len(DEMAND_CATEGORIES) for c in DEMAND_CATEGORIES}


@dataclass(frozen=True)
class BuildConfig:
    name: str
    sources: tuple[str, ...]  # speech sources, named as in fetch.FETCHERS
    noise: tuple[str, ...]  # noise pools: DEMAND, WHAM! and the MUSAN noise files
    languages: tuple[str, ...] | None  # None: every language of the sources
    plan_version: str = "v1"
    global_seed: int = 0
    condition_shares: dict[str, float] = field(default_factory=_full_shares)
    snr_grid_db: tuple[int, ...] = (-5, 0, 5, 10, 15, 20)
    # Padding, labels and clean-sample dither (G is applied only at rasterisation).
    speech_fraction: float = 0.33
    gap_ms: int = 200
    dither_dbfs: tuple[float, float] = (-90.0, -70.0)
    # Sources cut at word gaps: speech per chunk and the length of the padded sample.
    max_chunk_speech_s: float = 6.6
    min_chunk_speech_s: float = 0.2
    chunk_total_s: tuple[float, float] = (10.0, 20.0)
    # Competing talker (ADR 0022) and babble.
    competing_segments: tuple[int, int] = (1, 3)
    competing_coverage: tuple[float, float] = (0.10, 0.50)
    competing_tir_db: tuple[float, float] = (0.0, 15.0)
    competing_lowpass_hz: tuple[float, float] = (1000.0, 4000.0)
    babble_talkers: int = 6
    babble_min_speakers: int = 50  # fewer speakers: babble draws from any language
    reserved_share: float = 0.05
    # Transients: synthetic clicks and DTMF.
    transient_p: float = 0.3
    transient_peak_db: tuple[float, float] = (-18.0, 0.0)
    # Stored channel chain: probability per step and its parameter ranges.
    room_p: float = 0.5
    handset_p: float = 0.7
    handset_highpass_hz: tuple[float, float] = (100.0, 300.0)
    handset_lowpass_hz: tuple[float, float] = (3400.0, 7000.0)
    handset_peak_db: float = 6.0
    agc_p: float = 0.5
    agc_target_dbfs: tuple[float, float] = (-26.0, -16.0)
    agc_attack_ms: tuple[float, float] = (5.0, 20.0)
    agc_release_ms: tuple[float, float] = (100.0, 500.0)
    clip_p: float = 0.15
    clip_drive_db: tuple[float, float] = (3.0, 12.0)
    suppression_p: float = 0.3
    # Stored codec leg: share and bitrate range in kbit/s (no bitrate for none).
    codec_shares: dict[str, float] = field(
        default_factory=lambda: {"none": 0.4, "g722": 0.3, "amr_wb": 0.15, "opus": 0.05, "amr_nb": 0.1}
    )
    codec_bitrate_kbps: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "g722": (64.0, 64.0),
            "amr_wb": (6.6, 23.85),
            "opus": (12.0, 32.0),
            "amr_nb": (4.75, 12.2),
        }
    )
    p_narrow: float = 0.35  # training parameter: share of wide samples narrowed at train time
    flac_mb_per_hour: float = 64.8  # measured on the slice, tars per hour of audio: docs/notes/datasets.md
    max_samples: int | None = None  # keep the lowest ranks once drawn; for the slice, whose draw is far larger


FULL = BuildConfig(
    name="openvad-v1",
    sources=(
        "common-voice",
        "mls",
        "fleurs",
        "swc",
        "voxpopuli",
        "ami",
        "callhome",
        "speech-commands",
        "voices",
        "librivad",
    ),
    noise=("demand", "wham", "musan"),
    languages=None,
)

SLICE = BuildConfig(
    name="openvad-slice",
    sources=("swc",),
    noise=("demand",),
    languages=("de",),
    condition_shares=_slice_shares(),
    max_samples=500,
    # The transient leg arrives with Task 18.
    transient_p=0.0,
)

# The MFA pilot that sets the quality bar (spec section "Quality bar"): Common Voice alone, in the languages it aligns.
# Nothing past `align` and the hand check reads it.
PILOT = BuildConfig(
    name="openvad-pilot",
    sources=("common-voice",),
    noise=(),
    languages=("en", "de", "it", "ja", "zh", "ca"),
)

# The configurations a stage's `--config` names; plan, construct, publish and verify take only `slice` and `full`.
CONFIGS = {"slice": SLICE, "full": FULL, "pilot": PILOT}


def dataset_root(cfg: BuildConfig) -> Path:
    return data_dir() / "datasets/processed" / cfg.name


def work_dir(cfg: BuildConfig) -> Path:
    """Working state of every stage of one configuration, so a slice run never overwrites the full build's."""
    return dataset_root(cfg).parent / "work" / cfg.name
