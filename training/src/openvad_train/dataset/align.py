# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stage `align`: word intervals of the MFA sources from the Montreal Forced Aligner (ADR 0025).

Per source and language, the foreground clips of the index go through MFA in batches: speakers in rank order, each
speaker whole in one batch, so a batch is a target size and one speaker may exceed it. Every clip is decoded in this
process by the decode `construct` uses, cut to its offset and duration and written as a 16 kHz WAV into a local
temporary directory, so the times MFA returns are relative to the samples `construct` will cut. MFA never reads a
source file. Each clip ends as exactly one row in `work/words/<source>/<language>/<batch>.parquet` or one drop in
`work/words/<source>/<language>/dropped/<batch>.parquet`; `work/dropped.parquet` is their union.

`mfa_models.tsv` has one row per language: the acoustic model, dictionary and G2P model with the release version to
download, a `status` (`downloaded`: used in a real run; `listed`: the name and version are in the mfa-models release
index but the model was not downloaded; `open`: ADR 0025 leaves the choice open and nothing runs) and a `note`.
"""

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import polars as pl
import soundfile as sf

from openvad_train.dataset.codecs import DecodeError, decode_16k
from openvad_train.dataset.config import FULL, SLICE, SPEAKER_CAP_HOURS, BuildConfig, work_dir
from openvad_train.dataset.index import SOURCES
from openvad_train.dataset.labels import SAMPLES_PER_FRAME, apply_gap_rule, speech_ratio
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.schema import Clip, Timing, rank
from openvad_train.fetch.common import data_dir, file_hash, write_manifest


log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[4]
MODELS_TSV = Path(__file__).with_name("mfa_models.tsv")
# Untuned: a batch is one MFA run, whose start costs about 35 s. The pilot (Task 15) sets this.
BATCH_HOURS = 5.0

# ADR 0025, drop rules: a word over 1.5 s, a phone over 0.5 s, a first word in the first 20 ms, speech under 20%,
# a speaking rate more than three median absolute deviations (unscaled) from the median, and the bottom 5% of
# log-likelihoods. The last two are fitted on the first batch (`DropRules`).
MAX_WORD_MS = 1500
MAX_PHONE_MS = 500
MIN_ONSET_MS = 20
MIN_COVERAGE = 0.2
RATE_MADS = 3.0
LOGLIK_QUANTILE = 0.05
# Speech Commands is exempt from the speaking-rate and first-20-ms rules (owner, 2026-11-06).
EXEMPT_FROM_RATE_AND_ONSET = frozenset({"speech-commands"})

_SILENCE = frozenset({"", "sil", "<eps>"})
_OOV_PHONE = "spn"

WORDS_SCHEMA = {
    "clip_id": pl.String,
    "kind": pl.String,
    "speech_ms": pl.List(pl.Struct({"start": pl.Int64, "end": pl.Int64})),
    "mask_ms": pl.List(pl.Struct({"start": pl.Int64, "end": pl.Int64})),
    "metrics": pl.String,  # JSON: MFA's analysis columns plus what is measured on the TextGrid
    "provenance": pl.String,  # JSON: MFA version and commit, each model's name, release, version and SHA-256
}
DROPPED_SCHEMA = {
    "clip_id": pl.String,
    "source": pl.String,
    "language": pl.String,
    "batch": pl.String,
    "reason": pl.String,
    "metrics": pl.String,  # JSON, or null when the clip never produced an alignment
}


class ToolsNotBuiltError(RuntimeError):
    """MFA or Kaldi is missing, or was built from a commit other than the pinned one."""


class MfaFailedError(RuntimeError):
    """An `mfa` command exited non-zero, or its output does not add up to the clips it was given."""


class StateMismatchError(RuntimeError):
    """Finished batches on disk belong to another batch composition or other tools than this run."""


# ---------------------------------------------------------------------------------------------------- models table


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    language: str
    acoustic: str
    acoustic_version: str
    dictionary: str
    dictionary_version: str
    g2p: str | None
    g2p_version: str | None
    status: str
    note: str


def load_models(path: Path = MODELS_TSV) -> dict[str, ModelSpec]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    models = {}
    for r in rows:
        if None in r.values():  # a short row, such as one whose empty last cell an editor trimmed
            raise ValueError(f"{path.name}: row {r['language']!r} has fewer cells than the header")
        spec = ModelSpec(**{**r, "g2p": r["g2p"] or None, "g2p_version": r["g2p_version"] or None})
        if spec.status != "open" and not (spec.acoustic and spec.acoustic_version and spec.dictionary):
            raise ValueError(f"{path.name}: {spec.language} needs an acoustic model, a version and a dictionary")
        if bool(spec.g2p) != bool(spec.g2p_version):
            raise ValueError(f"{path.name}: {spec.language} has a G2P model without a version or the reverse")
        models[spec.language] = spec
    return models


def spec_for(models: dict[str, ModelSpec], language: str) -> ModelSpec:
    spec = models.get(language)
    if spec is None:
        raise ValueError(f"no row for language {language!r} in {MODELS_TSV.name}; have {sorted(models)}")
    if spec.status == "open":
        raise ValueError(f"{language}: no MFA model is chosen yet ({spec.note})")
    return spec


# ---------------------------------------------------------------------------------------------------- tools


class Tools(NamedTuple):
    root: Path
    version: str
    commit: str  # the full gitlink of tools/montreal-forced-aligner


def _gitlink(root: Path) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), "rev-parse", ":tools/montreal-forced-aligner"], capture_output=True, text=True
    )
    if done.returncode:
        raise ToolsNotBuiltError(f"cannot read the pinned MFA commit from git in {root}: {done.stderr.strip()}")
    return done.stdout.strip()


def check_tools(root: Path = REPO_ROOT) -> Tools:
    """The MFA of `tools/` if it is built from the commit the repository pins, else `ToolsNotBuiltError` naming
    `just build-mfa`. The commit comes from MFA's generated `_version.py`, which the editable install writes.
    """
    version_file = root / "tools/montreal-forced-aligner/montreal_forced_aligner/_version.py"
    built = (root / "tools/kaldi/dist/bin").is_dir() and (root / "tools/mfa/.venv/bin/mfa").exists()
    if not (built and version_file.is_file()):
        raise ToolsNotBuiltError(f"MFA or Kaldi is not built in {root / 'tools'}: run `just build-mfa`")
    text = version_file.read_text()
    version = re.search(r"__version__ = version = '([^']+)'", text)
    commit = re.search(r"__commit_id__ = commit_id = '([0-9a-z]+)'", text)
    if not (version and commit):
        raise ToolsNotBuiltError(f"{version_file} has no version or commit: run `just build-mfa`")
    pinned = _gitlink(root)
    short = commit.group(1).removeprefix("g")  # vcs-versioning writes `g<abbreviated sha>`
    if len(short) < 7 or not pinned.startswith(short):
        raise ToolsNotBuiltError(
            f"MFA was built from {short} but tools/montreal-forced-aligner pins {pinned}: run `just build-mfa`"
        )
    return Tools(root, version.group(1), pinned)


def _mfa(tools: Tools, mfa_root: Path, *args: str) -> None:
    """`mfa <args>` as `just mfa` runs it, with `mfa_root` as MFA's root (its models, databases and temp files)."""
    env = os.environ | {
        "PATH": f"{tools.root / 'tools/kaldi/dist/bin'}{os.pathsep}{os.environ['PATH']}",
        "MFA_ROOT_DIR": str(mfa_root),
    }
    env.pop("VIRTUAL_ENV", None)  # the root workspace's, which `uv run --project tools/mfa` would warn about
    cmd = ["uv", "run", "--project", str(tools.root / "tools/mfa"), "--no-sync", "mfa", *args]
    done = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if done.returncode:
        tail = "\n".join(done.stdout.splitlines()[-25:])
        raise MfaFailedError(f"`mfa {' '.join(args[:2])}` exited {done.returncode}:\n{tail}")


class Models(NamedTuple):
    acoustic: Path
    dictionary: Path
    g2p: Path | None
    info: dict  # name, release, version (as read from the model, None for a dictionary) and SHA-256 of each


def _fetch_model(tools: Tools, kind: str, name: str, release: str) -> Path:
    """The downloaded file of one model release. Each release has its own directory, and the download goes through a
    private MFA root, so a second process fetching the same model cannot see a half-written file.
    """
    models_dir = data_dir() / "models/mfa"
    dest = models_dir / kind / name / release
    if dest.is_dir() and (found := sorted(dest.iterdir())):
        return found[0]
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir, prefix=".download-") as tmp:
        _mfa(tools, Path(tmp), "model", "download", kind, name, "--version", release)
        (file,) = (Path(tmp) / "pretrained_models" / kind).iterdir()
        dest.mkdir(parents=True, exist_ok=True)
        os.replace(file, dest / file.name)
    return dest / file.name


def _model_version(path: Path) -> str | None:
    """The version in the model's own `meta.json`. A dictionary is a plain text file and carries none."""
    if path.suffix != ".zip":
        return None
    with zipfile.ZipFile(path) as z:
        meta = next(n for n in z.namelist() if n.endswith("meta.json"))
        return json.loads(z.read(meta))["version"]


def fetch_models(tools: Tools, spec: ModelSpec) -> Models:
    wanted = {
        "acoustic": (spec.acoustic, spec.acoustic_version),
        "dictionary": (spec.dictionary, spec.dictionary_version),
        **({"g2p": (spec.g2p, spec.g2p_version)} if spec.g2p and spec.g2p_version else {}),
    }
    paths = {kind: _fetch_model(tools, kind, name, release) for kind, (name, release) in wanted.items()}
    info = {
        kind: {
            "name": name,
            "release": release,
            "version": _model_version(paths[kind]),
            "sha256": file_hash(paths[kind], "sha256"),
        }
        for kind, (name, release) in wanted.items()
    }
    return Models(paths["acoustic"], paths["dictionary"], paths.get("g2p"), info)


def _provenance(tools: Tools, models: Models) -> str:
    return json.dumps(
        {"aligner": "mfa", "version": tools.version, "commit": tools.commit, "models": models.info},
        sort_keys=True,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------------------------------- text


def prepare_transcript(text: str, language: str) -> str | None:
    """The transcript MFA gets, or None when no word is left to align.

    Punctuation at the edges of a word goes. A word with a digit becomes `<unk>`; Japanese and Mandarin have no
    spaces, so there each digit run becomes one. `<unk>` is not a word: a transcript of digits and punctuation is
    None. Whether MFA's Japanese and Mandarin tokenisers keep a literal `<unk>` is not verified.
    """
    words = []
    for token in text.split():
        for part in re.split(r"(\d+)", token) if language in ("ja", "zh") else [token]:
            word = re.sub(r"^\W+|\W+$", "", part)
            if word:
                words.append("<unk>" if any(c.isdigit() for c in word) else word)
    return " ".join(words) if any(w != "<unk>" for w in words) else None


# ---------------------------------------------------------------------------------------------------- TextGrid


def parse_textgrid(text: str) -> dict[str, list[tuple[float, float, str]]]:
    """The interval tiers of a TextGrid in Praat's long format, which MFA writes: name to (xmin, xmax, text)."""
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    tier: list[tuple[float, float, str]] | None = None
    interval_tier = False
    xmin = xmax = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("class = "):
            interval_tier, tier = s == 'class = "IntervalTier"', None
        elif s.startswith("name = ") and interval_tier:
            tier = tiers.setdefault(s[len("name = ") :].strip().strip('"'), [])
        elif tier is not None and s.startswith("intervals ["):
            xmin = xmax = None
        elif tier is not None and s.startswith("xmin = "):
            xmin = float(s[len("xmin = ") :])
        elif tier is not None and s.startswith("xmax = "):
            xmax = float(s[len("xmax = ") :])
        elif tier is not None and (m := re.fullmatch(r'text = "(.*)"', s)) and xmin is not None and xmax is not None:
            tier.append((xmin, xmax, m.group(1).replace('""', '"')))
            xmin = xmax = None
    if not tiers:
        raise ValueError("no interval tier found: expected a TextGrid in the long format")
    return tiers


def _tier(tiers: dict[str, list[tuple[float, float, str]]], kind: str) -> list[tuple[float, float, str]]:
    # One speaker per file is named `words`; MFA prefixes the speaker (`<speaker> - words`) when a file has several.
    names = [n for n in tiers if n == kind or n.endswith(f" - {kind}")]
    if len(names) != 1:
        raise ValueError(f"expected one {kind!r} tier, found {sorted(tiers)}")
    return tiers[names[0]]


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def read_alignment(textgrid: Path, metrics: dict, provenance: str = "") -> Timing:
    """The word intervals of one TextGrid in whole milliseconds, with `metrics` extended by what the two tiers say.

    Every non-silent word interval is speech, a word MFA could not look up (`spn` in the phones tier, its spelling
    kept in the words tier) included; nothing is excluded for being out of vocabulary. `clip_id` is the file stem.
    """
    tiers = parse_textgrid(textgrid.read_text(encoding="utf-8"))
    words = [(_ms(a), _ms(b)) for a, b, t in _tier(tiers, "words") if t not in _SILENCE and _ms(b) > _ms(a)]
    phones = [(_ms(a), _ms(b), t) for a, b, t in _tier(tiers, "phones") if t not in _SILENCE and _ms(b) > _ms(a)]
    speech_s = sum(e - s for s, e in words) / 1000
    oov = sum(t == _OOV_PHONE for _, _, t in phones)
    measured = {
        "n_words": len(words),
        "n_phones": len(phones),
        "oov_words": oov,
        "oov_share": oov / len(words) if words else 0.0,
        "max_word_ms": max((e - s for s, e in words), default=0),
        "max_phone_ms": max((e - s for s, e, _ in phones), default=0),
        "first_word_ms": words[0][0] if words else None,
        "rate": len(phones) / speech_s if speech_s > 0 else None,  # phones per second of speech
    }
    return Timing(textgrid.stem, "word", words, [], metrics | measured, provenance)


# ---------------------------------------------------------------------------------------------------- drop rules


@dataclasses.dataclass(frozen=True)
class DropRules:
    """ADR 0025's drop rules for one language and source. The thresholds that depend on data (the log-likelihood
    floor and the speaking rate's median and spread) are fitted once, on the first batch, and then saved and loaded.
    """

    source: str
    loglik_floor: float | None
    rate_median: float | None
    rate_mad: float | None

    @classmethod
    def fit(cls, timings: Iterable[Timing], source: str = "") -> "DropRules":
        metrics = [t.metrics or {} for t in timings]
        logliks = [m["loglik"] for m in metrics if m.get("loglik") is not None]
        rates = np.array([m["rate"] for m in metrics if m.get("rate") is not None])
        median = float(np.median(rates)) if rates.size else None
        return cls(
            source,
            float(np.quantile(logliks, LOGLIK_QUANTILE)) if logliks else None,
            median,
            float(np.median(np.abs(rates - median))) if median is not None else None,
        )

    def apply(self, timing: Timing) -> str | None:
        """The reason to drop the clip, or None. A rule whose threshold could not be fitted is skipped."""
        m = timing.metrics or {}
        if m.get("retried"):
            return "needs_wider_beam"
        words = apply_gap_rule(timing.speech_ms, 0)
        if max((e - s for s, e in timing.speech_ms), default=0) > MAX_WORD_MS:
            return "word_too_long"
        if m.get("max_phone_ms", 0) > MAX_PHONE_MS:
            return "phone_too_long"
        if speech_ratio(words, m["duration_ms"]) < MIN_COVERAGE:
            return "low_coverage"
        exempt = self.source in EXEMPT_FROM_RATE_AND_ONSET
        if not exempt and timing.speech_ms and timing.speech_ms[0][0] < MIN_ONSET_MS:
            return "starts_too_early"
        rate = m.get("rate")
        spread = RATE_MADS * (self.rate_mad or 0.0)  # a spread of 0 (one clip, or equal rates) switches the rule off
        if not exempt and rate is not None and spread and abs(rate - (self.rate_median or 0.0)) > spread:
            return "rate_outlier"
        if self.loglik_floor is not None and m.get("loglik") is not None and m["loglik"] < self.loglik_floor:
            return "low_loglik"
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), sort_keys=True) + "\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "DropRules":
        return cls(**json.loads(path.read_text()))


# ---------------------------------------------------------------------------------------------------- batches, cap


def _by_rank(clips: Iterable[Clip]) -> list[Clip]:
    return sorted(clips, key=lambda c: (rank(c.clip_id), c.clip_id))


def make_batches(clips: Iterable[Clip], batch_hours: float) -> Iterator[list[Clip]]:
    """Batches of about `batch_hours`: speakers in the order of their lowest-ranked clip, each speaker whole in one
    batch even if that exceeds the target. A clip without a speaker is a unit of its own.
    """
    units: dict[tuple[str, str], list[Clip]] = defaultdict(list)
    for c in clips:
        units[("s", c.speaker) if c.speaker is not None else ("c", c.clip_id)].append(c)
    ordered = sorted((_by_rank(u) for u in units.values()), key=lambda u: (rank(u[0].clip_id), u[0].clip_id))
    batch: list[Clip] = []
    seconds = 0.0
    for unit in ordered:
        size = sum(c.duration_s for c in unit)
        if batch and seconds + size > batch_hours * 3600:
            yield batch
            batch, seconds = [], 0.0
        batch += unit
        seconds += size
    if batch:
        yield batch


def _leading(clips: list[Clip], budget_s: float) -> int:
    """How many leading clips fit in `budget_s` together; the first that would exceed it ends the count, even if a
    smaller one follows (the rule of `apply_cap_rank`).
    """
    total = 0.0
    for i, c in enumerate(clips):
        total += c.duration_s
        if total > budget_s:
            return i
    return len(clips)


class _Walk:
    """The cap walk of one speaker: clips in rank order and the seconds of surviving audio still allowed."""

    def __init__(self, clips: Iterable[Clip], hours: float):
        self.pending = _by_rank(clips)
        self.left_s = hours * 3600

    def next_chunk(self) -> list[Clip]:
        """The next clips to align: as many as fit if every one survived. Empty once the cap leaves no room."""
        n = _leading(self.pending, self.left_s)
        chunk, self.pending = self.pending[:n], self.pending[n:] if n else []
        return chunk

    def survived(self, clip: Clip) -> None:
        self.left_s -= clip.duration_s


def _walk(
    walks: list[_Walk],
    align_batch: Callable[[list[Clip]], list[Timing]],
    judge: Callable[[list[Timing]], list[str | None]],
    dropped: list[tuple[Timing, str]] | None = None,
) -> Iterator[Timing]:
    """Rounds over several speakers, one `align_batch` call per round. A round takes from each speaker the clips that
    fit its remaining budget if all survive; drops leave room, and the next round refills it from the rank order.
    """
    while True:
        owner: dict[str, tuple[_Walk, Clip]] = {}
        chunk: list[Clip] = []
        for w in walks:
            for c in w.next_chunk():
                owner[c.clip_id] = (w, c)
                chunk.append(c)
        if not chunk:
            return
        timings = align_batch(chunk)
        for timing, reason in zip(timings, judge(timings), strict=True):
            if reason is None:
                owner[timing.clip_id][0].survived(owner[timing.clip_id][1])
                yield timing
            elif dropped is not None:
                dropped.append((timing, reason))


class _Judge(Protocol):
    def apply(self, timing: Timing) -> str | None: ...


def walk_cap(
    speaker_clips: Iterable[Clip],
    align_batch: Callable[[list[Clip]], list[Timing]],
    rules: _Judge,
    hours: float = SPEAKER_CAP_HOURS,
    dropped: list[tuple[Timing, str]] | None = None,
) -> Iterator[Timing]:
    """The surviving timings of one speaker: clips in rank order, never aligning a clip that would not fit in
    `hours` of surviving audio. A clip MFA could not align is simply missing from what `align_batch` returns.
    The first clip that would not fit ends the walk unexamined, so a clip the rules would have dropped is not
    skipped to reach a smaller one behind it.
    """
    return _walk([_Walk(speaker_clips, hours)], align_batch, lambda ts: [rules.apply(t) for t in ts], dropped)


# ---------------------------------------------------------------------------------------------------- one MFA run


class Drop(NamedTuple):
    clip_id: str
    reason: str
    metrics: dict | None


_CSV_FLOATS = {
    "loglik": "overall_log_likelihood",
    "speech_loglik": "speech_log_likelihood",
    "phone_duration_deviation": "phone_duration_deviation",
    "snr": "snr",
    "intensity_deviation": "intensity_deviation",
}


def _analysis(path: Path) -> dict[str, dict]:
    """`alignment_analysis.csv` by file stem. A clip MFA could not align has a row of blank cells."""
    with path.open(newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def _metrics_of(row: dict, duration_ms: int, retried: bool) -> dict:
    metrics = {k: float(row[col]) if row[col] else None for k, col in _CSV_FLOATS.items()}
    short = row["max_running_short_interval"]
    return metrics | {
        "max_running_short_interval": int(float(short)) if short else None,
        "duration_ms": duration_ms,
        "retried": retried,
    }


def _retried(mfa_root: Path, corpus: str) -> set[str]:
    """File stems that MFA aligned only with its retry beam. kalpy logs `Retried <speaker id>-<utterance id>` at
    debug level (`-v`) in the per-job alignment logs of both passes; the database maps those ids to file names.
    Anything missing from this layout is an error, because a clip not checked would pass the beam rule.
    """
    db, logs = mfa_root / corpus / f"{corpus}.db", sorted((mfa_root / corpus / "alignment/log").glob("align.*.log"))
    if not (db.is_file() and logs):
        raise MfaFailedError(
            f"MFA left no database or alignment logs under {mfa_root / corpus}: cannot tell retried clips"
        )
    with sqlite3.connect(db) as con:
        names = dict(
            con.execute("SELECT u.speaker_id || '-' || u.id, f.name FROM utterance u JOIN file f ON f.id = u.file_id")
        )
    ids = {m for log_file in logs for m in re.findall(r"^.* - DEBUG - Retried (\S+)$", log_file.read_text(), re.M)}
    if unknown := ids - names.keys():
        raise MfaFailedError(f"MFA retried utterances {sorted(unknown)} that its database does not know")
    return {names[i] for i in ids}


def _run_mfa(tools: Tools, models: Models, corpus: Path, out: Path, mfa_root: Path) -> None:
    args = ["align", "--clean", "-v", str(corpus), str(models.dictionary), str(models.acoustic), str(out)]
    if models.g2p is not None:
        args += ["--g2p_model_path", str(models.g2p)]
    _mfa(tools, mfa_root, *args)


def _align_round(
    clips: list[Clip], language: str, tools: Tools, models: Models, provenance: str
) -> tuple[list[Timing], list[Drop]]:
    """One MFA run on `clips`. Every clip ends as one timing or one drop, or the run fails.

    A clip without a usable transcript or that does not decode is dropped before MFA starts, because one corrupt
    file aborts a whole MFA run. A clip MFA returns no TextGrid for, or a blank analysis row, is `no_alignment`
    although MFA exits 0.
    """
    drops: list[Drop] = []
    sent: dict[str, tuple[Clip, int]] = {}
    with tempfile.TemporaryDirectory(prefix="openvad-align-") as tmp:
        corpus, out, mfa_root = Path(tmp) / "corpus", Path(tmp) / "out", Path(tmp) / "mfa"
        speakers: dict[tuple[bool, str], str] = {}
        for i, clip in enumerate(clips):
            text = prepare_transcript(clip.transcript or "", language)
            if text is None:
                drops.append(Drop(clip.clip_id, "no_transcript", None))
                continue
            n = round(clip.duration_s * 16000)
            try:
                audio = decode_16k(str(data_dir() / clip.path), clip.offset_s, clip.duration_s)
                if abs(len(audio) - n) > SAMPLES_PER_FRAME:  # the check `construct` makes, so its labels would drift
                    raise DecodeError(f"{clip.path}: decoded {len(audio)} samples for {n}")
            except DecodeError as e:
                log.warning("%s: %s", clip.clip_id, e)
                drops.append(Drop(clip.clip_id, "decode_failed", None))
                continue
            stem = f"u{i:07d}"
            # A clip without a speaker is a speaker of its own, so fMLLR never adapts across unrelated voices.
            key = (clip.speaker is None, clip.clip_id if clip.speaker is None else clip.speaker)
            folder = corpus / speakers.setdefault(key, f"s{len(speakers)}")
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / f"{stem}.wav", audio, 16000, subtype="PCM_16")
            (folder / f"{stem}.lab").write_text(text + "\n", encoding="utf-8")
            sent[stem] = (clip, len(audio))
        timings: list[Timing] = []
        if sent:
            _run_mfa(tools, models, corpus, out, mfa_root)
            analysis = _analysis(out / "alignment_analysis.csv")
            textgrids = {p.stem: p for p in out.rglob("*.TextGrid")}
            if extra := (textgrids.keys() | analysis.keys()) - sent.keys():
                raise MfaFailedError(f"MFA returned results for files it was not given: {sorted(extra)[:5]}")
            retried = _retried(mfa_root, "corpus")
            for stem, (clip, n_samples) in sent.items():
                row = analysis.get(stem)
                if stem not in textgrids or row is None or not row["overall_log_likelihood"]:
                    drops.append(Drop(clip.clip_id, "no_alignment", None))
                    continue
                metrics = _metrics_of(row, round(n_samples / 16), stem in retried)
                timing = read_alignment(textgrids[stem], metrics, provenance)
                timings.append(dataclasses.replace(timing, clip_id=clip.clip_id))
    accounted = [t.clip_id for t in timings] + [d.clip_id for d in drops]
    if sorted(accounted) != sorted(c.clip_id for c in clips):
        raise MfaFailedError(f"{len(clips)} clips went in but {len(accounted)} came out as a row or a drop")
    return timings, drops


# ---------------------------------------------------------------------------------------------------- the stage


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _interval_rows(intervals: list[tuple[int, int]]) -> list[dict]:
    return [{"start": s, "end": e} for s, e in intervals]


def _table(timings: list[Timing]) -> pl.LazyFrame:
    rows = [
        {
            "clip_id": t.clip_id,
            "kind": t.kind,
            "speech_ms": _interval_rows(t.speech_ms),
            "mask_ms": _interval_rows(t.mask_ms),
            "metrics": json.dumps(t.metrics, sort_keys=True),
            "provenance": t.provenance,
        }
        for t in sorted(timings, key=lambda t: t.clip_id)
    ]
    return pl.DataFrame(rows, schema=WORDS_SCHEMA).lazy()


def _drop_table(source: str, language: str, batch: str, drops: list[Drop]) -> pl.LazyFrame:
    rows = [
        {
            "clip_id": d.clip_id,
            "source": source,
            "language": language,
            "batch": batch,
            "reason": d.reason,
            "metrics": None if d.metrics is None else json.dumps(d.metrics, sort_keys=True),
        }
        for d in sorted(drops, key=lambda d: d.clip_id)
    ]
    return pl.DataFrame(rows, schema=DROPPED_SCHEMA).lazy()


def _read_tables(paths: list[Path], schema: dict, columns: list[str] | None = None) -> pl.DataFrame:
    async def go() -> list[pl.DataFrame]:
        return await asyncio.gather(*(read_parquet(p, columns) for p in paths))

    frames = asyncio.run(go()) if paths else []
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema).select(columns or list(schema))


def _dropped_files(base: Path) -> list[Path]:
    # Kept inside each language's words folder, so deleting that folder to start over deletes its drops too.
    return sorted((base / "words").glob("*/*/dropped/*.parquet"))


def _build_dropped(base: Path) -> pl.DataFrame:
    """`work/dropped.parquet`, the union of the per-batch drop tables, written atomically."""
    frame = _read_tables(_dropped_files(base), DROPPED_SCHEMA)
    base.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({base / "dropped.parquet": frame.sort("source", "language", "batch", "clip_id").lazy()}))
    return frame


def _read_index(base: Path, source: str) -> list[Clip]:
    path = base / "index" / f"{source}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: run the index stage first")
    frame = asyncio.run(read_parquet(path, [f.name for f in dataclasses.fields(Clip)]))
    return [Clip(**row) for row in frame.iter_rows(named=True)]


def _digest(clip_ids: Iterable[str]) -> str:
    return hashlib.blake2b("\0".join(clip_ids).encode(), digest_size=16).hexdigest()


def _align_batch(
    batch: list[Clip],
    full: dict[str | None, list[Clip]],
    name: str,
    source: str,
    language: str,
    words_dir: Path,
    run: Callable[[list[Clip]], tuple[list[Timing], list[Drop]]],
) -> None:
    """Align one batch to its two tables. The rules are fitted on the first batch and saved before its tables are
    written, and the words table is written last, so its existence means the batch is finished.
    """
    rules_path = words_dir / "rules.json"
    rules = DropRules.load(rules_path) if rules_path.is_file() else None
    if rules is None and name != "00000":
        raise StateMismatchError(f"{rules_path} is missing although batch 00000 is finished: delete {words_dir}")
    unaligned: list[Drop] = []

    def align(chunk: list[Clip]) -> list[Timing]:
        timings, drops = run(chunk)
        unaligned.extend(drops)
        return timings

    def judge(timings: list[Timing]) -> list[str | None]:
        nonlocal rules
        if rules is None and timings:
            rules = DropRules.fit(timings, source)
            rules.save(rules_path)
        return [rules.apply(t) for t in timings] if rules else []

    walks = [_Walk(clips, hours) for clips, hours in _walks_of(batch, full)]
    ruled: list[tuple[Timing, str]] = []
    kept = list(_walk(walks, align, judge, ruled))
    if rules is None:
        raise MfaFailedError(f"{source}/{language} batch {name}: no clip aligned, so the drop rules cannot be fitted")
    drops = unaligned + [Drop(t.clip_id, reason, t.metrics) for t, reason in ruled]
    (words_dir / "dropped").mkdir(parents=True, exist_ok=True)
    asyncio.run(
        write_parquet(
            {
                words_dir / "dropped" / f"{name}.parquet": _drop_table(source, language, name, drops),
                words_dir / f"{name}.parquet": _table(kept),
            }
        )
    )
    log.info("%s/%s batch %s: %d clips, %d kept, %d dropped", source, language, name, len(batch), len(kept), len(drops))


def _walks_of(batch: list[Clip], full: dict[str | None, list[Clip]]) -> list[tuple[list[Clip], float]]:
    """The clips and cap of each walk in a batch: a speaker's whole list (the walk takes what fits, refills included),
    and the batch's own clips without a speaker, which are not capped.
    """
    speakers = dict.fromkeys(c.speaker for c in batch if c.speaker is not None)
    walks = [(full[speaker], SPEAKER_CAP_HOURS) for speaker in speakers]
    if loose := [c for c in batch if c.speaker is None]:
        walks.append((loose, float("inf")))
    return walks


def _align_language(
    cfg: BuildConfig,
    source: str,
    language: str,
    clips: list[Clip],
    tools: Tools,
    models: Models,
) -> None:
    base = work_dir(cfg)
    words_dir = base / "words" / source / language
    # Reserved speakers feed babble and competing talkers only and are never foreground (R-i), so they are not aligned.
    by_speaker: dict[str | None, list[Clip]] = defaultdict(list)
    for c in clips:
        if not c.reserved:
            by_speaker[c.speaker].append(c)
    full = {speaker: _by_rank(group) for speaker, group in by_speaker.items()}
    first_round = [
        c
        for speaker, group in full.items()
        for c in (group if speaker is None else group[: _leading(group, SPEAKER_CAP_HOURS * 3600)])
    ]
    batches = list(make_batches(first_round, BATCH_HOURS))
    provenance = _provenance(tools, models)
    state = {
        "provenance": provenance,
        "batch_hours": BATCH_HOURS,
        "batches": [_digest(c.clip_id for c in b) for b in batches],
    }
    state_path = words_dir / "state.json"
    if state_path.is_file() and json.loads(state_path.read_text()) != state:
        raise StateMismatchError(
            f"{words_dir} was made with other batches or tools than this run: delete it to start over"
        )
    if batches and not state_path.is_file():
        _write_atomically(state_path, json.dumps(state, sort_keys=True) + "\n")

    def run(chunk: list[Clip]) -> tuple[list[Timing], list[Drop]]:
        return _align_round(chunk, language, tools, models, provenance)

    for i, batch in enumerate(batches):
        name = f"{i:05d}"
        if (words_dir / f"{name}.parquet").is_file():
            log.info("%s/%s batch %s is finished", source, language, name)
            continue
        _align_batch(batch, full, name, source, language, words_dir, run)


def run_align(cfg: BuildConfig, languages: list[str] | None = None) -> None:
    """Align the MFA sources of `cfg`, one MFA run per batch, resumable per batch; see the module docstring."""
    tools = check_tools()
    table = load_models()
    unknown = [n for n in cfg.sources if n not in SOURCES]
    if unknown:
        raise ValueError(f"no index adapter for source {unknown[0]!r}; have {sorted(SOURCES)}")
    base = work_dir(cfg)
    per_source: dict[str, dict[str, list[Clip]]] = {}
    for name in cfg.sources:
        if hasattr(SOURCES[name], "timings"):
            continue  # its labels do not come from MFA
        by_language: dict[str, list[Clip]] = defaultdict(list)
        for c in _read_index(base, name):
            if languages is None or c.language in languages:
                by_language[c.language].append(c)
        per_source[name] = dict(by_language)
    specs = {lang: spec_for(table, lang) for by_language in per_source.values() for lang in by_language}
    if languages is not None and (bad := [lang for lang in languages if lang not in table]):
        raise ValueError(f"no row for language {bad[0]!r} in {MODELS_TSV.name}; have {sorted(table)}")

    fetched: dict[str, Models] = {}
    try:
        for name, by_language in per_source.items():
            for language in sorted(by_language):
                if language not in fetched:
                    fetched[language] = fetch_models(tools, specs[language])
                _align_language(cfg, name, language, by_language[language], tools, fetched[language])
    finally:
        _build_dropped(base)
    _write_manifest(cfg, tools, per_source, fetched)


def _write_manifest(
    cfg: BuildConfig, tools: Tools, per_source: dict[str, dict[str, list[Clip]]], fetched: dict[str, Models]
) -> None:
    base = work_dir(cfg)
    dropped = _read_tables(_dropped_files(base), DROPPED_SCHEMA)
    summary: dict[str, dict] = {}
    for source, by_language in per_source.items():
        for language, clips in by_language.items():
            words = _read_tables(
                sorted((base / "words" / source / language).glob("*.parquet")), WORDS_SCHEMA, ["clip_id"]
            ).height
            mine = dropped.filter((pl.col("source") == source) & (pl.col("language") == language))
            summary.setdefault(source, {})[language] = {
                "index_clips": len(clips),
                "reserved_not_aligned": sum(c.reserved for c in clips),
                "kept": words,
                "dropped": dict(mine.group_by("reason").len().sort("reason").iter_rows()),
                "over_cap_not_aligned": sum(not c.reserved for c in clips) - words - mine.height,
                "models": fetched[language].info if language in fetched else None,
            }
    out = base / "words"
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(
        out,
        {
            "stage": "align",
            "config": cfg.name,
            "aligner": {"mfa": tools.version, "commit": tools.commit},
            "per_source": summary,
        },
    )


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset align", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    parser.add_argument("--languages", help="comma-separated language codes; default: every language of the index")
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_align({"slice": SLICE, "full": FULL}[ns.config], ns.languages.split(",") if ns.languages else None)
