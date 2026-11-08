# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The hand check and the quality bar of the `align` stage (ADR 0025, spec section "Quality bar").

`handcheck-sample` draws clips from the words tables across the log-likelihood quantiles and writes, for the owner to
label blind in Praat, a 16 kHz WAV and an empty TextGrid (one tier `speech`, nothing labelled) per clip.
`handcheck-score` compares the owner's tier with the labels training would see. Everything stays on the share
(ADR 0024), under `work/<config>/handcheck/`:

- `clips/`: the only folder the labeller opens. Files are named `h001.wav` and `h001.TextGrid`, numbered in `rank`
  order across languages, so a name carries neither the language, the source nor the score.
- `key.json`: file id to clip, language, score, and the MFA intervals and mask that scoring needs; not in `clips/`.
- `report.json`: the output of `handcheck-score`.
"""

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

import soundfile as sf

from openvad_train.dataset.align import WORDS_SCHEMA, _read_index, _read_tables, _tier, parse_textgrid
from openvad_train.dataset.codecs import decode_16k
from openvad_train.dataset.config import FULL, SLICE, BuildConfig, work_dir
from openvad_train.dataset.labels import apply_gap_rule, rasterise
from openvad_train.dataset.schema import rank
from openvad_train.fetch.common import data_dir


log = logging.getLogger(__name__)

HANDCHECK_LANGUAGES = ("en", "de", "it", "ca")
PER_LANGUAGE = 25
WITHIN_MS = (20, 50)
# Above this many hours in the pool a failing language is the owner's decision, not dropped (ADR 0025).
OWNER_DECIDES_OVER_HOURS = 500.0

_WORDS_COLUMNS = ["clip_id", "speech_ms", "mask_ms", "metrics"]


def handcheck_dir(cfg: BuildConfig) -> Path:
    """Beside the words tables on the share, not under the dataset tree that `publish` writes."""
    return work_dir(cfg) / "handcheck"


# ---------------------------------------------------------------------------------------------------- quality bar


class QualityBarUnsetError(RuntimeError):
    """The bar has no thresholds yet: the pilot sets them, and until then no language is judged."""


class QualityBar:
    """The automatic bar per language: the most alignment failures and the most clips removed by the drop rules that a
    language may have, each as a share of its clips and inclusive. Both are unset until the pilot (Task 15, step 6)
    sets them from its distribution; an unset bar raises instead of passing, because passing everything would let a
    language into the full run unchecked and failing everything would drop every language.
    """

    def __init__(self, max_failure_rate: float | None = None, max_drop_share: float | None = None):
        self.max_failure_rate = max_failure_rate
        self.max_drop_share = max_drop_share

    def passes(self, failure_rate: float, drop_share: float) -> bool:
        if self.max_failure_rate is None or self.max_drop_share is None:
            raise QualityBarUnsetError("the quality bar has no thresholds yet: the pilot sets them")
        return failure_rate <= self.max_failure_rate and drop_share <= self.max_drop_share


def apply_failure_rule(language: str, hours: float, passed: bool) -> Literal["keep", "drop", "owner"]:
    """A language that passed is kept; one that failed is dropped, unless its pool is over 500 hours (exactly 500 is
    dropped), when the owner decides. `language` is only for the caller's records.
    """
    if passed:
        return "keep"
    return "owner" if hours > OWNER_DECIDES_OVER_HOURS else "drop"


# ---------------------------------------------------------------------------------------------------- sample


def _stratified(scored: list[tuple[float, str]], k: int) -> list[str]:
    """`k` clip ids spread over the scores: the clips sorted by score are cut into `k` equal strata and each gives the
    clip with the lowest `rank`, so the draw is a function of the table alone.
    """
    ordered = sorted(scored)
    n = len(ordered)
    return [min((c for _, c in ordered[i * n // k : (i + 1) * n // k]), key=lambda c: (rank(c), c)) for i in range(k)]


def _empty_textgrid(n_samples: int) -> str:
    """Praat's long format with the `speech` tier as one blank interval over the whole clip (the way Praat represents
    an empty interval tier, which cannot have none), so no speech is labelled.
    """
    end = f"{n_samples / 16000:.7f}"
    return "\n".join(
        [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            f"xmax = {end}",
            "tiers? <exists>",
            "size = 1",
            "item []:",
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "speech"',
            "        xmin = 0",
            f"        xmax = {end}",
            "        intervals: size = 1",
            "        intervals [1]:",
            "            xmin = 0",
            f"            xmax = {end}",
            '            text = ""',
            "",
        ]
    )


def sample_handcheck(
    cfg: BuildConfig, per_language: int = PER_LANGUAGE, languages: tuple[str, ...] = HANDCHECK_LANGUAGES
) -> Path:
    """Write the hand-check sample of `cfg` and return its directory (hand `<dir>/clips` to the labeller).

    Per language, the clips of every words table of `align` (clips that passed the drop rules) are sorted by
    log-likelihood and cut into `per_language` equal strata, each giving the clip of lowest `rank`. Each clip is
    decoded by `decode_16k` as `align` and `construct` do and written as a 16 kHz WAV. Refuses to run over an
    existing sample, whose TextGrids may hold the owner's labels.
    """
    out = handcheck_dir(cfg)
    if out.exists():
        raise FileExistsError(f"{out} exists and may hold labels: move it away to draw a new sample")
    base = work_dir(cfg)
    picked: dict[str, dict] = {}
    short = []
    for language in languages:
        rows = [
            (source_dir.parent.name, r)
            for source_dir in sorted((base / "words").glob(f"*/{language}"))
            for r in _read_tables(sorted(source_dir.glob("*.parquet")), WORDS_SCHEMA, _WORDS_COLUMNS).iter_rows(
                named=True
            )
            if json.loads(r["metrics"]).get("loglik") is not None
        ]
        by_id = {r["clip_id"]: (source, r) for source, r in rows}
        if len(by_id) < per_language:
            short.append(f"{language} ({len(by_id)})")
            continue
        scored = [(json.loads(r["metrics"])["loglik"], r["clip_id"]) for _, r in by_id.values()]
        for clip_id in _stratified(scored, per_language):
            source, r = by_id[clip_id]
            picked[clip_id] = {
                "language": language,
                "source": source,
                "score": json.loads(r["metrics"])["loglik"],
                "speech_ms": [[w["start"], w["end"]] for w in r["speech_ms"]],
                "mask_ms": [[w["start"], w["end"]] for w in r["mask_ms"]],
            }
    if short:
        raise ValueError(f"fewer than {per_language} aligned clips with a score for: {', '.join(short)}")

    clips = {c.clip_id: c for source in {p["source"] for p in picked.values()} for c in _read_index(base, source)}
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "clips").mkdir(parents=True)
    files = {}
    for n, clip_id in enumerate(sorted(picked, key=lambda c: (rank(c), c)), 1):
        clip = clips[clip_id]
        audio = decode_16k(str(data_dir() / clip.path), clip.offset_s, clip.duration_s)
        name = f"h{n:03d}"
        sf.write(tmp / "clips" / f"{name}.wav", audio, 16000, subtype="PCM_16")
        (tmp / "clips" / f"{name}.TextGrid").write_text(_empty_textgrid(len(audio)), encoding="utf-8")
        files[name] = {"clip_id": clip_id, "n_samples": len(audio)} | picked[clip_id]
    key = {"gap_ms": cfg.gap_ms, "per_language": per_language, "files": files}
    (tmp / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    log.info("hand check: %d clips in %s", len(files), out / "clips")
    return out


# ---------------------------------------------------------------------------------------------------- score


def pair_boundaries(owner: list[int], mfa: list[int]) -> list[tuple[int, int]]:
    """One-to-one nearest pairing of two lists of boundary times (ms), as (owner, mfa) index pairs.

    All pairs are ranked by distance, ties by the two indices, and taken in that order unless one of the two is
    already paired. No distance limit applies, so a far boundary is still paired (and then misses the 20 and 50 ms
    windows); the boundaries left over, `abs(len(owner) - len(mfa))` of them, are unpaired.
    """
    candidates = sorted((abs(o - m), i, j) for i, o in enumerate(owner) for j, m in enumerate(mfa))
    used_owner: set[int] = set()
    used_mfa: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i not in used_owner and j not in used_mfa:
            used_owner.add(i)
            used_mfa.add(j)
            pairs.append((i, j))
    return pairs


def _owner_speech_ms(textgrid: Path) -> list[tuple[int, int]]:
    """The owner's speech intervals in ms: every interval of the `speech` tier whose text is not blank. So a label
    other than `speech` counts as speech too, and an empty or whitespace-only one does not.
    """
    tier = _tier(parse_textgrid(textgrid.read_text(encoding="utf-8")), "speech")
    return [
        (round(a * 1000), round(b * 1000)) for a, b, text in tier if text.strip() and round(b * 1000) > round(a * 1000)
    ]


def _boundaries(runs: list[tuple[int, int]]) -> dict[str, list[int]]:
    return {"onsets": [s for s, _ in runs], "offsets": [e for _, e in runs]}


def _share(n: int, of: int) -> float | None:
    return n / of if of else None


def _summary(c: Counter) -> dict:
    return {
        "clips": c["clips"],
        "frames": c["frames"],
        "frame_agreement": _share(c["agree"], c["frames"]),
        **{
            kind: {
                "paired": c[f"{kind}_paired"],
                **{f"within_{w}_ms": _share(c[f"{kind}_within_{w}"], c[f"{kind}_paired"]) for w in WITHIN_MS},
                "unpaired_owner": c[f"{kind}_unpaired_owner"],
                "unpaired_mfa": c[f"{kind}_unpaired_mfa"],
            }
            for kind in ("onsets", "offsets")
        },
    }


def score_handcheck(path: Path) -> dict:
    """Compare the owner's `speech` tier with the labels training would see, and write `report.json` beside the key.

    `path` is the sample directory that `sample_handcheck` returned. The training labels are the MFA word intervals
    after the gap rule with the G of the build configuration, rasterised on the 10 ms grid by `labels.rasterise`
    with the clip's mask. The owner's intervals get the same gap rule, because the comparison is of the labels
    training would see on both sides, so a labeller who marks the words and pauses by the same convention scores 1.0.
    An interval counts as speech if its text is not blank (see `_owner_speech_ms`).

    - Frame agreement: the share of frames with the same label, over the frames the mask leaves unmasked, summed
      over the clips (not an average of per-clip shares).
    - Onsets and offsets: the starts and ends of the owner's speech runs (after the same gap rule)
      against those of the MFA runs after the gap rule, in ms, paired per clip by `pair_boundaries`. A boundary
      inside a masked stretch is not excluded. `within_20_ms` and `within_50_ms` are shares of the paired
      boundaries; the unpaired ones are counted apart (`unpaired_owner`, `unpaired_mfa`), because they are misses
      of a different kind.

    The report has the totals and `per_language`. Every sampled clip passed MFA, so a TextGrid with no speech in it
    means a clip not yet labelled: it is left out of every total and counted in `empty_textgrids`.
    """
    key = json.loads((path / "key.json").read_text(encoding="utf-8"))
    gap_ms = key["gap_ms"]
    totals: Counter = Counter()
    by_language: dict[str, Counter] = {}
    for name, f in sorted(key["files"].items()):
        owner_ms = _owner_speech_ms(path / "clips" / f"{name}.TextGrid")
        by_language.setdefault(f["language"], Counter())
        if not owner_ms:
            totals["empty"] += 1
            by_language[f["language"]]["empty"] += 1
            continue
        mfa_ms = apply_gap_rule([(a, b) for a, b in f["speech_ms"]], gap_ms)
        mask_ms = [(a, b) for a, b in f["mask_ms"]]
        owner_runs = apply_gap_rule(owner_ms, gap_ms)
        theirs = rasterise(mfa_ms, mask_ms, f["n_samples"])
        ours = rasterise(owner_runs, [], f["n_samples"])
        unmasked = theirs != 2
        c = Counter(clips=1, frames=int(unmasked.sum()), agree=int((theirs == ours)[unmasked].sum()))
        mine, reference = _boundaries(owner_runs), _boundaries(mfa_ms)
        for kind in ("onsets", "offsets"):
            pairs = pair_boundaries(mine[kind], reference[kind])
            errors = [abs(mine[kind][i] - reference[kind][j]) for i, j in pairs]
            c[f"{kind}_paired"] = len(pairs)
            for w in WITHIN_MS:
                c[f"{kind}_within_{w}"] = sum(e <= w for e in errors)
            c[f"{kind}_unpaired_owner"] = len(mine[kind]) - len(pairs)
            c[f"{kind}_unpaired_mfa"] = len(reference[kind]) - len(pairs)
        totals += c
        by_language[f["language"]].update(c)
    report = {
        "gap_ms": gap_ms,
        "empty_textgrids": totals["empty"],
        **_summary(totals),
        "per_language": {
            lang: _summary(c) | {"empty_textgrids": c["empty"]} for lang, c in sorted(by_language.items())
        },
    }
    tmp = path / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path / "report.json")
    return report


# ---------------------------------------------------------------------------------------------------- stages


def main_sample(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset handcheck-sample", description=__doc__)
    parser.add_argument("--config", choices=("slice", "full"), required=True)
    parser.add_argument("--per-language", type=int, default=PER_LANGUAGE)
    parser.add_argument("--languages", default=",".join(HANDCHECK_LANGUAGES), help="comma-separated language codes")
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sample_handcheck({"slice": SLICE, "full": FULL}[ns.config], ns.per_language, tuple(ns.languages.split(",")))
    print(f"hand over: {out / 'clips'}")


def main_score(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dataset handcheck-score", description=__doc__)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--config", choices=("slice", "full"))
    where.add_argument("--path", type=Path, help="the sample directory, if it was moved")
    ns = parser.parse_args(args)
    path = ns.path or handcheck_dir({"slice": SLICE, "full": FULL}[ns.config])
    print(json.dumps(score_handcheck(path), indent=1, sort_keys=True))
