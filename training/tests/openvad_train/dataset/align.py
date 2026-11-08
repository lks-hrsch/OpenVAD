# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import csv
import dataclasses
import json
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter
from collections.abc import Collection
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from align_support import FIXTURES, Killed, install, make_clip
from openvad_train.dataset import align, index
from openvad_train.dataset.align import (
    DropRules,
    MfaFailedError,
    StateMismatchError,
    ToolsNotBuiltError,
    check_tools,
    make_batches,
    parse_textgrid,
    prepare_transcript,
    read_alignment,
    run_align,
    walk_cap,
)
from openvad_train.dataset.config import SLICE, work_dir
from openvad_train.dataset.models import load_models
from openvad_train.dataset.parquet_io import read_parquet, write_parquet
from openvad_train.dataset.schema import Clip, Timing, rank


# ---------------------------------------------------------------------------------------------------- transcripts


def test_empty_or_digit_transcripts_drop_with_reason():
    assert prepare_transcript("...", "en") is None
    assert prepare_transcript("12 34", "en") is None
    assert "<unk>" in (prepare_transcript("i have 12 dogs", "en") or "")


def test_edge_punctuation_goes_and_inner_apostrophes_stay():
    assert prepare_transcript('  "Don\'t," she said -- (twice)!', "en") == "Don't she said twice"
    assert prepare_transcript("3rd place, 4.5 km", "en") == "<unk> place <unk> km"


def test_thai_tone_and_vowel_marks_at_the_end_of_a_word_stay():
    assert (
        prepare_transcript("\u0e17\u0e35\u0e48\u0e19\u0e35\u0e48 \u0e14\u0e35", "th")
        == "\u0e17\u0e35\u0e48\u0e19\u0e35\u0e48 \u0e14\u0e35"
    )


def test_text_that_is_not_precomposed_is_composed_and_keeps_its_accents():
    assert prepare_transcript("re\u0301sume\u0301 u\u0308ber.", "de") == "r\u00e9sum\u00e9 \u00fcber"
    assert prepare_transcript("\u00ab r\u00e9sum\u00e9 \u00bb \u20ac5 \u2014 ok", "fr") == "r\u00e9sum\u00e9 <unk> ok"


def test_format_characters_at_a_word_edge_go_and_a_lone_combining_mark_is_no_word():
    assert prepare_transcript("\ufeffhallo welt\u200b", "de") == "hallo welt"  # byte order mark, zero-width space
    assert prepare_transcript("\u200fhallo\u200e \u200b", "ar") == "hallo"  # direction marks, a lone zero-width space
    assert prepare_transcript("hallo \u0301 welt", "de") == "hallo welt"
    assert prepare_transcript("\u0301 \u200b", "de") is None
    assert prepare_transcript("zero\u200bwidth", "de") == "zero\u200bwidth"  # inside a word it stays


def test_chinese_and_japanese_digit_runs_become_unk_inside_a_token():
    assert prepare_transcript("我有12只狗", "zh") == "我有 <unk> 只狗"
    assert prepare_transcript("12", "ja") is None


# ---------------------------------------------------------------------------------------------------- the tools


def _fake_tree(root: Path, commit: str | None = "g93411fc0d") -> None:
    (root / "tools/kaldi/dist/bin").mkdir(parents=True)
    (root / "tools/mfa/.venv/bin").mkdir(parents=True)
    (root / "tools/mfa/.venv/bin/mfa").write_text("")
    version = root / "tools/montreal-forced-aligner/montreal_forced_aligner/_version.py"
    version.parent.mkdir(parents=True)
    version.write_text(f"__version__ = version = '3.4.3'\n__commit_id__ = commit_id = {commit!r}\n")


def test_tools_must_be_built(tmp_path):
    with pytest.raises(ToolsNotBuiltError, match="just build-mfa"):
        check_tools(tmp_path)


def test_tools_must_be_built_from_the_pinned_commit(tmp_path, monkeypatch):
    _fake_tree(tmp_path)
    monkeypatch.setattr(align, "_gitlink", lambda root: "93411fc0df3bc770dfd34c4282dc136fdf6e8dcb")
    assert check_tools(tmp_path) == align.Tools(tmp_path, "3.4.3", "93411fc0df3bc770dfd34c4282dc136fdf6e8dcb")
    monkeypatch.setattr(align, "_gitlink", lambda root: "1111111" + "0" * 33)
    with pytest.raises(ToolsNotBuiltError, match=r"93411fc0d.*just build-mfa"):
        check_tools(tmp_path)


def test_a_build_without_a_commit_is_refused(tmp_path, monkeypatch):
    _fake_tree(tmp_path, commit=None)
    monkeypatch.setattr(align, "_gitlink", lambda root: "93411fc0df3bc770dfd34c4282dc136fdf6e8dcb")
    with pytest.raises(ToolsNotBuiltError, match="no version or commit"):
        check_tools(tmp_path)


def test_run_align_stops_before_any_batch_if_the_tools_are_wrong(monkeypatch):
    monkeypatch.setattr(align, "_gitlink", lambda root: "1" * 40)
    monkeypatch.setattr(align, "_run_mfa", lambda *_: pytest.fail("a batch ran"))
    with pytest.raises(ToolsNotBuiltError, match="just build-mfa"):
        run_align(SLICE, None)


def _mfa_logs(tmp_path, **logs: str) -> None:
    """A corpus database and alignment logs; ids 1-1 and 3-2 are files u0000000 and u0000001."""
    corpus = tmp_path / "corpus"
    (corpus / "alignment/log").mkdir(parents=True)
    with closing(sqlite3.connect(corpus / "corpus.db")) as con, con:
        con.execute("CREATE TABLE file (id INTEGER, name TEXT)")
        con.execute("CREATE TABLE utterance (id INTEGER, file_id INTEGER, speaker_id INTEGER)")
        con.executemany("INSERT INTO file VALUES (?, ?)", [(1, "u0000000"), (2, "u0000001")])
        con.executemany("INSERT INTO utterance VALUES (?, ?, ?)", [(1, 1, 1), (2, 2, 3)])
    for name, text in logs.items():
        (corpus / "alignment/log" / name.replace("_", ".", 1).replace("_", ".")).write_text(text)


START = "2026-11-07 22:02:05,400 - kalpy.align - 6208811008 - 1 - DEBUG - Aligning with /root/final.mdl\n"
LINE = "2026-11-07 22:02:05,469 - kalpy.align - 6208811008 - 1 - DEBUG - {} {}\n"


def test_retried_clips_come_from_the_debug_log_through_the_database(tmp_path):
    _mfa_logs(tmp_path, align_1_log=START + LINE.format("Retried", "1-1") + LINE.format("Processing", "3-2"))
    assert align._retried(tmp_path, "corpus") == {"u0000000"}


def test_only_the_pass_whose_alignment_is_written_counts(tmp_path):
    # Retried only under the speaker-independent model: the written fMLLR alignment did not need it.
    _mfa_logs(
        tmp_path,
        align_1_log=START + LINE.format("Retried", "1-1"),
        align_1_fmllr_log=START + LINE.format("Retried", "3-2"),
    )
    assert align._retried(tmp_path, "corpus") == {"u0000001"}


def test_a_log_without_kalpys_start_message_is_an_error_not_an_empty_set(tmp_path):
    _mfa_logs(tmp_path, align_1_fmllr_log=START, align_2_fmllr_log=LINE.format("Retired", "1-1"))
    with pytest.raises(MfaFailedError, match=r"align.2.fmllr.log.*retries cannot be read"):
        align._retried(tmp_path, "corpus")


def test_mfas_own_aligning_with_model_line_does_not_satisfy_the_canary(tmp_path):
    mfa_line = "2026-11-07 22:02:05,300 - kalpy.align - 6208811008 - 1 - DEBUG - Aligning with model: /root/final.mdl\n"
    _mfa_logs(tmp_path, align_1_fmllr_log=mfa_line + LINE.format("Retried", "1-1"))
    with pytest.raises(MfaFailedError, match=r"align.1.fmllr.log.*retries cannot be read"):
        align._retried(tmp_path, "corpus")
    log_file = tmp_path / "corpus/alignment/log/align.1.fmllr.log"
    log_file.write_text(mfa_line + START + LINE.format("Retried", "1-1"))
    assert align._retried(tmp_path, "corpus") == {"u0000000"}


def test_an_unknown_utterance_or_missing_logs_are_errors(tmp_path):
    _mfa_logs(tmp_path, align_1_log=START + LINE.format("Retried", "9-9"))
    with pytest.raises(MfaFailedError, match="does not know"):
        align._retried(tmp_path, "corpus")
    (tmp_path / "corpus/alignment/log/align.1.log").unlink()
    with pytest.raises(MfaFailedError, match="cannot tell retried clips"):
        align._retried(tmp_path, "corpus")


# ---------------------------------------------------------------------------------------------------- real MFA output


def test_a_real_textgrid_parses_to_both_tiers():
    tiers = parse_textgrid((FIXTURES / "u000002.TextGrid").read_text(encoding="utf-8"))
    assert set(tiers) == {"words", "phones"}
    assert tiers["words"][1] == (0.03, 0.23, "wir")
    assert tiers["phones"][2] == (0.06, 0.14, "i\u02d0")


def test_a_short_format_textgrid_is_refused():
    with pytest.raises(ValueError, match="long format"):
        parse_textgrid('"ooTextFile"\n"TextGrid"\n0\n1\n<exists>\n')


def _speech_tier(*cells: str, size: int | None = None) -> str:
    """A one-tier TextGrid whose `cells` are the raw `text = ...` values (quotes and line breaks included)."""
    lines = ['class = "IntervalTier"', 'name = "speech"', f"intervals: size = {len(cells) if size is None else size}"]
    for i, cell in enumerate(cells):
        lines += [f"intervals [{i + 1}]:", f"xmin = {i}", f"xmax = {i + 1}", f"text = {cell}"]
    return "\n".join(lines) + "\n"


def test_a_label_that_spans_lines_is_one_interval_with_its_text():
    tiers = parse_textgrid(_speech_tier('"first\nxmin = 9\nsecond ""quoted"""', '""'))
    assert tiers["speech"] == [(0.0, 1.0, 'first\nxmin = 9\nsecond "quoted"'), (1.0, 2.0, "")]


def test_a_tier_with_fewer_intervals_than_it_says_is_refused():
    with pytest.raises(ValueError, match="says 3 intervals but 2 were read"):
        parse_textgrid(_speech_tier('""', '"speech"', size=3))


def test_a_label_that_is_never_closed_is_refused():
    with pytest.raises(ValueError, match="never closed"):
        parse_textgrid(_speech_tier('"speech\nmore'))


def test_real_mfa_output_gives_word_intervals_in_ms_and_the_oov_share():
    csv_row = next(csv.DictReader((FIXTURES / "alignment_analysis.csv").open(newline="")))
    t = read_alignment(FIXTURES / "u000002.TextGrid", {"loglik": float(csv_row["overall_log_likelihood"])}, "prov")
    assert (t.clip_id, t.kind, t.mask_ms, t.provenance) == ("u000002", "word", [], "prov")
    # Silence is no speech; `<unk>` (a word MFA could not look up, `spn` in the phones tier) is, and 1.32 s long.
    assert t.speech_ms == [(30, 230), (230, 490), (490, 1810), (1970, 2120)]
    m = t.metrics or {}
    assert (m["n_words"], m["oov_words"], m["oov_share"]) == (4, 1, 0.25)
    assert (m["n_phones"], m["max_word_ms"], m["max_phone_ms"], m["first_word_ms"]) == (14, 1320, 1320, 30)
    assert m["max_phone_ms_without_spn"] == 120  # the `spn` of the unknown word is the 1320 ms phone
    assert m["rate"] == pytest.approx(14 / 1.93)
    assert m["loglik"] == pytest.approx(-59.8957625679)


def test_a_blank_analysis_row_is_what_a_failed_clip_looks_like():
    rows = {r["file"]: r for r in csv.DictReader((FIXTURES / "alignment_analysis.csv").open(newline=""))}
    assert rows["u000003"]["overall_log_likelihood"] == ""
    assert not (FIXTURES / "u000003.TextGrid").exists()
    assert align._metrics_of(rows["u000002"], 2299, retried=False)["max_running_short_interval"] == 0


# ---------------------------------------------------------------------------------------------------- drop rules


def timing(words, duration=1000, rate=3.0, loglik=-50.0, max_phone=100, retried=False) -> Timing:
    metrics = {"duration_ms": duration, "rate": rate, "loglik": loglik, "max_phone_ms": max_phone, "retried": retried}
    return Timing("c", "word", list(words), [], metrics, "")


def rules(source="x") -> DropRules:
    return DropRules(source, loglik_floor=-60.0, rate_median=3.0, rate_mad=1.0)


def test_rule_word_over_1_5_s_drops():
    assert rules().apply(timing(words=[(0, 1600)])) == "word_too_long"
    assert rules().apply(timing(words=[(100, 1600)])) is None


def test_rule_phone_over_0_5_s_drops():
    assert rules().apply(timing(words=[(100, 900)], max_phone=501)) == "phone_too_long"


def test_rule_first_word_in_20_ms_drops():
    assert rules().apply(timing(words=[(10, 500)])) == "starts_too_early"
    assert rules().apply(timing(words=[(20, 500)])) is None


def test_rule_low_coverage_drops():
    assert rules().apply(timing(words=[(0, 100)], duration=1000)) == "low_coverage"
    assert rules().apply(timing(words=[(100, 300)], duration=1000)) is None


def test_a_clip_without_words_has_no_coverage():
    assert rules().apply(timing(words=[])) == "low_coverage"


def test_rule_rate_more_than_three_mads_from_the_median_drops():
    # The deviation is scaled by 1.4826, so the limit is 3 * 1.4826 * 1.0 = 4.4478 from the median 3.0.
    assert rules().apply(timing(words=[(100, 500)], rate=7.5)) == "rate_outlier"
    assert rules().apply(timing(words=[(100, 500)], rate=-1.5)) == "rate_outlier"
    assert rules().apply(timing(words=[(100, 500)], rate=7.4)) is None


def test_a_clean_population_loses_well_under_one_percent_to_the_rate_rule():
    rates = np.random.default_rng(0).normal(12.0, 2.0, 20_000)
    batch = [timing([(100, 500)], rate=float(r)) for r in rates]
    fitted = DropRules.fit(batch, "x")
    dropped = sum(fitted.apply(t) == "rate_outlier" for t in batch)
    assert dropped / len(batch) < 0.005  # 0.27% expected; the unscaled deviation gave 4.3%
    assert fitted.mad_scale == 1.4826


def test_rule_low_loglik_and_the_wider_beam_drop():
    assert rules().apply(timing(words=[(100, 500)], loglik=-60.1)) == "low_loglik"
    assert rules().apply(timing(words=[(100, 500)], loglik=-60.0)) is None
    assert rules().apply(timing(words=[(100, 500)], retried=True)) == "needs_wider_beam"


def test_speech_commands_is_exempt_from_rate_and_onset():
    exempt = rules(source="speech-commands")
    assert exempt.apply(timing(words=[(10, 500)], rate=100.0)) is None
    assert rules().apply(timing(words=[(10, 500)], rate=100.0)) == "starts_too_early"
    assert exempt.apply(timing(words=[(10, 1700)])) == "word_too_long"


def test_a_rule_that_could_not_be_fitted_is_skipped():
    unfitted = DropRules("x", None, None, None)
    assert unfitted.apply(timing(words=[(100, 500)], rate=1000.0, loglik=-1e9)) is None
    constant = DropRules("x", None, 3.0, 0.0)  # all rates equal: a spread of 0 would drop any other rate
    assert constant.apply(timing(words=[(100, 500)], rate=3.5)) is None


def test_fit_takes_the_5th_percentile_and_the_median_and_mad_of_the_rate():
    batch = [timing([(100, 500)], loglik=-50.0 - i, rate=2.0 + i % 3) for i in range(21)]
    fitted = DropRules.fit(batch, "x")
    assert fitted.loglik_floor == pytest.approx(np.quantile([-50.0 - i for i in range(21)], 0.05))
    assert (fitted.rate_median, fitted.rate_mad) == (3.0, 1.0)
    assert DropRules.fit([], "x") == DropRules("x", None, None, None)


def test_thresholds_are_frozen_after_the_calibration_batch(tmp_path):
    r = DropRules.fit([timing([(100, 500)], loglik=-50.0 - i) for i in range(10)], "common-voice", {"clips": 10})
    r.save(tmp_path / "sub/r.json")
    loaded = DropRules.load(tmp_path / "sub/r.json")
    assert loaded == r
    assert loaded.calibration == {"clips": 10}
    assert json.loads((tmp_path / "sub/r.json").read_text())["mad_scale"] == 1.4826
    assert loaded.loglik_floor == r.loglik_floor
    assert not list(tmp_path.rglob("*.tmp"))


# ---------------------------------------------------------------------------------------------------- batches


def clip(i, speaker, duration_s=100.0) -> Clip:
    return Clip(f"c{i}", "x", "de", speaker, "train", f"datasets/raw/x/c{i}.wav", 0.0, duration_s, "CC0", "a")


def many_clips():
    return [clip(i, f"s{i % 7}") for i in range(60)]


def test_speaker_stays_in_one_batch():
    seen = {}
    for i, b in enumerate(make_batches(many_clips(), 0.1)):
        for c in b:
            assert seen.setdefault(c.speaker, i) == i
    assert len(set(seen.values())) > 1


def test_batches_are_in_rank_order_and_do_not_depend_on_input_order():
    clips = many_clips()
    batches = list(make_batches(clips, 0.2))
    assert [[c.clip_id for c in b] for b in batches] == [
        [c.clip_id for c in b] for b in make_batches(reversed(clips), 0.2)
    ]
    first_ranks = [min(rank(c.clip_id) for c in b) for b in batches]
    assert first_ranks == sorted(first_ranks)
    for b in batches:  # within a speaker, rank order
        for speaker in {x.speaker for x in b}:
            mine = [rank(x.clip_id) for x in b if x.speaker == speaker]
            assert mine == sorted(mine)


def test_a_speaker_over_the_target_is_kept_whole_and_the_batch_is_a_target():
    clips = [clip(i, "big", 3600.0) for i in range(5)] + [clip(10 + i, f"s{i}", 600.0) for i in range(6)]
    batches = list(make_batches(clips, 1.0))
    assert sum(len(b) for b in batches) == 11
    big = next(b for b in batches if any(c.speaker == "big" for c in b))
    assert len([c for c in big if c.speaker == "big"]) == 5
    assert all(sum(c.duration_s for c in b) <= 3600 for b in batches if b is not big)


def test_clips_without_a_speaker_are_units_of_their_own():
    clips = [clip(i, None, 1800.0) for i in range(6)]
    assert [len(b) for b in make_batches(clips, 1.0)] == [2, 2, 2]


# ---------------------------------------------------------------------------------------------------- the cap walk


def ranked_hours(hours: list[float]) -> list[Clip]:
    """One speaker's clips, the i-th in rank order lasting hours[i]."""
    order = align._by_rank(clip(i, "s") for i in range(len(hours)))
    return [dataclasses.replace(c, duration_s=h * 3600) for c, h in zip(order, hours, strict=True)]


def aligner(calls, dropping=()):
    def align_batch(chunk):
        calls.append([c.duration_s / 3600 for c in chunk])
        return [
            Timing(c.clip_id, "word", [(100, 500)], [], {"duration_ms": 1000}, "")
            for c in chunk
            if c.clip_id not in dropping
        ]

    return align_batch


class MarkedRules:
    def __init__(self, marked=()):
        self.marked = set(marked)

    def apply(self, timing):
        return "marked" if timing.clip_id in self.marked else None


def test_cap_aligns_only_until_30_h():
    calls: list[list[float]] = []
    out = list(walk_cap(ranked_hours([1.0] * 50), aligner(calls), MarkedRules(), hours=30.0))
    assert [sum(c) for c in calls] == [30.0]
    assert len(out) == 30


def test_the_first_clip_that_would_exceed_the_cap_ends_the_walk_even_if_a_smaller_one_follows():
    calls: list[list[float]] = []
    clips = ranked_hours([20, 20, 5, 5])
    assert len(list(walk_cap(clips, aligner(calls), MarkedRules(), hours=30.0))) == 1
    assert calls == [[20]]


def test_drops_are_refilled_from_rank():
    clips = ranked_hours([1.0] * 50)
    calls: list[list[float]] = []
    dropped: list = []
    marked = {c.clip_id for c in clips[:5]}
    out = list(walk_cap(clips, aligner(calls), MarkedRules(marked), hours=30.0, dropped=dropped))
    assert [t.clip_id for t in out] == [c.clip_id for c in clips[5:35]]
    assert [sum(c) for c in calls] == [30.0, 5.0]
    assert [(t.clip_id, reason) for t, reason in dropped] == [(c.clip_id, "marked") for c in clips[:5]]


def test_clips_the_aligner_could_not_align_are_refilled_too():
    clips = ranked_hours([1.0] * 50)
    calls: list[list[float]] = []
    out = list(walk_cap(clips, aligner(calls, dropping={c.clip_id for c in clips[:5]}), MarkedRules(), hours=30.0))
    assert len(out) == 30
    assert [sum(c) for c in calls] == [30.0, 5.0]


def test_a_speaker_under_the_cap_is_aligned_in_one_call():
    calls: list[list[float]] = []
    assert len(list(walk_cap(ranked_hours([1.0] * 5), aligner(calls), MarkedRules(), hours=30.0))) == 5
    assert len(calls) == 1


# ---------------------------------------------------------------------------------------------------- the stage


@pytest.fixture
def stage(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(align, "BATCH_HOURS", 1.0 / 3600)  # one speaker per batch
    monkeypatch.setattr(align, "MIN_CALIBRATION_CLIPS", 1)
    fake = install(monkeypatch)
    cfg = dataclasses.replace(SLICE, name="align-test", sources=("fake",), languages=None)
    monkeypatch.setitem(index.SOURCES, "fake", SimpleNamespace(MFA_ALIGNED=True, clips=lambda raw, languages: iter(())))
    return SimpleNamespace(root=root, cfg=cfg, tmp=tmp_path / "tmp", **vars(fake))


def write_index(stage, clips: list[Clip], reserved: Collection[tuple[str, str, str]] = ()) -> None:
    out = work_dir(stage.cfg) / "index"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(write_parquet({out / "fake.parquet": index._frame(clips, set(reserved))}))


def tables(stage) -> dict:
    """Everything the stage wrote that a later stage reads, as plain Python."""
    base = work_dir(stage.cfg)
    words = {
        p.relative_to(base).as_posix(): asyncio.run(read_parquet(p)).to_dicts()
        for p in sorted((base / "words").glob("*/*/*.parquet"))
    }
    dropped = asyncio.run(read_parquet(base / "dropped.parquet")).to_dicts()
    return {"words": words, "dropped": dropped}


def reasons(stage) -> dict[str, str]:
    return {r["clip_id"]: r["reason"] for r in tables(stage)["dropped"]}


def batch_calls(stage) -> list[list[str]]:
    """The MFA runs after the calibration run, which comes first."""
    return stage.fake.calls[1:]


def speakers_clips(stage, n=4, per=2, **kw) -> list[Clip]:
    return [make_clip(stage.root, f"s{s}/c{i}", f"s{s}", **kw) for s in range(n) for i in range(per)]


def test_run_align_writes_one_table_per_batch_and_drops_none_when_all_is_well(stage):
    clips = speakers_clips(stage)
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    t = tables(stage)
    assert len(t["words"]) == 4
    assert len(batch_calls(stage)) == 4
    assert len(stage.fake.calls[0]) == len(clips)  # the calibration run, before any label batch
    rows = [r for rows in t["words"].values() for r in rows]
    assert sorted(r["clip_id"] for r in rows) == sorted(c.clip_id for c in clips)
    assert t["dropped"] == []
    first = rows[0]
    assert first["kind"] == "word"
    assert first["mask_ms"] == []
    assert first["speech_ms"][0]["start"] == 200
    prov = json.loads(first["provenance"])
    assert prov["aligner"] == "mfa"
    assert prov["version"] == "3.4.3"
    assert prov["commit"] == "a" * 40
    assert prov["models"]["acoustic"]["release"] == "3.0.0"
    assert json.loads(first["metrics"])["duration_ms"] == 1500
    assert stage.fetched == ["de"]


def test_finished_batch_is_skipped(stage):
    write_index(stage, speakers_clips(stage))
    run_align(stage.cfg, ["de"])
    first = tables(stage)
    run_align(stage.cfg, ["de"])
    assert len(stage.fake.calls) == 1 + 4 == 1 + len(first["words"])  # the calibration is not repeated
    assert tables(stage) == first


def test_a_clip_with_an_offset_gets_times_relative_to_the_clip_not_the_file(stage):
    # The file has speech at 1.5 to 2.5 s; the clip starts at 1.0 s, so its speech starts at 0.5 s.
    c = make_clip(stage.root, "a/c0", "a", "aa bb", burst=(1.5, 2.5), total=4.0, offset=1.0, duration=2.0)
    write_index(stage, [c])
    run_align(stage.cfg, ["de"])
    (row,) = next(iter(tables(stage)["words"].values()))
    starts = [w["start"] for w in row["speech_ms"]]
    assert 490 <= starts[0] <= 510, f"times are relative to the file, not the clip: {starts}"
    assert json.loads(row["metrics"])["duration_ms"] == 2000


def test_a_clip_that_does_not_decode_is_dropped_before_mfa_and_the_batch_goes_on(stage):
    clips = [make_clip(stage.root, "a/bad", "a", corrupt=True, total=1.5), make_clip(stage.root, "a/ok", "a")]
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    assert reasons(stage) == {"a/bad": "decode_failed"}
    assert stage.fake.calls == [["aa bb cc"]] * 2  # the calibration run and the batch; the corrupt file is in neither


def test_a_clip_that_decodes_to_the_wrong_length_is_dropped(stage):
    short = make_clip(stage.root, "a/short", "a", total=1.5, duration=1.3)
    past_end = make_clip(stage.root, "a/past_end", "a", total=1.5, duration=2.0)
    write_index(stage, [short, past_end, make_clip(stage.root, "a/ok", "a")])
    run_align(stage.cfg, ["de"])
    assert reasons(stage) == {"a/past_end": "decode_failed"}


def test_a_clip_without_a_usable_transcript_is_dropped_with_its_reason(stage):
    write_index(stage, [make_clip(stage.root, "a/c0", "a", "12 ..."), make_clip(stage.root, "a/c1", "a")])
    run_align(stage.cfg, ["de"])
    assert reasons(stage) == {"a/c0": "no_transcript"}


def test_exit_0_without_a_textgrid_or_with_a_blank_row_is_no_alignment(stage):
    clips = [make_clip(stage.root, f"a/c{i}", "a", text) for i, text in enumerate(["aa FAIL", "aa BLANK", "aa bb"])]
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    assert reasons(stage) == {"a/c0": "no_alignment", "a/c1": "no_alignment"}
    kept = next(iter(tables(stage)["words"].values()))
    assert [r["clip_id"] for r in kept] == ["a/c2"]


def test_every_clip_ends_as_one_row_or_one_drop(stage):
    texts = ["aa bb", "FAIL", "BLANK", "12", "aa RETRY", "aa bb LOW"]
    clips = [make_clip(stage.root, f"a/c{i}", "a", t) for i, t in enumerate(texts)] + [
        make_clip(stage.root, "a/bad", "a", corrupt=True)
    ]
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    t = tables(stage)
    kept = [r["clip_id"] for rows in t["words"].values() for r in rows]
    dropped = [r["clip_id"] for r in t["dropped"]]
    assert sorted(kept + dropped) == sorted(c.clip_id for c in clips)
    assert len(set(kept + dropped)) == len(clips)
    assert reasons(stage)["a/c4"] == "needs_wider_beam"
    assert [c for c, r in reasons(stage).items() if r == "needs_wider_beam"] == [
        "a/c4"
    ]  # no mark leaks to another clip
    assert reasons(stage)["a/c5"] == "low_loglik"
    assert json.loads(next(r for r in t["dropped"] if r["clip_id"] == "a/c4")["metrics"])["retried"] is True


def test_exit_0_without_an_analysis_file_makes_every_clip_no_alignment_and_still_reconciles(stage):
    clips = speakers_clips(stage, n=1)
    stage.fake.no_csv = True
    models = align.Models(Path("a.zip"), Path("d.dict"), None, {})
    timings, drops = align._align_round(clips, "de", stage.tools, models, "p")
    assert timings == []
    assert sorted(drops) == sorted(align.Drop(c.clip_id, "no_alignment", None) for c in clips)


def test_a_textgrid_for_a_file_it_was_not_given_fails_the_batch(stage):
    write_index(stage, speakers_clips(stage, n=1))
    stage.fake.extra_textgrid = True
    with pytest.raises(MfaFailedError, match="files it was not given"):
        run_align(stage.cfg, ["de"])
    assert not list((work_dir(stage.cfg) / "words").glob("*/*/*.parquet"))


def test_a_failing_mfa_leaves_the_batch_unfinished_and_no_temp_copy(stage):
    write_index(stage, speakers_clips(stage))
    stage.fake.kill_at = 3  # the calibration and two batches ran
    with pytest.raises(Killed):
        run_align(stage.cfg, ["de"])
    assert not list(stage.tmp.iterdir())
    assert len(list((work_dir(stage.cfg) / "words/fake/de").glob("*.parquet"))) == 2


def test_clips_without_a_transcript_never_reach_mfa_and_are_counted(stage):
    spoken = speakers_clips(stage, n=2)
    noise = [dataclasses.replace(make_clip(stage.root, "noise/n0", None), transcript=None)]
    write_index(stage, spoken + noise)
    run_align(stage.cfg, ["de"])
    t = tables(stage)
    assert {r["clip_id"] for rows in t["words"].values() for r in rows} == {c.clip_id for c in spoken}
    assert t["dropped"] == []
    assert all("noise/n0" not in call for calls in stage.fake.calls for call in calls)
    assert sum(len(call) for call in stage.fake.calls) == 2 * len(spoken)  # the calibration run and the batches
    summary = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())["per_source"]["fake"]["de"]
    assert (summary["index_clips"], summary["kept"], summary["no_transcript_not_aligned"]) == (5, 4, 1)
    assert summary["over_cap_not_aligned"] == 0


def test_reserved_speakers_are_never_aligned(stage):
    clips = speakers_clips(stage, n=3)
    write_index(stage, clips, reserved={("train", "de", "s1")})
    clips = [dataclasses.replace(c, reserved=c.speaker == "s1") for c in clips]
    run_align(stage.cfg, ["de"])
    done = {r["clip_id"] for rows in tables(stage)["words"].values() for r in rows}
    assert done == {c.clip_id for c in clips if c.speaker != "s1"}
    assert sum(len(call) for call in batch_calls(stage)) == 4


def test_languages_are_chosen_and_each_gets_its_own_models_and_tables(stage):
    clips = [
        *speakers_clips(stage, n=2, language="de"),
        make_clip(stage.root, "f/c0", "f", language="fr"),
        make_clip(stage.root, "f/c1", "f", language="fr"),
    ]
    write_index(stage, clips)
    run_align(stage.cfg, ["fr"])
    assert stage.fetched == ["fr"]
    assert set(tables(stage)["words"]) == {"words/fake/fr/00000.parquet"}
    run_align(stage.cfg, None)
    assert stage.fetched == ["fr", "de", "fr"]
    assert len(stage.fake.calls) == 2 + 3  # fr is finished, de is calibrated and runs two batches
    assert {k.split("/")[2] for k in tables(stage)["words"]} == {"de", "fr"}


def test_an_unknown_or_open_language_fails_before_any_batch(stage):
    write_index(stage, [make_clip(stage.root, "a/c0", "a", language="cs")])
    with pytest.raises(ValueError, match="no MFA model is chosen"):
        run_align(stage.cfg, None)
    with pytest.raises(ValueError, match="no row for language 'xx'"):
        run_align(stage.cfg, ["xx"])
    assert stage.fake.calls == []
    assert stage.fetched == []


def test_a_source_labelled_without_mfa_is_left_alone(stage, monkeypatch):
    monkeypatch.setitem(index.SOURCES, "fake", SimpleNamespace(MFA_ALIGNED=False, clips=None, timings=None))
    run_align(stage.cfg, None)
    assert stage.fake.calls == []


@pytest.mark.parametrize("failing", [0, 5])
def test_the_cap_aligns_a_speaker_only_while_room_is_left_and_refills_drops(stage, monkeypatch, failing):
    monkeypatch.setattr(align, "SPEAKER_CAP_HOURS", 4.5 / 3600)  # 4.5 s: three 1.5 s clips
    plain = [make_clip(stage.root, f"a/c{i}", "a", f"w{'x' * i}") for i in range(6)]
    order = align._by_rank(plain)
    clips = [dataclasses.replace(c, transcript="FAIL") if c is order[failing] else c for c in plain]
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    kept = sorted(r["clip_id"] for rows in tables(stage)["words"].values() for r in rows)
    texts = {c.clip_id: c.transcript for c in clips}
    if failing == 0:  # the first clip fails, so a fourth is aligned to fill its place
        assert [len(call) for call in batch_calls(stage)] == [3, 1]
        assert kept == sorted(c.clip_id for c in order[1:4])
        assert reasons(stage) == {order[0].clip_id: "no_alignment"}
    else:  # the failing clip is past the cap, so it is never aligned or even looked at
        assert [len(call) for call in batch_calls(stage)] == [3]
        assert kept == sorted(c.clip_id for c in order[:3])
        assert texts[order[5].clip_id] == "FAIL"
        assert reasons(stage) == {}
    summary = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())["per_source"]["fake"]["de"]
    assert summary["kept"] == 3
    assert summary["index_clips"] == 6
    assert summary["kept"] + sum(summary["dropped"].values()) + summary["over_cap_not_aligned"] == 6


def test_clips_without_a_speaker_are_aligned_once_each_over_several_batches_each_as_its_own_speaker(stage):
    clips = [make_clip(stage.root, f"n/c{i}", None) for i in range(5)]
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    t = tables(stage)
    kept = [r["clip_id"] for rows in t["words"].values() for r in rows]
    assert sorted(kept) == sorted(c.clip_id for c in clips)
    assert len(t["words"]) == 5
    assert [len(call) for call in batch_calls(stage)] == [1] * 5
    manifest = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())
    assert manifest["per_source"]["fake"]["de"]["over_cap_not_aligned"] == 0


def test_every_clip_id_is_in_exactly_one_file_when_speakers_and_loose_clips_mix(stage):
    clips = [
        *speakers_clips(stage, n=3),
        *(make_clip(stage.root, f"n/c{i}", None, "FAIL" if i == 2 else "aa") for i in range(4)),
    ]
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    t = tables(stage)
    seen = [r["clip_id"] for rows in t["words"].values() for r in rows] + [r["clip_id"] for r in t["dropped"]]
    assert sorted(seen) == sorted(c.clip_id for c in clips)
    manifest = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())
    assert manifest["per_source"]["fake"]["de"]["over_cap_not_aligned"] == 0


def test_loose_clips_in_one_run_do_not_share_an_mfa_speaker(stage, monkeypatch):
    monkeypatch.setattr(align, "BATCH_HOURS", 100.0 / 3600)
    folders = []

    def spy(tools, models, corpus, out, mfa_root):
        folders.append(sorted(p.name for p in corpus.iterdir()))
        stage.fake(tools, models, corpus, out, mfa_root)

    monkeypatch.setattr(align, "_run_mfa", spy)
    write_index(
        stage, [*(make_clip(stage.root, f"n/c{i}", None) for i in range(3)), make_clip(stage.root, "a/c0", "a")]
    )
    run_align(stage.cfg, ["de"])
    assert folders == [["s0", "s1", "s2", "s3"]] * 2  # the calibration run and the batch


def test_clips_without_a_speaker_are_not_capped(stage, monkeypatch):
    monkeypatch.setattr(align, "SPEAKER_CAP_HOURS", 1.5 / 3600)
    monkeypatch.setattr(align, "BATCH_HOURS", 100 / 3600)
    write_index(stage, [make_clip(stage.root, f"n/c{i}", None) for i in range(5)])
    run_align(stage.cfg, ["de"])
    assert sum(len(rows) for rows in tables(stage)["words"].values()) == 5


def test_the_batch_composition_is_checked_on_a_rerun(stage, monkeypatch):
    write_index(stage, speakers_clips(stage))
    run_align(stage.cfg, ["de"])
    monkeypatch.setattr(align, "BATCH_HOURS", 100 / 3600)
    with pytest.raises(StateMismatchError, match="delete it to start over"):
        run_align(stage.cfg, ["de"])


def test_a_language_with_too_few_calibration_clips_is_not_aligned_and_says_so(stage, monkeypatch):
    monkeypatch.setattr(align, "MIN_CALIBRATION_CLIPS", 50)
    write_index(stage, speakers_clips(stage, n=5, per=2))
    run_align(stage.cfg, ["de"])
    words = work_dir(stage.cfg) / "words/fake/de"
    assert len(stage.fake.calls) == 1
    assert not list(words.glob("*.parquet"))
    assert not (words / "rules.json").exists()
    summary = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())["per_source"]["fake"]["de"]
    cal = summary["calibration"]
    assert cal["outcome"] == "too_few_to_calibrate"
    assert (cal["clips"], cal["aligned"], cal["min_aligned"]) == (10, 10, 50)
    assert (summary["kept"], summary["not_aligned_too_few_to_calibrate"], summary["over_cap_not_aligned"]) == (0, 10, 0)
    run_align(stage.cfg, ["de"])
    assert len(stage.fake.calls) == 1  # the outcome is reproduced, not recomputed


def test_clips_that_fail_to_align_do_not_count_toward_the_calibration_minimum(stage, monkeypatch):
    monkeypatch.setattr(align, "MIN_CALIBRATION_CLIPS", 3)
    write_index(stage, [make_clip(stage.root, f"s{i}/c0", f"s{i}", "FAIL" if i < 2 else "aa") for i in range(4)])
    run_align(stage.cfg, ["de"])
    assert not (work_dir(stage.cfg) / "words/fake/de/rules.json").exists()
    summary = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())["per_source"]["fake"]["de"]
    assert (summary["calibration"]["clips"], summary["calibration"]["aligned"]) == (4, 2)


def test_calibration_clips_that_the_fixed_rules_drop_do_not_move_the_floor_or_count_toward_the_minimum(
    stage, monkeypatch
):
    speakers = [make_clip(stage.root, f"s{i}/c0", f"s{i}", "aa") for i in range(6)]
    retried = [make_clip(stage.root, f"r{i}/c0", f"r{i}", "aa RETRY LOWER") for i in range(3)]  # -95, needs the beam

    def floor_of(clips: list[Clip]) -> dict:
        shutil.rmtree(work_dir(stage.cfg), ignore_errors=True)
        write_index(stage, clips)
        run_align(stage.cfg, ["de"])
        return json.loads((work_dir(stage.cfg) / "words/fake/de/rules.json").read_text())

    plain, mixed = floor_of(speakers), floor_of(speakers + retried)
    assert plain["loglik_floor"] == mixed["loglik_floor"] == -50.0
    cal = mixed["calibration"]
    assert (cal["aligned"], cal["passing_fixed_rules"], cal["fixed_rule_drops"]) == (9, 6, {"needs_wider_beam": 3})
    assert reasons(stage) == {c.clip_id: "needs_wider_beam" for c in retried}
    monkeypatch.setattr(align, "MIN_CALIBRATION_CLIPS", 7)  # 9 aligned, but only 6 pass
    shutil.rmtree(work_dir(stage.cfg))
    write_index(stage, speakers + retried)
    run_align(stage.cfg, ["de"])
    outcome = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())["per_source"]["fake"]["de"]
    assert outcome["calibration"]["outcome"] == "too_few_to_calibrate"
    assert (outcome["calibration"]["aligned"], outcome["calibration"]["passing_fixed_rules"]) == (9, 6)


def test_a_kill_during_the_calibration_leaves_nothing_behind(stage):
    write_index(stage, speakers_clips(stage))
    stage.fake.kill_at = 0
    with pytest.raises(Killed):
        run_align(stage.cfg, ["de"])
    words = work_dir(stage.cfg) / "words/fake/de"
    assert not (words / "rules.json").exists()
    assert not (words / "calibration.json").exists()
    stage.fake.kill_at = None
    run_align(stage.cfg, ["de"])
    assert len(tables(stage)["words"]) == 4


def prolific_speaker_corpus(stage, n_big=12, others=6, per=2) -> list[Clip]:
    """One speaker owns the lowest ranks, which makes its whole-speaker batch the first one."""
    ids = sorted((f"c{i}" for i in range(n_big + others * per)), key=rank)
    return [make_clip(stage.root, i, "big" if k < n_big else f"o{(k - n_big) % others}") for k, i in enumerate(ids)]


def test_the_calibration_batch_takes_many_speakers_even_when_one_owns_the_lowest_ranks(stage):
    clips = prolific_speaker_corpus(stage)
    first_label_batch = next(iter(make_batches(clips, 100.0 / 3600)))
    assert {c.speaker for c in first_label_batch} == {c.speaker for c in clips}  # one big batch
    chosen = align.calibration_clips(clips)
    per_speaker = Counter(c.speaker for c in chosen)
    assert per_speaker["big"] == align.CALIBRATION_PER_SPEAKER == 10
    assert len(per_speaker) == 7
    assert [rank(c.clip_id) for c in chosen] == sorted(rank(c.clip_id) for c in chosen)


def test_the_rules_do_not_change_when_clips_of_the_prolific_speaker_outside_the_calibration_change(stage):
    clips = prolific_speaker_corpus(stage)
    in_calibration = {c.clip_id for c in align.calibration_clips(clips)}
    outside = {c.clip_id for c in clips if c.speaker == "big" and c.clip_id not in in_calibration}
    inside = {c.clip_id for c in clips if c.speaker == "big" and c.clip_id in in_calibration}
    assert outside

    def rules_with(changed: set[str]) -> dict:
        shutil.rmtree(work_dir(stage.cfg), ignore_errors=True)
        write_index(
            stage, [dataclasses.replace(c, transcript="aa LOWER") if c.clip_id in changed else c for c in clips]
        )
        run_align(stage.cfg, ["de"])
        return json.loads((work_dir(stage.cfg) / "words/fake/de/rules.json").read_text())

    base = rules_with(set())
    assert rules_with(outside) == base
    assert rules_with(inside)["loglik_floor"] < base["loglik_floor"]  # the same test can see a change
    assert base["calibration"]["speakers"] == 7


def test_the_calibration_stops_at_the_clip_limit_and_a_clip_without_a_speaker_is_its_own_speaker(stage, monkeypatch):
    loose = [make_clip(stage.root, f"n/c{i}", None) for i in range(30)]
    assert len(align.calibration_clips(loose)) == 30  # more than the per-speaker limit, as each is a speaker
    monkeypatch.setattr(align, "CALIBRATION_MAX_CLIPS", 12)
    chosen = align.calibration_clips(loose)
    assert [c.clip_id for c in chosen] == [c.clip_id for c in align._by_rank(loose)][:12]


def test_the_manifest_records_the_calibration_and_the_fitted_rules(stage):
    clips = speakers_clips(stage)
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    summary = json.loads((work_dir(stage.cfg) / "words/manifest.json").read_text())["per_source"]["fake"]["de"]
    cal = summary["calibration"]
    assert cal["outcome"] == "fitted"
    assert (cal["clips"], cal["speakers"], cal["aligned"]) == (8, 4, 8)
    assert (cal["passing_fixed_rules"], cal["fixed_rule_drops"]) == (8, {})
    assert cal["rank_cut"] == max(rank(c.clip_id) for c in clips)
    assert cal["digest"] == align._digest(c.clip_id for c in align._by_rank(clips))
    assert summary["rules"]["mad_scale"] == 1.4826
    assert "calibration" not in summary["rules"]


def test_dropped_parquet_is_the_union_of_the_per_batch_tables(stage):
    write_index(
        stage, [make_clip(stage.root, f"s{s}/c{i}", f"s{s}", t) for s in range(3) for i, t in enumerate(["aa", "FAIL"])]
    )
    run_align(stage.cfg, ["de"])
    per_batch = sorted((work_dir(stage.cfg) / "words/fake/de/dropped").glob("*.parquet"))
    assert len(per_batch) == 3
    union = asyncio.run(read_parquet(work_dir(stage.cfg) / "dropped.parquet"))
    assert union.height == 3
    assert set(union["reason"]) == {"no_alignment"}
    assert union.schema == pl.Schema(align.DROPPED_SCHEMA)
    assert sorted(union["batch"]) == ["00000", "00001", "00002"]


# ---------------------------------------------------------------------------------------------------- resuming


def reference(stage, clips: list[Clip]) -> dict:
    """The tables of one uninterrupted run, after which the stage's work is deleted for the run under test."""
    write_index(stage, clips)
    run_align(stage.cfg, ["de"])
    result = tables(stage)
    shutil.rmtree(work_dir(stage.cfg))
    stage.fake.calls.clear()
    return result


def varied_clips(stage) -> list[Clip]:
    """Eight speakers with a clip each (one batch each), of every kind; the first batch is an ordinary clip."""
    texts = ["aa bb", "aa bb", "FAIL", "aa bb LOW", "aa", "aa bb cc LOW", "aa bb", "BLANK"]
    clips = [make_clip(stage.root, f"s{i}/c0", f"s{i}", t) for i, t in enumerate(texts)]
    first = next(iter(make_batches(clips, align.BATCH_HOURS)))[0]
    return [dataclasses.replace(c, transcript="aa bb") if c.clip_id == first.clip_id else c for c in clips]


def test_the_rules_are_fitted_on_the_calibration_batch_and_saved_before_the_first_label_batch(stage, monkeypatch):
    monkeypatch.setattr(align, "CALIBRATION_MAX_CLIPS", 2)
    clips = varied_clips(stage)
    calibration = {c.clip_id for c in align.calibration_clips(clips)}
    # Everything outside the calibration batch is much worse than anything inside it.
    clips = [dataclasses.replace(c, transcript="aa" if c.clip_id in calibration else "aa LOWER") for c in clips]
    write_index(stage, clips)
    stage.fake.kill_at = 1  # the first label batch
    with pytest.raises(Killed):
        run_align(stage.cfg, ["de"])
    rules_file = work_dir(stage.cfg) / "words/fake/de/rules.json"
    saved = rules_file.read_text()
    rules = json.loads(saved)
    assert rules["loglik_floor"] == -50.0
    assert rules["calibration"]["clips"] == 2
    stage.fake.kill_at = None
    run_align(stage.cfg, ["de"])
    assert rules_file.read_text() == saved
    assert len(stage.fake.calls) == 1 + 8  # not calibrated again
    assert set(reasons(stage).values()) == {"low_loglik"}
    assert set(reasons(stage)) == {c.clip_id for c in clips} - calibration


def test_a_kill_between_batches_resumes_to_the_uninterrupted_result(stage):
    clips = varied_clips(stage)
    expected = reference(stage, clips)
    write_index(stage, clips)
    stage.fake.kill_at = 4  # the calibration and three batches ran
    with pytest.raises(Killed):
        run_align(stage.cfg, ["de"])
    assert len(list((work_dir(stage.cfg) / "words/fake/de").glob("*.parquet"))) == 3
    # The drops of the finished batches survive the kill: `dropped.parquet` is built from them in a `finally`.
    assert len(reasons(stage)) == sum(
        len(asyncio.run(read_parquet(p))) for p in (work_dir(stage.cfg) / "words/fake/de/dropped").glob("*.parquet")
    )
    stage.fake.kill_at, calls_before = None, len(stage.fake.calls)
    run_align(stage.cfg, ["de"])
    assert len(stage.fake.calls) - calls_before == 5  # only the unfinished batches
    assert tables(stage) == expected


def test_a_kill_in_the_middle_of_a_batch_leaves_no_partial_table_and_resumes_the_same(stage, monkeypatch):
    clips = varied_clips(stage)
    expected = reference(stage, clips)
    write_index(stage, clips)
    real_write, writes = align.write_parquet, []

    async def dying(frames):
        writes.append(list(frames))
        if len(writes) == 3:  # the third batch: its drops are replaced, then the process dies before its words
            await real_write({next(iter(frames)): frames[next(iter(frames))]})
            raise Killed
        await real_write(frames)

    monkeypatch.setattr(align, "write_parquet", dying)
    with pytest.raises(Killed):
        run_align(stage.cfg, ["de"])
    words = work_dir(stage.cfg) / "words/fake/de"
    assert len(list(words.glob("*.parquet"))) == 2
    assert (words / "dropped/00002.parquet").exists()
    assert not list(work_dir(stage.cfg).rglob("*.tmp"))
    monkeypatch.setattr(align, "write_parquet", real_write)
    run_align(stage.cfg, ["de"])
    assert tables(stage) == expected


def test_dropped_parquet_holds_only_the_drops_of_finished_batches(stage):
    base = work_dir(stage.cfg)
    for batch, finished in (("00000", True), ("00001", False)):
        out = base / "words/fake/de"
        (out / "dropped").mkdir(parents=True, exist_ok=True)
        drops = align._drop_table("fake", "de", batch, [align.Drop(f"c{batch}", "no_alignment", None)])
        frames = {out / "dropped" / f"{batch}.parquet": drops}
        if finished:
            frames[out / f"{batch}.parquet"] = align._table([])
        asyncio.run(write_parquet(frames))
    assert align._build_dropped(base)["clip_id"].to_list() == ["c00000"]
    assert asyncio.run(read_parquet(base / "dropped.parquet"))["batch"].to_list() == ["00000"]


def test_the_temp_copy_is_removed_after_a_run_and_a_failure(stage):
    write_index(stage, speakers_clips(stage, n=2))
    run_align(stage.cfg, ["de"])
    assert not list(stage.tmp.iterdir())
    stage.fake.extra_textgrid = True
    shutil.rmtree(work_dir(stage.cfg) / "words")
    with pytest.raises(MfaFailedError):
        run_align(stage.cfg, ["de"])
    assert not list(stage.tmp.iterdir())


def test_the_audio_and_transcripts_stay_on_local_temp(stage, monkeypatch):
    seen = []

    def spy(tools, models, corpus, out, mfa_root):
        seen.append(corpus)
        stage.fake(tools, models, corpus, out, mfa_root)

    monkeypatch.setattr(align, "_run_mfa", spy)
    write_index(stage, speakers_clips(stage, n=1))
    run_align(stage.cfg, ["de"])
    assert seen
    assert all(Path(tempfile.gettempdir()) in c.parents for c in seen)
    assert not list(stage.root.rglob("*.lab"))
    assert not list(stage.root.rglob("*.wav.tmp"))


# ---------------------------------------------------------------------------------------------------- running mfa


def test_mfa_runs_as_just_mfa_does_with_its_own_root(monkeypatch, tmp_path):
    seen = {}

    def run(cmd, env, **kw):
        seen.update(cmd=cmd, env=env, kw=kw)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(align.subprocess, "run", run)
    monkeypatch.setenv("VIRTUAL_ENV", "/root/venv")
    tools = align.Tools(Path("/repo"), "3.4.3", "a" * 40)
    align._mfa(tools, tmp_path / "root", "version")
    assert seen["cmd"] == ["uv", "run", "--project", "/repo/tools/mfa", "--no-sync", "mfa", "version"]
    assert seen["env"]["PATH"].startswith("/repo/tools/kaldi/dist/bin:")
    assert seen["env"]["MFA_ROOT_DIR"] == str(tmp_path / "root")
    assert "VIRTUAL_ENV" not in seen["env"]


def test_a_non_zero_exit_fails_loudly_with_the_end_of_the_output(monkeypatch):
    out = "\n".join(f"line {i}" for i in range(100))
    monkeypatch.setattr(align.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=out))
    with pytest.raises(MfaFailedError, match=r"(?s)exited 1:.*line 99"):
        align._mfa(align.Tools(Path("/repo"), "3.4.3", "a" * 40), Path("/x"), "align", "corpus")


def test_align_gets_the_cleaning_flag_the_models_and_the_g2p_only_where_there_is_one(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(align, "_mfa", lambda tools, root, *args: calls.append((root, args)))
    tools = align.Tools(Path("/repo"), "3.4.3", "a" * 40)
    with_g2p = align.Models(Path("am.zip"), Path("d.dict"), Path("g.zip"), {})
    align._run_mfa(tools, with_g2p, Path("corpus"), Path("out"), Path("root"))
    align._run_mfa(tools, with_g2p._replace(g2p=None), Path("corpus"), Path("out"), Path("root"))
    assert calls[0][1] == ("align", "--clean", "-v", "corpus", "d.dict", "am.zip", "out", "--g2p_model_path", "g.zip")
    assert "--g2p_model_path" not in calls[1][1]
    assert "--use_g2p" not in calls[0][1]


def test_models_are_downloaded_by_version_once_and_their_versions_are_read_from_them(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    downloads = []

    def fake_mfa(tools, root, *args):
        assert args[:2] == ("model", "download")
        assert args[-2] == "--version"
        kind, name, release = args[2], args[3], args[-1]
        downloads.append((kind, name, release))
        target = root / "pretrained_models" / kind
        target.mkdir(parents=True)
        if kind == "dictionary":
            (target / f"{name}.dict").write_text("a\t1\ta\n")
        else:
            with zipfile.ZipFile(target / f"{name}.zip", "w") as z:
                z.writestr(f"{name}/meta.json", json.dumps({"version": f"read-{kind}"}))

    monkeypatch.setattr(align, "_mfa", fake_mfa)
    tools = align.Tools(Path("/repo"), "3.4.3", "a" * 40)
    spec = load_models()["de"]
    models = align.fetch_models(tools, spec)
    assert downloads == [
        ("acoustic", "german_mfa", "3.0.0"),
        ("dictionary", "german_mfa", "3.0.0"),
        ("g2p", "german_mfa", "3.0.0"),
    ]
    assert models.dictionary.name == "german_mfa.dict"
    assert models.acoustic.parent.parts[-2:] == ("german_mfa", "3.0.0")
    assert models.info["acoustic"]["version"] == "read-acoustic"
    assert models.info["acoustic"]["release"] == "3.0.0"
    assert models.info["dictionary"]["version"] is None
    assert len(models.info["dictionary"]["sha256"]) == 64
    assert not list((tmp_path / "models/mfa").glob(".download-*"))
    align.fetch_models(tools, spec)
    assert len(downloads) == 3
    assert "g2p" not in align.fetch_models(tools, load_models()["it"]).info
