# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import dataclasses
import json
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from align_support import make_clip
from openvad_train.dataset import align, handcheck, index
from openvad_train.dataset.align import parse_textgrid
from openvad_train.dataset.codecs import decode_16k
from openvad_train.dataset.config import SLICE, work_dir
from openvad_train.dataset.handcheck import (
    QualityBar,
    QualityBarUnsetError,
    apply_failure_rule,
    pair_boundaries,
    sample_handcheck,
    score_handcheck,
)
from openvad_train.dataset.labels import apply_gap_rule
from openvad_train.dataset.parquet_io import write_parquet
from openvad_train.dataset.schema import Timing
from openvad_train.fetch.common import data_dir


WORDS = [(200, 500), (600, 900), (1300, 1500)]  # the second gap is 400 ms, the first 100 ms: G = 200 merges the first
TOTAL_S = 1.6


@pytest.fixture
def sampled(monkeypatch, tmp_path):
    """A config with 40 aligned German and 30 aligned English clips whose log-likelihoods run from -1 up."""
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path / "data"))
    cfg = dataclasses.replace(SLICE, name="handcheck-test", sources=("fake",), languages=None)
    monkeypatch.setitem(index.SOURCES, "fake", SimpleNamespace(MFA_ALIGNED=True, clips=lambda raw, languages: iter(())))
    root = tmp_path / "data"
    clips, timings = [], {}
    for language, n in (("de", 40), ("en", 30)):
        for i in range(n):
            clip = make_clip(root, f"fake/{language}{i:02d}", f"s{i}", language=language, total=TOTAL_S)
            clips.append(clip)
            timings.setdefault(language, []).append(
                Timing(clip.clip_id, "word", WORDS, [], {"loglik": -1.0 - i}, "provenance")
            )
    base = work_dir(cfg)
    (base / "index").mkdir(parents=True)
    asyncio.run(write_parquet({base / "index" / "fake.parquet": index._frame(clips, set())}))
    for language, items in timings.items():
        words_dir = base / "words" / "fake" / language
        words_dir.mkdir(parents=True)
        asyncio.run(write_parquet({words_dir / "00000.parquet": align._table(items)}))
    return cfg


def test_the_sample_has_the_asked_number_per_language_across_the_scores(sampled):
    out = sample_handcheck(sampled, languages=("de", "en"), per_language=10)
    key = json.loads((out / "key.json").read_text())
    for language, n in (("de", 40), ("en", 30)):
        scores = [f["score"] for f in key["files"].values() if f["language"] == language]
        assert len(scores) == 10
        # strata of the sorted scores: the lowest and the highest decile are both drawn from
        assert min(scores) < -1.0 - 0.1 * n
        assert max(scores) > -1.0 - 0.9 * n
        assert (max(scores) - min(scores)) / n > 0.7


def test_the_labeller_gets_wavs_and_blank_textgrids_with_neutral_names_and_no_score(sampled):
    out = sample_handcheck(sampled, languages=("de", "en"), per_language=5)
    files = sorted((out / "clips").iterdir())
    assert len(files) == 20
    assert all(re.fullmatch(r"h\d{3}\.(wav|TextGrid)", f.name) for f in files)
    key = json.loads((out / "key.json").read_text())
    for name, f in key["files"].items():
        audio, rate = sf.read(out / "clips" / f"{name}.wav")
        assert rate == 16000
        assert len(audio) == f["n_samples"] == round(TOTAL_S * 16000)
        text = (out / "clips" / f"{name}.TextGrid").read_text()
        tiers = parse_textgrid(text)
        assert list(tiers) == ["speech"]
        assert [t for _, _, t in tiers["speech"] if t] == []  # no interval is labelled
        assert str(f["score"]) not in text
        assert f["clip_id"] not in text
    assert not list(out.glob("clips/*.json"))


def test_the_sample_is_a_function_of_the_tables_alone(sampled):
    first = (sample_handcheck(sampled, languages=("de",), per_language=7) / "key.json").read_text()
    again = dataclasses.replace(sampled, name="handcheck-test-2")
    shutil.copytree(work_dir(sampled), work_dir(again), ignore=shutil.ignore_patterns("handcheck"))
    assert (sample_handcheck(again, languages=("de",), per_language=7) / "key.json").read_text() == first


def test_an_existing_sample_is_never_overwritten(sampled):
    out = sample_handcheck(sampled, languages=("de",), per_language=3)
    label = out / "clips" / "h001.TextGrid"
    label.write_text("the owner's work")
    with pytest.raises(FileExistsError, match="may hold labels"):
        sample_handcheck(sampled, languages=("de",), per_language=3)
    assert label.read_text() == "the owner's work"


def test_the_wav_is_the_16_bit_quantisation_of_the_decoded_clip(sampled):
    out = sample_handcheck(sampled, languages=("de",), per_language=3)
    clips = {c.clip_id: c for c in align._read_index(work_dir(sampled), "fake")}
    for name, f in json.loads((out / "key.json").read_text())["files"].items():
        clip = clips[f["clip_id"]]
        decoded = decode_16k(str(data_dir() / clip.path), clip.offset_s, clip.duration_s)
        expected = np.clip(np.rint(decoded * 32768), -32768, 32767).astype(np.int16)
        written, rate = sf.read(out / "clips" / f"{name}.wav", dtype="int16")
        assert rate == 16000
        assert np.array_equal(written, expected)


def test_duplicate_clips_do_not_count_towards_the_size(sampled):
    base = work_dir(sampled) / "words" / "fake" / "de"
    shutil.copy(base / "00000.parquet", base / "00001.parquet")  # the same 40 clips again
    with pytest.raises(ValueError, match=r"de \(40\)"):
        sample_handcheck(sampled, languages=("de",), per_language=41)


def test_a_language_with_too_few_clips_is_an_error_naming_it(sampled):
    with pytest.raises(ValueError, match=r"en \(30\), ca \(0\)"):
        sample_handcheck(sampled, languages=("de", "en", "ca"), per_language=31)
    assert not handcheck.handcheck_dir(sampled).exists()


# ---------------------------------------------------------------------------------------------------- scoring


def label(
    out, intervals_by_file: dict[str, list[tuple[tuple[float, float], str]]] | None = None, *, shift_s: float = 0.0
) -> None:
    """Write the owner's TextGrids: by default the MFA words after the gap rule, `shift_s` later."""
    key = json.loads((out / "key.json").read_text())
    for name, f in key["files"].items():
        end = f["n_samples"] / 16000
        speech = [(a / 1000 + shift_s, b / 1000 + shift_s, "speech") for a, b in apply_gap_rule(WORDS, key["gap_ms"])]
        if intervals_by_file and name in intervals_by_file:
            speech = [(a, b, t) for (a, b), t in intervals_by_file[name]]
        cells, at = [], 0.0
        for a, b, t in speech:
            if a > at:
                cells.append((at, a, ""))
            cells.append((a, b, t))
            at = b
        if at < end:
            cells.append((at, end, ""))
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            f"xmin = 0\nxmax = {end}\ntiers? <exists>\nsize = 1\nitem []:",
            '    item [1]:\n        class = "IntervalTier"\n        name = "speech"',
            f"        xmin = 0\n        xmax = {end}\n        intervals: size = {len(cells)}",
        ]
        for i, (a, b, t) in enumerate(cells, 1):
            lines.append(
                f'        intervals [{i}]:\n            xmin = {a}\n            xmax = {b}\n            text = "{t}"'
            )
        (out / "clips" / f"{name}.TextGrid").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def out(sampled):
    return sample_handcheck(sampled, languages=("de", "en"), per_language=4)


def test_exact_agreement_is_perfect(out):
    label(out)
    report = score_handcheck(out)
    assert report["frame_agreement"] == 1.0
    for kind in ("onsets", "offsets"):
        assert report[kind] == {
            "paired": 16,  # two runs in each of 8 clips: the 100 ms gap is merged at G = 200, the 400 ms one is not
            "within_20_ms": 1.0,
            "within_50_ms": 1.0,
            "unpaired_owner": 0,
            "unpaired_mfa": 0,
        }
    assert set(report["per_language"]) == {"de", "en"}
    assert report["per_language"]["de"]["frame_agreement"] == 1.0
    assert json.loads((out / "report.json").read_text()) == report


def test_the_owner_gets_the_same_gap_rule_so_a_pause_under_g_costs_nothing(out):
    # The owner marks the words and a 150 ms pause between the first two (G = 200 ms): training merges them.
    key = json.loads((out / "key.json").read_text())
    paused = {name: [((0.2, 0.5), "speech"), ((0.65, 0.9), "speech"), ((1.3, 1.5), "speech")] for name in key["files"]}
    label(out, paused)
    report = score_handcheck(out)
    assert report["frame_agreement"] == 1.0
    for kind in ("onsets", "offsets"):
        assert report[kind]["paired"] == 16
        assert report[kind]["within_20_ms"] == 1.0
        assert (report[kind]["unpaired_owner"], report[kind]["unpaired_mfa"]) == (0, 0)


def test_a_pause_over_g_stays_a_gap_for_the_owner_as_for_training(out):
    key = json.loads((out / "key.json").read_text())
    split = {name: [((0.2, 0.5), "speech"), ((0.75, 0.9), "speech"), ((1.3, 1.5), "speech")] for name in key["files"]}
    label(out, split)  # the owner's 250 ms pause is over G and stays open; training bridged its 100 ms one
    report = score_handcheck(out)
    assert report["onsets"]["unpaired_owner"] == 8  # one extra run per clip
    assert report["frame_agreement"] == pytest.approx(1 - 25 / 160)  # 500 to 750 ms


def test_boundaries_are_scored_within_20_and_50_ms(out):
    label(out, shift_s=0.03)
    report = score_handcheck(out)
    assert report["onsets"]["within_20_ms"] == 0.0
    assert report["onsets"]["within_50_ms"] == 1.0
    assert report["frame_agreement"] < 1.0


def test_any_label_but_a_blank_one_is_speech(out):
    label(out, {"h001": [((0.2, 0.9), " "), ((1.3, 1.5), "noise")]})
    report = score_handcheck(out)
    # the whitespace-only interval is no speech, so h001 has one run where MFA has two, and "noise" is speech
    assert report["onsets"]["unpaired_owner"] == 0
    assert report["onsets"]["unpaired_mfa"] == 1
    assert report["onsets"]["paired"] == 15


def test_masked_frames_are_left_out_of_agreement(out):
    key = json.loads((out / "key.json").read_text())
    key["files"]["h001"]["mask_ms"] = [[0, 1000]]
    (out / "key.json").write_text(json.dumps(key))
    label(out, {"h001": [((1.3, 1.5), "speech")]})  # the owner leaves out the speech under the mask
    report = score_handcheck(out)
    assert report["frames"] == 7 * 160 + 60  # h001 keeps 60 unmasked frames
    assert report["frame_agreement"] == 1.0


def test_a_clip_with_no_speech_marked_is_not_labelled_and_leaves_every_total(out):
    label(out, {"h001": []})
    report = score_handcheck(out)
    assert report["empty_textgrids"] == 1
    assert report["clips"] == 7
    assert report["frames"] == 7 * 160
    assert report["onsets"]["paired"] == 14
    languages = {lang: r["empty_textgrids"] for lang, r in report["per_language"].items()}
    assert sorted(languages.values()) == [0, 1]


def test_pairing_is_one_to_one_by_nearest_distance():
    assert pair_boundaries([100, 105], [103]) == [(1, 0)]  # 105 is 2 away from 103, 100 is 3 away
    assert pair_boundaries([100], [90, 110, 400]) == [(0, 0)]  # a tie goes to the lower index
    assert pair_boundaries([0, 50], [40, 1000]) == [(1, 0), (0, 1)]  # nearest first, then what is left
    assert pair_boundaries([], [1, 2]) == []


# ---------------------------------------------------------------------------------------------------- the bar


def test_an_unset_bar_refuses_to_judge_any_language():
    with pytest.raises(QualityBarUnsetError, match="no thresholds"):
        QualityBar().passes(0.0, 0.0)
    with pytest.raises(QualityBarUnsetError):
        QualityBar(max_failure_rate=0.1).passes(0.0, 0.0)


def test_a_set_bar_is_inclusive_on_both_thresholds():
    bar = QualityBar(max_failure_rate=0.05, max_drop_share=0.3)
    assert bar.passes(0.05, 0.3)
    assert not bar.passes(0.06, 0.1)
    assert not bar.passes(0.01, 0.31)


def test_failure_rule():
    assert apply_failure_rule("xx", 100.0, passed=False) == "drop"
    assert apply_failure_rule("ca", 2636.0, passed=False) == "owner"
    assert apply_failure_rule("de", 1199.0, passed=True) == "keep"


def test_exactly_500_hours_is_dropped_and_the_owner_decides_above_it():
    assert apply_failure_rule("xx", 500.0, passed=False) == "drop"
    assert apply_failure_rule("xx", 500.01, passed=False) == "owner"
    assert apply_failure_rule("xx", 0.0, passed=True) == "keep"
