# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Literal

import pytest

from openvad_train.dataset.index import ForbiddenSource, apply_cap_rank, guard_path, reserve_speakers
from openvad_train.dataset.schema import Clip, rank


Split = Literal["train", "validation", "test"]


def clip(id="c", speaker: str | None = "s", language="en", split: Split = "train", source="x", duration_s=1.0) -> Clip:
    return Clip(id, source, language, speaker, split, f"datasets/raw/{source}/{id}.flac", 0.0, duration_s, "CC0", None)


@pytest.mark.parametrize(
    "path",
    [
        "raw/ava-speech/a.flac",
        "raw/simsamu/a.m4a",
        "raw/fearless-steps/a.wav",
        "raw/librispeech/LibriSpeech/test-clean/a.flac",
        "raw/librispeech/LibriSpeech/train-clean-100/a.flac",
        "raw/picovoice/a.wav",
    ],
)
def test_guard_rejects_benchmark_paths(path):
    with pytest.raises(ForbiddenSource):
        guard_path(path, "test")


def test_guard_keeps_librivad_evaluation_out_of_train():
    guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "test")
    guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "validation")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "train")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "train")


def test_guard_keeps_each_librivad_evaluation_in_its_own_split():
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "validation")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "test")


def test_guard_rejects_evaluation_folders_outside_librivad():
    with pytest.raises(ForbiddenSource):
        guard_path("raw/voices/LibriSpeech/test-clean/a.wav", "test")


def test_guard_accepts_ordinary_training_paths():
    guard_path("datasets/raw/swc/german/A/audio.ogg", "train")
    guard_path("datasets/raw/librivad/generated/LibriSpeech/train-clean-100/a.wav", "train")


def test_reserve_at_least_one_per_language():
    clips = [clip(speaker=f"s{i}", language="kk") for i in range(6)]
    assert len(reserve_speakers(clips)) == 1


def test_reserve_five_percent_per_split_and_language():
    clips = [clip(speaker=f"s{i}", language="en") for i in range(100)]
    clips += [clip(speaker=f"t{i}", language="de") for i in range(40)]
    clips += [clip(speaker=f"v{i}", split="test") for i in range(3)]
    reserved = reserve_speakers(clips)
    assert len([r for r in reserved if r[:2] == ("train", "en")]) == 5
    assert len([r for r in reserved if r[:2] == ("train", "de")]) == 2
    assert len([r for r in reserved if r[:2] == ("test", "en")]) == 1


def test_reserve_ignores_clips_without_a_speaker():
    assert reserve_speakers([clip(speaker=None) for _ in range(5)]) == set()


def test_reserve_is_stable_across_clip_order():
    clips = [clip(id=f"c{i}", speaker=f"s{i}") for i in range(60)]
    assert reserve_speakers(clips) == reserve_speakers(reversed(clips))


def test_cap_keeps_rank_prefix_under_30_h():
    clips = [clip(id=f"c{i}", speaker="s", duration_s=3600.0) for i in range(40)]
    kept = list(apply_cap_rank(clips))
    assert sum(c.duration_s for c in kept) <= 30 * 3600
    assert {c.clip_id for c in kept} == {c.clip_id for c in sorted(clips, key=lambda c: rank(c.clip_id))[:30]}


def test_cap_stops_at_the_first_clip_that_would_exceed_it():
    clips = [clip(id=f"c{i}", duration_s=d) for i, d in enumerate([2.0, 2.0, 0.5, 0.5, 0.5])]
    by_rank = sorted(clips, key=lambda c: rank(c.clip_id))
    kept = list(apply_cap_rank(clips, hours=3.0 / 3600))
    expected = []
    total = 0.0
    for c in by_rank:
        total += c.duration_s
        if total > 3.0:
            break
        expected.append(c.clip_id)
    assert {c.clip_id for c in kept} == set(expected)


def test_cap_counts_each_speaker_and_source_on_its_own():
    clips = [clip(id=f"a{i}", speaker="s", source="x", duration_s=3600.0) for i in range(3)]
    clips += [clip(id=f"b{i}", speaker="s", source="y", duration_s=3600.0) for i in range(3)]
    clips += [clip(id=f"c{i}", speaker="t", source="x", duration_s=3600.0) for i in range(3)]
    assert len(list(apply_cap_rank(clips, hours=2.0))) == 6


def test_cap_leaves_clips_without_a_speaker_alone():
    clips = [clip(id=f"c{i}", speaker=None, duration_s=3600.0) for i in range(40)]
    assert list(apply_cap_rank(clips)) == clips
