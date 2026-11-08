# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from as_is_support import LIBRIVAD_SAMPLES, make_librivad_tree, save_samplewise_labels
from openvad_train.dataset.config import SLICE
from openvad_train.dataset.index import guard_path
from openvad_train.dataset.labels import rasterise
from openvad_train.dataset.plan import PLAN_SCHEMA, clip_rows
from openvad_train.dataset.publish import _as_is_with_labels
from openvad_train.dataset.sources import librivad


@pytest.fixture
def librivad_tree(tmp_path):
    return make_librivad_tree(tmp_path / "datasets/raw/librivad")


def test_librivad_maps_each_split(librivad_tree):
    clips = list(librivad.clips(librivad_tree))
    assert {c.split for c in clips if "train-clean-100" in c.path} == {"train"}
    assert {c.split for c in clips if "dev-clean" in c.path} == {"validation"}
    assert {c.split for c in clips if "test-clean" in c.path} == {"test"}
    assert any("Concat" in c.path for c in clips)
    assert any("Concat" not in c.path for c in clips)


def test_a_pair_whose_label_is_not_as_long_as_its_wav_is_dropped_and_counted_by_split(librivad_tree):
    report: dict = {}
    clips = list(librivad.clips(librivad_tree, None, report))
    assert report["label_length_mismatch"] == {"test": 4, "train": 4, "validation": 4}
    assert not any("1089-134686-0001_+_1188-133604-0016" in c.clip_id for c in clips)
    # A pair of two chapters whose label has the wav's length stays, and so does a pair of one chapter.
    assert any("1188-133604-0016_+_1089-134686-0001" in c.clip_id for c in clips)
    assert any("1089-134686-0000_+_0001" in c.clip_id for c in clips)


def test_the_length_is_the_wav_header_not_the_label(librivad_tree):
    c = next(c for c in librivad.clips(librivad_tree) if "1089-134686-0000_+_0001" in c.clip_id)
    assert c.duration_s == 56000 / 16000
    wav = librivad_tree / "generated/LibriSpeechConcat/train-clean-100/Babble_noise/5/1089-134686-0000_+_0001.wav"
    sf.write(wav, np.zeros(30000, np.float32), 16000, subtype="PCM_16")  # now shorter than its label
    report: dict = {}
    ids = {c.clip_id for c in librivad.clips(librivad_tree, None, report)}
    assert "LibriSpeechConcat/train-clean-100/Babble_noise/5/1089-134686-0000_+_0001" not in ids
    assert "LibriSpeechConcat/train-clean-100/Babble_noise/10/1089-134686-0000_+_0001" in ids
    assert report["label_length_mismatch"]["train"] == 4 + 1  # the short-label pair and this wav


def test_every_mixture_is_a_clip_and_passes_the_split_guard(librivad_tree):
    clips = list(librivad.clips(librivad_tree))
    # 3 LibriSpeech and 2 concatenated utterances, in 2 noises at 2 SNRs, in 3 splits.
    assert len(clips) == 5 * 4 * 3 == len({c.clip_id for c in clips})
    for c in clips:
        guard_path(c.path, c.split)


def test_validation_is_read_from_the_sibling_val_tree_and_absent_without_it(tmp_path):
    raw = make_librivad_tree(tmp_path / "datasets/raw/librivad", with_val=False)
    clips = list(librivad.clips(raw))
    assert {c.split for c in clips} == {"train", "test"}
    full = make_librivad_tree(tmp_path / "other/datasets/raw/librivad")
    val = [c for c in librivad.clips(full) if c.split == "validation"]
    assert val
    assert all(c.path.startswith("datasets/raw/librivad-val/generated/") for c in val)


def test_a_slice_and_the_full_build_give_every_clip_the_same_split(librivad_tree):
    full = {c.clip_id: c.split for c in librivad.clips(librivad_tree)}
    assert {c.clip_id: c.split for c in librivad.clips(librivad_tree, {"en"})} == full
    assert list(librivad.clips(librivad_tree, {"de"})) == []


def test_clip_fields_and_a_length_from_the_label(librivad_tree):
    c = next(
        c
        for c in librivad.clips(librivad_tree)
        if c.clip_id.endswith("Babble_noise/5/1089-134686-0001") and "test" in c.path
    )
    assert (c.source, c.language, c.speaker, c.licence, c.offset_s, c.transcript) == (
        "librivad",
        "en",
        "1089",
        "CC BY-NC-SA 4.0",
        0.0,
        None,
    )
    assert c.duration_s == LIBRIVAD_SAMPLES["1089-134686-0001"] / 16000
    assert (
        sf.info(librivad_tree / "generated/LibriSpeech/test-clean/Babble_noise/5/1089-134686-0001.wav").frames
        == (LIBRIVAD_SAMPLES["1089-134686-0001"])
    )


def test_a_concatenated_pair_of_two_chapters_finds_its_label_in_the_folder_of_the_source_tree(librivad_tree):
    # `match_labels.py` upstream would look in 1188/133604; the label is in 1089/134686.
    pair = [c for c in librivad.clips(librivad_tree) if "1188-133604-0016_+_1089-134686-0001" in c.clip_id]
    assert len(pair) == 4 * 3
    assert {c.speaker for c in pair} == {"1188"}
    (t, *_) = librivad.timings(librivad_tree, pair[:1])
    assert t.speech_ms == [(300, 700)]


def test_a_mixture_without_a_label_is_refused_by_name(librivad_tree):
    (librivad_tree / "repo/Files/Labels/LibriSpeech/test-clean/1089/134686/1089-134686-0001.npy").unlink()
    with pytest.raises(FileNotFoundError, match=r"LibriSpeech/test-clean/1089/134686/1089-134686-0001.npy"):
        list(librivad.clips(librivad_tree))


def test_librivad_labels_by_majority(tmp_path):
    # 10 ms is 160 samples; a tie is non-speech.
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [1] * 100 + [0] * 60)).tolist() == [1]
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [1] * 80 + [0] * 80)).tolist() == [0]


def test_the_frame_count_is_the_one_rasterise_gives_and_a_short_last_frame_counts_missing_samples_as_silence(tmp_path):
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [1] * 100)).tolist() == [1]
    # 1.5 frames: rounds to 2, and the half frame (80 of 160 samples) is a tie, so non-speech.
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [1] * 240)).tolist() == [1, 0]
    # 1.4 frames: the 64 samples past the first frame are not counted.
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [0] * 160 + [1] * 64)).tolist() == [0]


def test_a_label_file_that_is_not_zeros_and_ones_is_refused(tmp_path):
    with pytest.raises(ValueError, match="0 or 1"):
        librivad.frame_labels(save_samplewise_labels(tmp_path, [0, 1, 2] * 100))


def test_timings_are_frame_labels_as_intervals_that_rasterise_back_unchanged(librivad_tree):
    clips = [c for c in librivad.clips(librivad_tree) if "1089-134686-0001" in c.clip_id]
    timings = list(librivad.timings(librivad_tree, clips))
    assert {t.kind for t in timings} == {"frame"}
    assert {t.provenance for t in timings} == {"librivad:labels"}
    # Samples 4800 to 11200 are speech: frames 30 to 70, that is 300 to 700 ms.
    assert {tuple(t.speech_ms) for t in timings} == {((300, 700),)}
    label = librivad_tree / "repo/Files/Labels/LibriSpeech/train-clean-100/1089/134686/1089-134686-0001.npy"
    n = LIBRIVAD_SAMPLES["1089-134686-0001"]
    assert rasterise(timings[0].speech_ms, [], n).tolist() == librivad.frame_labels(label).tolist()


def test_published_labels_are_the_frame_labels_without_the_gap_rule(librivad_tree):
    # Two speech runs a frame apart: the 200 ms gap rule would join them, frame labels must stay as they are.
    label = np.zeros(32000, np.int16)
    label[1600:3200] = 1
    label[3360:4800] = 1
    folder = librivad_tree / "repo/Files/Labels/LibriSpeech/train-clean-100/1089/134686"
    save_samplewise_labels(folder, label.tolist(), "1089-134686-0000.npy")
    clip = next(
        c
        for c in librivad.clips(librivad_tree)
        if c.clip_id == "LibriSpeech/train-clean-100/Babble_noise/5/1089-134686-0000"
    )
    (timing,) = librivad.timings(librivad_tree, [clip])
    (row,) = clip_rows(clip, timing, SLICE)
    published = _as_is_with_labels(pl.DataFrame([row], schema=PLAN_SCHEMA))["labels"][0].to_list()
    assert published == librivad.frame_labels(folder / "1089-134686-0000.npy").tolist()
    assert published[10:20] == [1] * 10
    assert published[20] == 0
    assert published[21:30] == [1] * 9


def test_librivad_has_no_mfa_and_no_cutting():
    assert (librivad.MFA_ALIGNED, librivad.CUTTING) == (False, None)
