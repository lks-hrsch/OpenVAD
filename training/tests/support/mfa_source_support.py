# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Synthetic trees shaped like the raw layouts of the MFA-labelled sources (`data/README.md`), for the adapter tests."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import soundfile as sf


CV_HEADER = ["client_id", "path", "sentence_id", "sentence", "up_votes", "down_votes", "locale"]
CV_RELEASE = "cv-corpus-27.0-2026-09-11"


def write_tsv(path: Path, rows: list[list], header: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def add_cv_locale(raw: Path, locale: str, clips: dict[str, tuple[str, str]], *, dev=(), test=(), train=()) -> Path:
    """`clips` maps a file name to (speaker, sentence) and all of them are validated; `dev`, `test` and `train` list
    file names of the official splits. Durations are 1.5 s per clip, 1.25 s for a name ending in `short.mp3`.
    """
    folder = raw / locale / CV_RELEASE / locale
    row = lambda name: [clips[name][0], name, f"id-{name}", clips[name][1], "2", "0", locale]  # noqa: E731
    write_tsv(folder / "validated.tsv", [row(n) for n in clips], CV_HEADER)
    for split, names in (("dev", dev), ("test", test), ("train", train)):
        write_tsv(folder / f"{split}.tsv", [row(n) for n in names], CV_HEADER)
    write_tsv(
        folder / "clip_durations.tsv",
        [[n, 1250 if n.endswith("short.mp3") else 1500] for n in clips],
        ["clip", "duration[ms]"],
    )
    return folder


def make_cv_tree(raw: Path) -> Path:
    """German: speakers a and b are plain, c is in dev, d in test, e has a clip in no official split. `c2` is a
    validated clip of a dev speaker. Finished locales have a `.<locale>-extracted` marker or are in the raw
    manifest (`zh-HK`); `ca` has TSVs and neither, as a locale does while its fetch runs; `en` is not fetched yet.
    """
    clips = {
        "common_voice_de_a1.mp3": ("a", "One sentence."),
        "common_voice_de_a2.mp3": ("a", "Another sentence."),
        "common_voice_de_b1.mp3": ("b", "Hello world."),
        "common_voice_de_c1.mp3": ("c", "Good morning."),
        "common_voice_de_c2.mp3": ("c", "Good evening."),
        "common_voice_de_d1.mp3": ("d", "Good night."),
        "validated_only_clip.mp3": ("e", "Only validated."),
        "common_voice_de_short.mp3": ("a", "Short."),
    }
    add_cv_locale(
        raw,
        "de",
        clips,
        dev=["common_voice_de_c1.mp3"],
        test=["common_voice_de_d1.mp3"],
        train=["common_voice_de_a1.mp3", "common_voice_de_a2.mp3", "common_voice_de_b1.mp3"],
    )
    for locale, speaker in (("zh-CN", "m"), ("zh-TW", "t"), ("kab", "k"), ("zh-HK", "h"), ("ca", "q")):
        name = f"common_voice_{locale}_1.mp3"
        add_cv_locale(raw, locale, {name: (speaker, "text")}, train=[name])
    for locale in ("de", "zh-CN", "zh-TW", "kab"):
        (raw / f".{locale}-extracted").write_text("0")
    (raw / "manifest.json").write_text(json.dumps({"locales": {"zh-HK": {}}}))
    return raw


def make_mls_tree(raw: Path) -> Path:
    """German with one clip per split and a second speaker in train; `mls_english` is not a language we use."""
    segments = {
        "train": [("10_100_000000", 12.34, 1.0), ("11_101_000003", 3.5, 3.75)],
        "dev": [("20_200_000000", 10.0, 20.5)],
        "test": [("30_300_000007", 100.0, 111.25)],
    }
    for split, items in segments.items():
        folder = raw / "mls_german" / split
        rows = [[i, f"http://example.org/{i}.mp3", start, start + length] for i, start, length in items]
        write_tsv(folder / "segments.txt", rows)
        write_tsv(folder / "transcripts.txt", [[i, f"text of {i}"] for i, _, _ in items])
    write_tsv(raw / "mls_english" / "train" / "transcripts.txt", [["1_2_3", "ignored"]])
    return raw


def make_fleurs_tree(raw: Path) -> Path:
    """`de_de`, `cmn_hans_cn` (Mandarin), and two directories without an MFA model."""
    for directory in ("de_de", "cmn_hans_cn", "yue_hant_hk", "hu_hu"):
        for split in ("train", "dev", "test"):
            name = f"{directory}-{split}.wav"
            rows = [[1, name, f"Raw {split}.", f"normalised {split}", "n o r m", 48000, "MALE"]]
            write_tsv(raw / directory / f"{split}.tsv", rows)
    return raw


VP_HEADER = [
    "id_", "paragraph_id", "session_id", "speaker_id", "original_text", "normed_text", "decoded",
    "start_time", "end_time", "cer", "wer", "vad", "split", "gender",
]  # fmt: skip


def add_vp_table(raw: Path, language: str, rows: list[tuple]) -> None:
    """`rows` are (id, speaker or "None", start, end, official split) and optionally the session file."""
    path = raw / "annotations" / f"asr_{language}.tsv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(VP_HEADER)
        for clip_id, speaker, start, end, split, *session in rows:
            session = session[0] if session else "20140224-0900-PLENARY-3"
            writer.writerow(
                [
                    clip_id,
                    "p",
                    session,
                    speaker,
                    "Original 6.",
                    "normed six",
                    "dec",
                    start,
                    end,
                    0,
                    0,
                    "[[0, 1]]",
                    split,
                    "na",
                ]
            )


def make_vp_tree(raw: Path) -> Path:
    """German with 30 speakers of 1 to 30 minutes in the official `train`, one `invalid` row, three rows without a
    speaker (one `dev`), and a table for Finnish, which has no MFA model.
    """
    rows = [(f"de_s{s}_{i}", str(s), 100.0 * i, 100.0 * i + 60.0, "train") for s in range(1, 31) for i in range(s)]
    rows += [("invalid_row", "1", 0.0, 5.0, "invalid")]
    # The same paragraph in a part session file and, hours later, in the whole-day file.
    rows += [
        ("de_dup", "1", 50.0, 55.0, "train", "20140224-0900-PLENARY-3"),
        ("de_dup", "1", 46850.0, 46855.0, "train", "20140224-0900-PLENARY-3_whole"),
    ]
    rows += [(f"de_none_{i}", "None", 10.0 * i, 10.0 * i + 5.0, "dev" if i == 0 else "train") for i in range(3)]
    add_vp_table(raw, "de", rows)
    add_vp_table(raw, "fi", [("fi_1", "1", 0.0, 5.0, "train")])
    return raw


def write_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(round(16000 * seconds)), 16000, subtype="PCM_16")


def make_sc_tree(raw: Path) -> Path:
    """Two words, three speakers, one clip each in validation and test, noise, and the separate test set."""
    root = raw / "speech_commands_v0.02"
    names = {
        "yes/aaaaaaaa_nohash_0.wav": 1.0,
        "yes/bbbbbbbb_nohash_0.wav": 0.5,
        "no/aaaaaaaa_nohash_1.wav": 1.0,
        "no/cccccccc_nohash_0.wav": 0.75,
        "no/dddddddd_nohash_0.wav": 1.0,
        "_background_noise_/running_tap.wav": 6.0,
    }
    for name, seconds in names.items():
        write_wav(root / name, seconds)
    (root / "validation_list.txt").write_text("yes/bbbbbbbb_nohash_0.wav\n")
    (root / "testing_list.txt").write_text("no/cccccccc_nohash_0.wav\nno/dddddddd_nohash_0.wav\n")
    write_wav(raw / "speech_commands_test_set_v0.02" / "_silence_" / "0.wav", 1.0)
    write_wav(raw / "speech_commands_test_set_v0.02" / "yes" / "x.wav", 1.0)
    return raw
