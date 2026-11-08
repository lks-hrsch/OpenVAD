# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Synthetic trees shaped like the raw layouts of the sources used as they are (`data/README.md`): AMI, CallHome,
VOiCES and LibriVAD. Audio is seeded noise, so a pointer that reads the wrong samples is caught.
"""

import csv
import io
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, round(16000 * seconds)).astype(np.float32)


def write_noise(path: Path, seconds: float, seed: int = 0) -> np.ndarray:
    """A 16 kHz mono 16-bit WAV of noise; returns the samples as the file holds them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, noise(seconds, seed), 16000, subtype="PCM_16")
    return sf.read(path, dtype="float32")[0]


# ---------------------------------------------------------------------------------------------------- AMI

# (start, end, kind) in seconds of one participant's words file; a word whose end is before its start is dropped.
AMI_WORDS = [
    (1.00, 1.50, "w"),
    (1.50, 1.50, "punc"),
    (2.00, 3.00, "vocalsound"),
    (3.50, 3.50, "disfmarker"),
    (4.00, 4.50, "w"),
    (5.00, 4.00, "w"),
]
LAUGHTER = (2000, 3000)


def _ami_words_xml(meeting: str, agent: str, shift: float) -> str:
    elements = []
    for i, (start, end, kind) in enumerate(AMI_WORDS):
        attrs = f'nite:id="{meeting}.{agent}.words{i}" starttime="{start + shift:.2f}" endtime="{end + shift:.2f}"'
        if kind == "w":
            elements.append(f"<w {attrs}>word</w>")
        elif kind == "punc":
            elements.append(f'<w {attrs} punc="true">,</w>')
        else:
            elements.append(f'<{kind} {attrs} type="laugh"/>' if kind == "vocalsound" else f"<{kind} {attrs}/>")
    elements.append("<w>untimed</w>")  # a few words of the corpus have no times
    return f'<nite:root xmlns:nite="http://nite.sourceforge.net/">{"".join(elements)}</nite:root>'


def make_ami_tree(raw: Path) -> Path:
    """Three meetings of the full-corpus-ASR partition: ES2002a (train), ES2011a (dev) and ES2004a (test), with two
    participants each. ES2002a has the headset of its second participant missing, as the fetch's `missing` lists.
    """
    meetings = {"ES2002a": ("MIO001", "FEE002"), "ES2011a": ("MIO003", "FEE004"), "ES2004a": ("MIO005", "FEE006")}
    xml = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<nite:root xmlns:nite="http://nite.sourceforge.net/">']
    for meeting, names in meetings.items():
        xml.append(f'<meeting observation="{meeting}">')
        for channel, (name, agent) in enumerate(zip(names, "AB", strict=True)):
            xml.append(f'<speaker channel="{channel}" nxt_agent="{agent}" global_name="{name}"/>')
            words = raw / "annotations/words" / f"{meeting}.{agent}.words.xml"
            words.parent.mkdir(parents=True, exist_ok=True)
            words.write_text(_ami_words_xml(meeting, agent, shift=10.0 * channel))  # the second wearer speaks later
        xml.append("</meeting>")
        for channel in (0, 1):
            if (meeting, channel) != ("ES2002a", 1):
                write_noise(raw / "audio" / meeting / f"{meeting}.Headset-{channel}.wav", 6.0, seed=channel)
    xml.append("</nite:root>")
    path = raw / "annotations/corpusResources/meetings.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(xml))
    return raw


# ---------------------------------------------------------------------------------------------------- CallHome

CALL_SECONDS = 2.0


def wav_bytes(seconds: float, seed: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, noise(seconds, seed), 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def make_callhome_tree(raw: Path, per_language: dict[str, list[int]] | None = None) -> Path:
    """Shards of the mirror's shape: `<lang>/data-<n>-of-<total>.parquet`, a row per call with its audio embedded
    and its turns. `per_language` maps a language to the number of calls in each of its shards (default: 4, 3 and 2
    calls in three English shards, 3 German calls and 2 Mandarin calls, so 14 calls). Every call is 2 s of noise
    seeded by its position, and has two turns, one per side, which overlap.
    """
    per_language = per_language or {"eng": [4, 3, 2], "deu": [3], "zho": [2]}
    seed = 0
    for lang, shards in per_language.items():
        for n, calls in enumerate(shards):
            rows = []
            for _ in range(calls):
                seed += 1
                rows.append(
                    {
                        "audio": {"bytes": wav_bytes(CALL_SECONDS, seed), "path": None},
                        "timestamps_start": [0.0, 0.5],
                        "timestamps_end": [1.0, CALL_SECONDS],
                        "speakers": ["A", "B"],
                    }
                )
            path = raw / lang / f"data-{n:05d}-of-{len(shards):05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows).write_parquet(path)
    return raw


# ---------------------------------------------------------------------------------------------------- VOiCES

DEV_CLEAN_SPEAKERS = {"0111", "0112"}
TRAIN_100_SPEAKERS = {"0083", "0084", "0085"}
TRAIN_360_SPEAKERS = {f"{n:04d}" for n in range(200, 230)}
VOICES_DELAY_SAMPLES = 160  # 10 ms
VOICES_SECONDS = 3.0
_INDEX_HEADER = ["index", "filename", "speaker", "noisy_length", "noisy_sr", "source"]
_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 3.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0.0
        xmax = 3.0
        intervals: size = 3
        intervals [1]:
            xmin = 0.0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 1.5
            text = "word"
        intervals [3]:
            xmin = 1.5
            xmax = 3.0
            text = ""
"""
# Words of the alignment above, before the delay.
VOICES_WORDS_MS = (500, 1500)


def make_voices_tree(raw: Path) -> Path:
    """A devkit of 35 speakers (2 from dev-clean, 3 from train-clean-100, 30 from train-clean-360), one utterance
    each, re-recorded once in the devkit's `train` folder and, for the first train-clean-360 speaker, also in its
    `test` folder; plus one distractor-only recording. A re-recording is its source delayed by 160 samples. The
    alignments are in the sibling `librispeech-alignments`, which is where `raw` lives (`raw.parent`).
    """
    root = raw / "VOiCES_devkit"
    datasets = dict.fromkeys(DEV_CLEAN_SPEAKERS, "dev-clean")
    datasets |= dict.fromkeys(TRAIN_100_SPEAKERS, "train-clean-100") | dict.fromkeys(
        TRAIN_360_SPEAKERS, "train-clean-360"
    )
    refs = root / "references"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "Lab41-SRI-VOiCES-speaker-gender-dataset.tbl").write_text(
        "Speaker Gender DataSet\n" + "".join(f"{s} F {d}\n" for s, d in sorted(datasets.items()))
    )
    rows: dict[str, list[list]] = {"train": [], "test": []}
    for i, (speaker, dataset) in enumerate(sorted(datasets.items())):
        chapter, segment = 9000 + i, 1
        devkit_splits = ("train", "test") if speaker == min(TRAIN_360_SPEAKERS) else ("train",)
        for devkit_split in devkit_splits:
            utterance = f"sp{speaker}-ch{chapter:06d}-sg{segment:04d}"
            src = f"source-16k/{devkit_split}/sp{speaker}/Lab41-SRI-VOiCES-src-{utterance}.wav"
            source = write_noise(root / src, VOICES_SECONDS, seed=i)
            mic = 1 if devkit_split == "train" else 5
            name = f"Lab41-SRI-VOiCES-rm1-musi-{utterance}-mc0{mic}-stu-clo-dg060"
            relative = f"distant-16k/speech/{devkit_split}/rm1/musi/sp{speaker}/{name}.wav"
            delayed = np.concatenate([np.zeros(VOICES_DELAY_SAMPLES, np.float32), source])[: len(source)]
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            sf.write(root / relative, delayed, 16000, subtype="PCM_16")
            rows[devkit_split].append([len(rows[devkit_split]), relative, int(speaker), len(source), 16000, src])
        grid = raw.parent / "librispeech-alignments" / dataset / str(int(speaker)) / str(chapter)
        grid.mkdir(parents=True, exist_ok=True)
        (grid / f"{int(speaker)}-{chapter}-{segment:04d}.TextGrid").write_text(_TEXTGRID)
    for devkit_split, split_rows in rows.items():
        with (refs / f"{devkit_split}_index.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_INDEX_HEADER)
            writer.writerows(split_rows)
    write_noise(root / "distant-16k/distractors/rm1/musi/distractor.wav", 1.0)
    return raw


# ---------------------------------------------------------------------------------------------------- LibriVAD

LIBRIVAD_SAMPLES = {"1089-134686-0000": 32000, "1089-134686-0001": 24000, "1188-133604-0016": 28000}


def save_samplewise_labels(folder: Path, labels: list[int], name: str = "labels.npy") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / name, np.asarray(labels, dtype=np.int16))
    return folder / name


# Concatenated pairs, with the folder their label sits in: the source tree keeps a pair in the folder of its second
# utterance when the two are from different chapters, which the stem does not say.
LIBRIVAD_CONCAT_FOLDERS = {
    "1089-134686-0000_+_0001": ("1089", "134686"),
    "1188-133604-0016_+_1089-134686-0001": ("1089", "134686"),
}


def make_librivad_tree(raw: Path, with_val: bool = True) -> Path:
    """`train-clean-100` and `test-clean` under `raw` (labels in `repo/Files/Labels`) and, when `with_val`, the layout
    the `librivad-val` fetcher is to write beside it: `dev-clean` under `generated/` and labels under `labels/`.
    Each utterance is mixed with two noises at two SNRs. A label is speech from its 3rd to its 7th tenth of a second,
    and the wav holds as many samples as its label.
    """
    trees: list[tuple[Path, str, tuple[str, ...]]] = [(raw, "repo/Files/Labels", ("train-clean-100", "test-clean"))]
    if with_val:
        trees.append((raw.with_name("librivad-val"), "labels", ("dev-clean",)))
    datasets = {
        "LibriSpeech": {stem: tuple(stem.split("-")[:2]) for stem in LIBRIVAD_SAMPLES},
        "LibriSpeechConcat": LIBRIVAD_CONCAT_FOLDERS,
    }
    for tree, labels, folders in trees:
        for dataset, stems in datasets.items():
            for folder in folders:
                for stem, (speaker, chapter) in stems.items():
                    n = LIBRIVAD_SAMPLES.get(stem, 56000)
                    label = np.zeros(n, np.int16)
                    label[4800 : 4800 + 6400] = 1
                    save_samplewise_labels(
                        tree / labels / dataset / folder / speaker / chapter, label.tolist(), f"{stem}.npy"
                    )
                    for noise_name in ("Babble_noise", "SSN_noise"):
                        for snr in ("5", "10"):
                            path = tree / "generated" / dataset / folder / noise_name / snr / f"{stem}.wav"
                            write_noise(path, n / 16000, seed=n)
    return raw
