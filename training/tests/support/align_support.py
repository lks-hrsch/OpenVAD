# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A stand-in for `mfa align` in the align stage tests, and the synthetic clips it aligns.

The stand-in reads the corpus the stage wrote (a WAV and a `.lab` per file under one directory per speaker), finds
the speech in each WAV by its energy and writes `<speaker>/<file>.TextGrid` and `alignment_analysis.csv` in the
formats real MFA 3.4.3 wrote for `mfa_output/` (German synthetic speech). A transcript can ask for a behaviour:
`FAIL` gives no TextGrid and a blank analysis row (what MFA does at exit 0 for a transcript far too long), `BLANK`
a TextGrid with a blank row, `RETRY` a clip that needed the retry beam, `LOW` and `LOWER` log-likelihoods of -90
and -95 (else -50). Digits in a transcript would become `<unk>`, so markers are letters.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from openvad_train.dataset import align
from openvad_train.dataset.schema import Clip


FIXTURES = Path(__file__).parent / "mfa_output"
HEADER = [
    "file",
    "begin",
    "end",
    "speaker",
    "overall_log_likelihood",
    "speech_log_likelihood",
    "phone_duration_deviation",
    "max_running_short_interval",
    "snr",
    "intensity_deviation",
]


class Killed(BaseException):
    """A process kill: nothing catches it, as nothing catches SIGKILL."""


def _intervals(items: list[tuple[float, float, str]], duration: float) -> list[tuple[float, float, str]]:
    filled, at = [], 0.0
    for start, end, text in items:
        if start > at:
            filled.append((at, start, ""))
        filled.append((start, end, text))
        at = end
    if at < duration:
        filled.append((at, duration, ""))
    return filled


def textgrid(words: list[tuple[float, float, str]], duration: float) -> str:
    """A long-format TextGrid as MFA writes it, with one 0.1 s phone per word (`spn` for `<unk>`)."""
    phones = [(a, min(b, round(a + 0.1, 3)), "spn" if w == "<unk>" else "a") for a, b, w in words]
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = 0 \nxmax = {duration} ",
        "tiers? <exists> ",
        "size = 2 ",
        "item []: ",
    ]
    for n, (name, items) in enumerate((("words", words), ("phones", phones)), 1):
        shown = _intervals(items, duration)
        lines += [f"    item [{n}]:", '        class = "IntervalTier" ', f'        name = "{name}" ']
        lines += ["        xmin = 0 ", f"        xmax = {duration} ", f"        intervals: size = {len(shown)} "]
        for i, (a, b, text) in enumerate(shown, 1):
            lines += [
                f"        intervals [{i}]:",
                f"            xmin = {a} ",
                f"            xmax = {b} ",
                f'            text = "{text}" ',
            ]
    return "\n".join(lines) + "\n"


class FakeMfa:
    def __init__(self):
        self.calls: list[list[str]] = []  # the transcripts of each run, in order
        self.kill_at: int | None = None  # the index of the run that dies
        self.extra_textgrid = False
        self.no_csv = False  # exit 0 without an analysis file
        self.retried: set[str] = set()

    def __call__(self, tools, models, corpus: Path, out: Path, mfa_root: Path) -> None:
        if self.kill_at == len(self.calls):
            raise Killed
        self.retried = set()  # stems restart at 0 in every run, so a mark from an earlier run would hit another clip
        out.mkdir(parents=True, exist_ok=True)
        labs = sorted(corpus.glob("*/*.lab"))
        self.calls.append(sorted(lab.read_text().strip() for lab in labs))
        rows = []
        for lab in labs:
            text, speaker = lab.read_text().strip(), lab.parent.name
            audio, rate = sf.read(lab.with_suffix(".wav"))
            duration = len(audio) / rate
            row = [lab.stem, 0.0, duration, speaker, "", "", "", "", "", ""]
            rows.append(row)
            if "FAIL" in text:
                continue
            frames = np.abs(audio)[: len(audio) // 160 * 160].reshape(-1, 160)
            active = np.flatnonzero(frames.max(axis=1) > 0.05)
            onset, offset = (active[0] * 0.01, (active[-1] + 1) * 0.01) if len(active) else (0.0, duration)
            tokens = text.split()
            edges = np.linspace(onset, offset, len(tokens) + 1)
            words = [(round(float(edges[i]), 3), round(float(edges[i + 1]), 3), t) for i, t in enumerate(tokens)]
            (out / lab.parent.name).mkdir(parents=True, exist_ok=True)
            (out / lab.parent.name / f"{lab.stem}.TextGrid").write_text(textgrid(words, duration), encoding="utf-8")
            if "BLANK" in text:
                continue
            ll = next((v for k, v in {"LOWER": -95.0, "LOW": -90.0}.items() if k in text), -50.0)
            row[4:] = [ll, ll, 1.0, 0, 10.0, ""]
            if "RETRY" in text:
                self.retried.add(lab.stem)
        if self.extra_textgrid:
            (out / "x").mkdir(parents=True, exist_ok=True)
            (out / "x" / "u9999999.TextGrid").write_text(textgrid([], 1.0), encoding="utf-8")
        if self.no_csv:
            return
        with (out / "alignment_analysis.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(rows)


def install(monkeypatch) -> SimpleNamespace:
    """Replace the tools, the model download and `mfa align` in the stage; returns the fake and what was fetched."""
    fake, fetched = FakeMfa(), []
    tools = align.Tools(Path("."), "3.4.3", "a" * 40)

    def fetch(_tools, spec):
        fetched.append(spec.language)
        info = {"acoustic": {"name": spec.acoustic, "release": spec.acoustic_version, "version": "x", "sha256": "0"}}
        return align.Models(Path("acoustic.zip"), Path("dictionary.dict"), None, info)

    monkeypatch.setattr(align, "check_tools", lambda: tools)
    monkeypatch.setattr(align, "fetch_models", fetch)
    monkeypatch.setattr(align, "_run_mfa", fake)
    monkeypatch.setattr(align, "_retried", lambda mfa_root, corpus: set(fake.retried))
    return SimpleNamespace(fake=fake, fetched=fetched, tools=tools)


def make_clip(
    root: Path,
    clip_id: str,
    speaker: str | None,
    transcript: str = "aa bb cc",
    *,
    language: str = "de",
    burst: tuple[float, float] = (0.2, 1.3),
    total: float = 1.5,
    offset: float = 0.0,
    duration: float | None = None,
    corrupt: bool = False,
    reserved: bool = False,
) -> Clip:
    """A synthetic clip: `total` seconds of silence with a tone over `burst`, in `datasets/raw/fake/`, of which the
    clip is `duration` seconds from `offset`.
    """
    path = root / "datasets/raw/fake" / f"{clip_id.replace('/', '_')}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        path.write_bytes(b"not audio at all" * 100)
    else:
        t = np.arange(round(total * 16000)) / 16000
        tone = 0.3 * np.sin(2 * np.pi * 440 * t) * ((t >= burst[0]) & (t < burst[1]))
        sf.write(path, tone.astype(np.float32), 16000, subtype="PCM_16")
    return Clip(
        clip_id,
        "fake",
        language,
        speaker,
        "train",
        path.relative_to(root).as_posix(),
        offset,
        total - offset if duration is None else duration,
        "CC0",
        transcript,
        reserved,
    )
