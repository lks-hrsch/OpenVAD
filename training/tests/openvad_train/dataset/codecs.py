# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from openvad_train.dataset import codecs
from openvad_train.dataset.codecs import (
    MissingCodec,
    delay_ms,
    measure_delay,
    preflight,
    roundtrip,
)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def chirp(n: int, f0: float = 100.0, f1: float = 7000.0, rate: int = 16000) -> np.ndarray:
    t = np.arange(n) / rate
    return (0.5 * signal.chirp(t, f0, n / rate, f1)).astype(np.float32)


def impulse_at(i: int, n: int) -> np.ndarray:
    x = np.zeros(n, np.float32)
    x[i] = 0.8
    return x


def require_encoders(*names: str) -> None:
    try:
        preflight(set(names))
    except MissingCodec as e:
        pytest.skip(f"this ffmpeg lacks {', '.join(names)}: {e}")


@pytest.fixture
def fake_ffmpeg_without_amr(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ffmpeg"
    script.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *-encoders*) printf ' A....D g722   G.722 ADPCM (codec adpcm_g722)\\n"
        " A....D libopus   libopus Opus\\n' ;;\n"
        '  *) echo "ffmpeg version 9.9.9 Copyright (c) test"; echo "configuration: --enable-libopus --disable-amr" ;;\n'
        "esac\n"
    )
    script.chmod(0o755)
    probe = bin_dir / "ffprobe"
    probe.write_text('#!/bin/sh\necho "ffprobe version 9.9.9"\n')
    probe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


def test_preflight_names_missing_encoder(fake_ffmpeg_without_amr):
    with pytest.raises(MissingCodec, match="libopencore_amrnb"):
        preflight({"libopencore_amrnb", "g722"})


def test_preflight_reports_version_and_flags(fake_ffmpeg_without_amr):
    info = preflight({"g722", "libopus"})
    assert info["version"] == "9.9.9"
    assert info["configuration"] == ["--enable-libopus", "--disable-amr"]


def test_preflight_names_a_missing_ffprobe(fake_ffmpeg_without_amr):
    (fake_ffmpeg_without_amr / "ffprobe").unlink()
    with pytest.raises(MissingCodec, match="ffprobe"):
        preflight({"g722"})


def test_preflight_without_ffmpeg_is_missing_codec(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(MissingCodec, match="ffmpeg"):
        preflight(set())


@needs_ffmpeg
def test_preflight_on_this_machine_finds_g722():
    info = preflight({"g722"})
    assert info["version"]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("codec", "bitrate", "encoder"),
    [
        ("g722", 64000, "g722"),
        ("opus", 12000, "libopus"),
        ("opus", 32000, "libopus"),
        ("amr_wb", 6600, "libvo_amrwbenc"),
        ("amr_wb", 23850, "libvo_amrwbenc"),
        ("amr_nb", 4750, "libopencore_amrnb"),
        ("amr_nb", 12200, "libopencore_amrnb"),
    ],
)
def test_roundtrip_keeps_length(codec, bitrate, encoder):
    require_encoders(encoder)
    rate = 8000 if codec == "amr_nb" else 16000
    for n in (rate, rate + 123):
        x = chirp(n, 100, 0.45 * rate, rate)
        y = roundtrip(x, codec, bitrate)
        assert y.shape == x.shape
        assert y.dtype == np.float32


@needs_ffmpeg
def test_g722_roundtrip_keeps_length():
    x = chirp(16000)
    assert len(roundtrip(x, "g722", 64000)) == len(x)


@needs_ffmpeg
def test_roundtrip_of_sixty_seconds_does_not_deadlock():
    x = chirp(60 * 16000)
    assert len(roundtrip(x, "g722", 64000)) == len(x)


@needs_ffmpeg
def test_roundtrip_error_carries_ffmpeg_stderr():
    with pytest.raises(RuntimeError, match="out of range"):
        roundtrip(chirp(16000), "g722", -5)


def test_roundtrip_rejects_an_unknown_codec():
    with pytest.raises(ValueError, match="codec"):
        roundtrip(chirp(16000), "mp3", None)


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "bitrate"), [("g722", 64000), ("opus", 12000), ("opus", 32000)])
def test_impulse_lands_at_table_delay(codec, bitrate):
    x = impulse_at(8000, n=16000)
    y = roundtrip(x, codec, bitrate)
    assert abs(np.argmax(np.abs(y)) / 16 - (500 + delay_ms(codec))) <= 10


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "bitrate"), [("g722", 64000), ("opus", 12000), ("opus", 32000)])
def test_committed_delay_matches_a_fresh_measurement(codec, bitrate):
    assert measure_delay(codec, bitrate) == pytest.approx(delay_ms(codec), abs=0.1)


@needs_ffmpeg
def test_measure_delay_is_positive_when_the_output_lags():
    # G.722 has a 22-sample QMF delay at 16 kHz.
    assert measure_delay("g722", 64000) == pytest.approx(1.375, abs=0.1)


def test_delay_table_names_the_ffmpeg_it_was_measured_on():
    table = json.loads(Path(codecs.__file__).with_name("codec_delays.json").read_text())
    assert table["ffmpeg"]


def test_delay_of_an_unmeasured_codec_is_an_error():
    with pytest.raises(KeyError, match="carrier_pigeon"):
        delay_ms("carrier_pigeon")


@needs_ffmpeg
@pytest.mark.parametrize(("lost_ms", "raises"), [(10, False), (50, True)])
def test_roundtrip_refuses_a_decode_that_lost_more_than_a_frame(monkeypatch, lost_ms, raises):
    real = codecs._run
    calls = []

    def truncating(cmd, data, error):
        done = real(cmd, data, error)
        calls.append(cmd)
        if len(calls) == 2:  # the decode
            done.stdout = done.stdout[: len(done.stdout) - 4 * 16 * lost_ms - 4 * 320]
        return done

    monkeypatch.setattr(codecs, "_run", truncating)
    x = chirp(16000)
    if raises:
        with pytest.raises(RuntimeError, match="decode returned"):
            roundtrip(x, "g722", 64000)
    else:
        assert len(roundtrip(x, "g722", 64000)) == len(x)


def test_roundtrip_without_ffmpeg_is_missing_codec_not_a_codec_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(MissingCodec, match="ffmpeg"):
        roundtrip(chirp(16000), "g722", 64000)
