# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from openvad_train.dataset import codecs
from openvad_train.dataset.codecs import (
    DecodeError,
    MissingCodec,
    decode_16k,
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


# decode_16k


@pytest.fixture(params=["whole_file", "seek"])
def route(request, monkeypatch):
    """Run a test on both routes through `decode_16k`: a limit of 0 sends every file down the seek route."""
    if request.param == "seek":
        monkeypatch.setattr(codecs, "_LONG_FILE_S", 0.0)


def burst(rate: int, at_s: float, n: int, dur_s: float = 0.05) -> np.ndarray:
    """A 200 to 3000 Hz chirp burst of dur_s starting at at_s, in a silent signal of n samples."""
    x = np.zeros(n, np.float32)
    i = round(at_s * rate)
    m = round(dur_s * rate)
    t = np.arange(m) / rate
    x[i : i + m] = 0.5 * signal.chirp(t, 200, dur_s, 3000)
    return x


def lag_samples_16k(y: np.ndarray, expected_at_s: float) -> float:
    """Samples by which the burst in y lags where a 16 kHz burst at expected_at_s would be."""
    ref = burst(16000, expected_at_s, len(y))
    c = signal.correlate(y, ref, mode="full", method="fft")
    return float(np.argmax(c) - (len(y) - 1))


@needs_ffmpeg
@pytest.mark.parametrize("rate", [22050, 44100, 48000])
def test_decode_resamples_and_keeps_time(tmp_path, rate, route):
    total = 4.0
    x = burst(rate, 2.2345, round(total * rate))
    path = tmp_path / "a.wav"
    sf.write(path, x, rate)
    y = decode_16k(str(path), 1.0003, 1.5)
    assert y.dtype == np.float32
    assert y.shape == (24000,)
    assert abs(lag_samples_16k(y, 2.2345 - 1.0003)) <= 16


@needs_ffmpeg
def test_decode_ogg_with_an_offset(tmp_path, route):
    x = burst(44100, 2.5, 4 * 44100)
    path = tmp_path / "a.ogg"
    sf.write(path, x, 44100, format="OGG", subtype="VORBIS")
    y = decode_16k(str(path), 1.25, 2.0)
    assert y.shape == (32000,)
    assert abs(lag_samples_16k(y, 2.5 - 1.25)) <= 16


@needs_ffmpeg
def test_decode_ogg_already_16k_with_an_offset(tmp_path, route):
    x = burst(16000, 2.5, 4 * 16000)
    path = tmp_path / "a.ogg"
    sf.write(path, x, 16000, format="OGG", subtype="VORBIS")
    y = decode_16k(str(path), 1.25, 2.0)
    assert y.shape == (32000,)
    assert abs(lag_samples_16k(y, 2.5 - 1.25)) <= 16


def test_decode_of_a_16k_file_is_exact(tmp_path, route):
    x = burst(16000, 1.0, 3 * 16000)
    path = tmp_path / "a.flac"
    sf.write(path, x, 16000, subtype="PCM_24")
    y = decode_16k(str(path), 0.5, 1.5)
    np.testing.assert_allclose(y, x[8000:32000], atol=1e-6)


@needs_ffmpeg
@pytest.mark.parametrize("rate", [16000, 44100])
def test_decode_downmixes_by_the_mean_of_channels(tmp_path, rate, route):
    n = 2 * rate
    left, right = np.full(n, 0.4, np.float32), np.full(n, -0.1, np.float32)
    path = tmp_path / "a.wav"
    sf.write(path, np.stack([left, right], axis=1), rate, subtype="FLOAT")
    y = decode_16k(str(path), 0.5, 1.0)
    assert y.shape == (16000,)
    np.testing.assert_allclose(y[2000:-2000], 0.15, atol=1e-3)


@needs_ffmpeg
def test_decode_downmixes_three_channels_by_the_mean(tmp_path, route):
    n = 44100
    x = np.stack([np.full(n, v, np.float32) for v in (0.3, 0.0, 0.3)], axis=1)
    path = tmp_path / "a.wav"
    sf.write(path, x, 44100, subtype="FLOAT")
    y = decode_16k(str(path), 0.0, 1.0)
    np.testing.assert_allclose(y[2000:-2000], 0.2, atol=1e-3)


@needs_ffmpeg
@pytest.mark.parametrize("rate", [16000, 22050])
def test_decode_returns_what_exists_when_the_file_ends_early(tmp_path, rate, route):
    path = tmp_path / "a.wav"
    sf.write(path, burst(rate, 0.1, rate), rate)
    y = decode_16k(str(path), 0.5, 2.0)
    assert abs(len(y) - 8000) <= 1


@needs_ffmpeg
@pytest.mark.parametrize("rate", [16000, 22050, 11025, 44100])
@pytest.mark.parametrize("offset_s", [1.0, 1.00005, 3.0])
def test_decode_offset_at_or_past_the_end_is_empty(tmp_path, rate, offset_s, route):
    path = tmp_path / "a.wav"
    sf.write(path, burst(rate, 0.1, rate), rate)
    assert len(decode_16k(str(path), offset_s, 1.0)) == 0


@needs_ffmpeg
def test_decode_of_a_garbage_file_raises_decode_error(tmp_path):
    path = tmp_path / "a.mp3"
    path.write_bytes(b"this is not audio" * 100)
    with pytest.raises(DecodeError):
        decode_16k(str(path), 0.0, 1.0)


@needs_ffmpeg
def test_decode_of_a_missing_file_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        decode_16k(str(tmp_path / "nope.wav"), 0.0, 1.0)


@needs_ffmpeg
def test_decode_of_a_corrupt_16k_file_raises_decode_error(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 60)
    with pytest.raises(DecodeError):
        decode_16k(str(path), 0.0, 1.0)


@needs_ffmpeg
def test_decode_error_carries_ffmpeg_stderr(tmp_path):
    path = tmp_path / "a.mp3"
    path.write_bytes(b"this is not audio" * 100)
    with pytest.raises(DecodeError, match="Invalid data found"):
        decode_16k(str(path), 0.0, 1.0)


@needs_ffmpeg
def test_decode_of_sixty_seconds_does_not_deadlock(tmp_path, route):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(60 * 44100, np.float32), 44100)
    assert len(decode_16k(str(path), 0.0, 60.0)) == 60 * 16000


@needs_ffmpeg
def test_decode_leaves_the_source_untouched(tmp_path, route):
    path = tmp_path / "a.wav"
    sf.write(path, burst(44100, 0.1, 44100), 44100)
    before = path.read_bytes()
    decode_16k(str(path), 0.0, 1.0)
    assert path.read_bytes() == before


@needs_ffmpeg
def test_decode_without_soundfile_slices_a_full_decode(tmp_path, route):
    # An AAC file in an MP4 container is one soundfile cannot open, so this takes the ffmpeg fallback.
    src = tmp_path / "a.wav"
    sf.write(src, burst(44100, 2.5, 4 * 44100), 44100)
    path = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "aac", str(path)], check=True)
    with pytest.raises(RuntimeError):
        sf.info(path)
    y = decode_16k(str(path), 1.25, 2.0)
    assert y.shape == (32000,)
    assert abs(lag_samples_16k(y, 2.5 - 1.25)) <= 16


def test_decode_refuses_a_negative_offset(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        decode_16k(str(tmp_path / "a.wav"), -1.0, 1.0)


def encode_ogg(tmp_path, seconds: float, rate: int = 44100, seed: int = 0):
    """A band-limited noise Ogg Vorbis made by ffmpeg (sf.write of 120 s Vorbis segfaulted in libsndfile 1.2.2)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(round(seconds * rate)).astype(np.float32)
    sos = signal.butter(4, 6000, fs=rate, output="sos")
    x = (0.2 * np.asarray(signal.sosfilt(sos, noise))).astype(np.float32)
    wav, ogg = tmp_path / "n.wav", tmp_path / "n.ogg"
    sf.write(wav, x, rate, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "libvorbis", "-q:a", "5", str(ogg)], check=True)
    return ogg


def full_decode_16k(path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32)


def overwrite(path, at: float, n_bytes: int = 20000):
    data = bytearray(path.read_bytes())
    i = int(len(data) * at)
    data[i : i + n_bytes] = np.random.default_rng(1).integers(0, 256, n_bytes, dtype=np.uint8).tobytes()
    path.write_bytes(bytes(data))


@needs_ffmpeg
def test_decode_of_a_damaged_ogg_raises_instead_of_returning_shifted_audio(tmp_path):
    path = encode_ogg(tmp_path, 120.0)
    overwrite(path, 0.3)
    with pytest.raises(DecodeError, match=r"\.ogg"):
        decode_16k(str(path), 20.0, 90.0)


@needs_ffmpeg
def test_decode_of_a_damaged_m4a_without_soundfile_raises(tmp_path, route):
    src = encode_ogg(tmp_path, 30.0)
    path = tmp_path / "n.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "aac", str(path)], check=True)
    overwrite(path, 0.3, 10000)
    with pytest.raises(DecodeError, match=r"\.m4a"):
        decode_16k(str(path), 2.0, 10.0)


@needs_ffmpeg
def test_decode_of_a_short_damaged_ogg_raises(tmp_path):
    path = encode_ogg(tmp_path, 30.0)
    overwrite(path, 0.3, 5000)
    with pytest.raises(DecodeError, match=r"\.ogg"):
        decode_16k(str(path), 2.0, 5.0)


def count_subprocesses(monkeypatch) -> list[list[str]]:
    real, calls = codecs._run, []

    def counting(cmd, data, error):
        calls.append(cmd)
        return real(cmd, data, error)

    monkeypatch.setattr(codecs, "_run", counting)
    return calls


@needs_ffmpeg
def test_a_short_file_spawns_one_decode_and_no_separate_verify(tmp_path, monkeypatch):
    calls = count_subprocesses(monkeypatch)
    monkeypatch.setattr(codecs, "_verify_intact", lambda path: pytest.fail(f"verified {path} separately"))
    path = encode_ogg(tmp_path, 10.0)
    for _ in range(2):
        assert len(decode_16k(str(path), 1.0, 2.0)) == 32000
    assert len(calls) == 2


@needs_ffmpeg
def test_a_file_soundfile_cannot_open_is_routed_by_ffprobe(tmp_path, monkeypatch):
    src = encode_ogg(tmp_path, 3.0)
    path = tmp_path / "n.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "aac", str(path)], check=True)
    calls = count_subprocesses(monkeypatch)
    monkeypatch.setattr(codecs, "_verify_intact", lambda path: pytest.fail(f"verified {path} separately"))
    assert len(decode_16k(str(path), 1.0, 1.0)) == 16000
    assert [cmd[0] for cmd in calls] == ["ffprobe", "ffmpeg"]


@needs_ffmpeg
def test_a_long_file_is_verified_once_and_then_only_seeks(tmp_path, monkeypatch):
    monkeypatch.setattr(codecs, "_LONG_FILE_S", 5.0)
    path = encode_ogg(tmp_path, 10.0, rate=16000)
    calls = count_subprocesses(monkeypatch)
    for offset in (1.0, 3.0, 7.0):
        assert len(decode_16k(str(path), offset, 1.0)) == 16000
    assert len(calls) == 1


@needs_ffmpeg
def test_a_file_at_the_limit_stays_on_the_whole_file_route(tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    sf.write(path, burst(16000, 0.1, 16000), 16000)
    calls = count_subprocesses(monkeypatch)
    monkeypatch.setattr(codecs, "_LONG_FILE_S", 1.0)
    decode_16k(str(path), 0.0, 1.0)
    assert len(calls) == 1
    monkeypatch.setattr(codecs, "_LONG_FILE_S", 0.999)
    decode_16k(str(path), 0.0, 1.0)
    assert len(calls) == 2  # now its one verify, and soundfile reads the samples


@needs_ffmpeg
def test_a_stream_without_a_known_duration_takes_the_seek_route(tmp_path, monkeypatch):
    real = codecs._run

    def unknown_duration(cmd, data, error):
        done = real(cmd, data, error)
        if cmd[0] == "ffprobe":
            done.stdout = done.stdout.replace(b"duration=", b"duration=N/A\nignored=")
        return done

    src = encode_ogg(tmp_path, 3.0)
    path = tmp_path / "n.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "aac", str(path)], check=True)
    monkeypatch.setattr(codecs, "_run", unknown_duration)
    assert codecs._probe(str(path))[2] == float("inf")


@needs_ffmpeg
@pytest.mark.parametrize("rate", [16000, 44100])
def test_both_routes_agree_on_a_short_file(tmp_path, monkeypatch, rate):
    path = encode_ogg(tmp_path, 12.0, rate)
    # A 20 ms grid is whole samples at both rates, so the two routes share a resampler phase.
    whole = decode_16k(str(path), 4.0, 3.0)
    monkeypatch.setattr(codecs, "_LONG_FILE_S", 0.0)
    seek = decode_16k(str(path), 4.0, 3.0)
    assert len(whole) == len(seek) == 48000
    np.testing.assert_allclose(whole, seek, atol=2e-3)


@needs_ffmpeg
def test_decode_of_a_clean_ogg_passes_the_integrity_check(tmp_path, route):
    path = encode_ogg(tmp_path, 30.0)
    assert len(decode_16k(str(path), 5.0, 10.0)) == 160000


@needs_ffmpeg
@pytest.mark.parametrize("rate", [44100, 22050])
@pytest.mark.parametrize("back_s", [0.05, 0.2, 0.5, 1.5, 4.9, 6.0])
def test_decode_near_the_end_of_an_ogg_matches_a_full_decode(tmp_path, monkeypatch, rate, back_s):
    # ffmpeg-encoded Vorbis seeks wrongly (up to 832 source samples) inside its last page: the seek route's tail rule.
    monkeypatch.setattr(codecs, "_LONG_FILE_S", 0.0)
    path = encode_ogg(tmp_path, 12.0, rate)
    full = full_decode_16k(path)
    total = len(full) / 16000
    # A 20 ms grid is whole samples at 16, 22.05 and 44.1 kHz, so the segment and the full decode share a phase.
    offset = np.floor((total - back_s) / 0.02) * 0.02
    n = min(round(0.15 * 16000), len(full) - round(offset * 16000))
    y = decode_16k(str(path), offset, n / 16000)
    ref = full[round(offset * 16000) :][:n]
    assert len(y) == n
    corr = signal.correlate(y, ref, mode="full", method="fft")
    assert np.argmax(corr) - (len(ref) - 1) == 0
    np.testing.assert_allclose(y, ref, atol=2e-3)


@needs_ffmpeg
def test_decode_of_8k_audio_is_refused_until_the_resampler_exists(tmp_path, route):
    path = tmp_path / "a.wav"
    sf.write(path, burst(8000, 0.1, 8000), 8000)
    with pytest.raises(ValueError, match="ADR 0009"):
        decode_16k(str(path), 0.0, 1.0)


@needs_ffmpeg
def test_decode_of_8k_audio_on_the_fallback_is_refused_too(tmp_path, route):
    src = tmp_path / "a.wav"
    sf.write(src, burst(8000, 0.1, 8000), 8000)
    path = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "aac", str(path)], check=True)
    with pytest.raises(ValueError, match="ADR 0009"):
        decode_16k(str(path), 0.0, 1.0)


@needs_ffmpeg
@pytest.mark.parametrize("rate", [16000, 44100])
def test_decoded_audio_is_writable_on_every_path(tmp_path, rate, route):
    path = tmp_path / "a.wav"
    sf.write(path, burst(rate, 0.1, 2 * rate), rate)
    y = decode_16k(str(path), 0.0, 1.0)
    y *= 2.0
    assert y.flags.writeable


@needs_ffmpeg
def test_decoded_audio_is_writable_on_the_fallback(tmp_path, route):
    src = tmp_path / "a.wav"
    sf.write(src, burst(44100, 0.1, 44100), 44100)
    path = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "aac", str(path)], check=True)
    y = decode_16k(str(path), 0.0, 0.5)
    y *= 2.0
    assert y.flags.writeable


def test_decode_without_ffmpeg_is_missing_codec_not_a_corrupt_clip(tmp_path, monkeypatch, route):
    path = tmp_path / "a.wav"
    sf.write(path, burst(16000, 0.1, 16000), 16000)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(MissingCodec, match="ffmpeg"):
        decode_16k(str(path), 0.0, 1.0)
