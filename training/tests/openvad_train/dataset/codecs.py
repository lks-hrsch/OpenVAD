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

from openvad_rs import GROUP_DELAY_16K
from openvad_train.dataset import codecs
from openvad_train.dataset.codecs import (
    AMR_MODES_BPS,
    DecodeError,
    MissingCodec,
    decode_16k,
    delay_ms,
    encoder_of,
    measure_delay,
    narrow_roundtrip,
    preflight,
    roundtrip,
    stored_delay_ms,
    stored_roundtrip,
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
        y = roundtrip(x, codec, bitrate, rate)
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
@pytest.mark.parametrize(
    ("codec", "bitrate"),
    [
        ("g722", 64000),
        ("opus", 12000),
        ("opus", 32000),
        ("amr_wb", 6600),
        ("amr_wb", 23850),
        ("amr_nb", 4750),
        ("amr_nb", 12200),
    ],
)
def test_committed_delay_matches_a_fresh_measurement(codec, bitrate):
    require_encoders(encoder_of(codec))
    assert measure_delay(codec, bitrate) == pytest.approx(delay_ms(codec), abs=0.1)


def committed_measurements() -> list[tuple[str, int, float]]:
    table = json.loads(Path(codecs.__file__).with_name("codec_delays.json").read_text())["measured_ms"]
    return [(codec, int(bitrate), ms) for codec, by_bitrate in table.items() for bitrate, ms in by_bitrate.items()]


def test_the_committed_measurements_cover_every_amr_mode_and_the_opus_range():
    measured = {(c, b) for c, b, _ in committed_measurements()}
    assert {(c, b) for c in AMR_MODES_BPS for b in AMR_MODES_BPS[c]} <= measured
    assert {b for c, b in measured if c == "opus"} >= {12000, 32000}
    assert ("g722", 64000) in measured


def test_the_delay_does_not_depend_on_the_bitrate_in_the_committed_measurements():
    assert all(ms == delay_ms(codec) for codec, _, ms in committed_measurements())


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "bitrate", "ms"), committed_measurements())
def test_every_committed_measurement_reproduces(codec, bitrate, ms):
    require_encoders(encoder_of(codec))
    assert measure_delay(codec, bitrate) == pytest.approx(ms, abs=0.1)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("codec", "bitrate"),
    [
        ("g722", 64000),
        ("opus", 12000),
        ("opus", 32000),
        ("amr_wb", 6600),
        ("amr_wb", 23850),
        ("amr_nb", 4750),
        ("amr_nb", 12200),
    ],
)
def test_the_stored_leg_keeps_length_and_lags_by_the_stored_delay(codec, bitrate):
    require_encoders(encoder_of(codec))
    n = 2 * 16000
    x = chirp(n, 100, 3500)  # a chirp, because an impulse is a poor probe for AMR: its peak is 0.003 of 0.9
    x[:2000] = 0
    y = stored_roundtrip(x, codec, bitrate)
    assert y.shape == x.shape
    lag_ms = (np.argmax(signal.correlate(y, x, mode="full", method="fft")) - (n - 1)) / 16
    assert lag_ms == pytest.approx(stored_delay_ms(codec), abs=0.5)


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "bitrate"), [("g722", 64000), ("amr_wb", 12650), ("opus", 32000), ("amr_nb", 12200)])
def test_each_codec_keeps_length_and_the_impulse_within_ten_milliseconds_of_its_delay(codec, bitrate):
    require_encoders(encoder_of(codec))
    y = stored_roundtrip(impulse_at(8000, n=16000), codec, bitrate)
    assert len(y) == 16000
    assert abs(np.argmax(np.abs(y)) / 16 - (500 + stored_delay_ms(codec))) <= 10


def test_the_stored_delay_adds_the_resampler_to_amr_nb_only():
    assert stored_delay_ms("amr_nb") == delay_ms("amr_nb") + 66 / 16
    assert stored_delay_ms("amr_nb") == pytest.approx(9.125)
    assert [stored_delay_ms(c) for c in ("g722", "opus", "amr_wb")] == [delay_ms(c) for c in ("g722", "opus", "amr_wb")]


@needs_ffmpeg
def test_the_stored_leg_of_amr_nb_takes_16_khz_audio():
    require_encoders("libopencore_amrnb")
    x = chirp(16000, 100, 3500)
    np.testing.assert_array_equal(stored_roundtrip(x, "amr_nb", 12200), narrow_roundtrip(x, "amr_nb", 12200))


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
def test_a_stream_without_a_known_duration_is_treated_as_long(tmp_path, monkeypatch):
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
    verified = []
    monkeypatch.setattr(codecs, "_verify_intact", verified.append)
    monkeypatch.setattr(codecs, "_decode_whole_file", lambda *_args: pytest.fail("took the whole-file route"))
    assert len(decode_16k(str(path), 1.0, 1.0)) == 16000
    assert verified == [str(path)]


@needs_ffmpeg
@pytest.mark.parametrize("rate", [16000, 44100])
def test_both_routes_agree_on_a_short_file(tmp_path, monkeypatch, rate):
    path = encode_ogg(tmp_path, 12.0, rate)
    # A 20 ms grid is whole samples at both rates, so the two routes share a resampler phase.
    whole = decode_16k(str(path), 4.0, 3.0)
    monkeypatch.setattr(codecs, "_LONG_FILE_S", 0.0)
    seek = decode_16k(str(path), 4.0, 3.0)
    assert len(whole) == len(seek) == 48000
    assert np.argmax(signal.correlate(whole, seek, mode="full", method="fft")) - (len(seek) - 1) == 0
    # About 1e-6 measured on real Common Voice MP3s.
    np.testing.assert_allclose(whole, seek, atol=5e-6)


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


def tone(freq: float, n: int = 16000, rate: int = 16000) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / rate)).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


@pytest.mark.parametrize(("codec", "rate"), [("amr_nb", 16000), ("g722", 8000), ("amr_nb", 44100)])
def test_roundtrip_refuses_audio_at_the_wrong_rate(codec, rate):
    with pytest.raises(ValueError, match="Hz"):
        roundtrip(chirp(16000), codec, None, rate)


def test_amr_nb_needs_a_declared_rate_of_8_khz():
    with pytest.raises(ValueError, match="8000"):
        roundtrip(chirp(8000), "amr_nb", 12200)


def test_narrow_roundtrip_only_runs_amr_nb():
    with pytest.raises(ValueError, match="amr_nb"):
        narrow_roundtrip(chirp(16000), "g722", 64000)


def test_narrow_roundtrip_hands_the_codec_audio_inside_the_int16_range(monkeypatch):
    seen = []

    def fake(audio, codec, bitrate, rate):
        seen.append((audio, rate))
        return audio

    monkeypatch.setattr(codecs, "roundtrip", fake)
    loud = np.where(np.arange(16000) // 40 % 2 == 0, 1.0, -1.0).astype(np.float32)  # a square wave overshoots
    narrow_roundtrip(loud, "amr_nb", 12200)
    ((audio, rate),) = seen
    assert rate == 8000
    assert audio.max() <= 32767 / 32768
    assert audio.min() >= -1.0
    assert audio.max() > 0.9  # clipped, not scaled down


@needs_ffmpeg
@pytest.mark.parametrize("n", [16000, 16001])
def test_narrow_roundtrip_keeps_length_and_dtype(n):
    require_encoders("libopencore_amrnb")
    y = narrow_roundtrip(chirp(n, 100, 3500), "amr_nb", 12200)
    assert y.shape == (n,)
    assert y.dtype == np.float32


@needs_ffmpeg
def test_amr_nb_removes_energy_above_4_khz():
    require_encoders("libopencore_amrnb")
    y = narrow_roundtrip(tone(6000), "amr_nb", 12200)
    assert rms(y) < 0.01 * rms(tone(6000))


@needs_ffmpeg
def test_narrow_roundtrip_lags_by_the_codec_table_plus_the_resampler():
    require_encoders("libopencore_amrnb")
    n = 32000
    x = chirp(n, 100, 3500)
    x[:2000] = 0
    y = narrow_roundtrip(x, "amr_nb", 12200)
    lag_ms = (np.argmax(signal.correlate(y, x, mode="full", method="fft")) - (n - 1)) / 16
    assert lag_ms == pytest.approx(delay_ms("amr_nb") + 2 * GROUP_DELAY_16K / 16, abs=0.25)


@needs_ffmpeg
def test_narrow_roundtrip_refuses_a_decode_that_lost_more_than_a_frame(monkeypatch):
    require_encoders("libopencore_amrnb")
    real = codecs._run
    calls = []

    def truncating(cmd, data, error):
        done = real(cmd, data, error)
        calls.append(cmd)
        if len(calls) == 2:  # the decode: 50 ms of 8 kHz float32 plus the padding frame
            done.stdout = done.stdout[: len(done.stdout) - 4 * 8 * 50 - 4 * 160]
        return done

    monkeypatch.setattr(codecs, "_run", truncating)
    with pytest.raises(RuntimeError, match="decode returned"):
        narrow_roundtrip(chirp(16000, 100, 3500), "amr_nb", 12200)


@pytest.mark.parametrize(
    ("codec", "bitrate"),
    [("amr_nb", 12.2), ("amr_nb", 7000), ("amr_nb", None), ("amr_wb", 23.85), ("amr_wb", 13000), ("amr_wb", None)],
)
def test_roundtrip_refuses_a_bitrate_that_is_not_an_amr_mode(codec, bitrate):
    rate = 8000 if codec == "amr_nb" else 16000
    with pytest.raises(ValueError, match="bit/s"):
        roundtrip(chirp(rate, 100, 3500, rate), codec, bitrate, rate)  # pyright: ignore[reportArgumentType]


def test_narrow_roundtrip_inherits_the_mode_check():
    with pytest.raises(ValueError, match="bit/s"):
        narrow_roundtrip(chirp(16000), "amr_nb", 12.2)  # pyright: ignore[reportArgumentType]


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["amr_nb", "amr_wb"])
def test_the_requested_mode_reaches_the_encoder(monkeypatch, codec):
    encoder = encoder_of(codec)
    require_encoders(encoder)
    rate = 8000 if codec == "amr_nb" else 16000
    # Magic "#!AMR\n" or "#!AMR-WB\n", then the first frame's header byte holds the mode index in bits 3 to 6.
    header = 6 if codec == "amr_nb" else 9
    real = codecs._run
    encoded = []

    def spying(cmd, data, error):
        done = real(cmd, data, error)
        if "-c:a" in cmd:
            encoded.append(done.stdout)
        return done

    monkeypatch.setattr(codecs, "_run", spying)
    outputs = []
    for index, bitrate in enumerate(AMR_MODES_BPS[codec]):
        outputs.append(roundtrip(chirp(rate, 100, 0.4 * rate, rate), codec, bitrate, rate))
        assert encoded[-1][header - 1 : header] in (b"\n",)
        assert (encoded[-1][header] >> 3) & 0xF == index
    assert not np.array_equal(outputs[0], outputs[-1])
