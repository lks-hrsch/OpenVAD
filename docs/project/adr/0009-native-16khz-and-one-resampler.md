# 0009: Native 16 kHz model and one resampler

Status: Accepted (2026-10-30)

## Context

Production audio, from [build-and-hardware.md](../../notes/build-and-hardware.md):

- The voicebot negotiates one of three SIP codecs: G.722 (16 kHz), PCMA (G.711 A-law, 8 kHz) or PCMU (G.711 μ-law, 8 kHz).
- Everything inside the voicebot runs at 16 kHz. Narrowband calls are upsampled to 16 kHz with libsamplerate `sinc_best` (after about 3 dB of headroom) before the VAD and speech recognition. Narrowband audio has nothing above 4 kHz even after upsampling.
- The current VAD is Silero at 16 kHz, fed 512-sample blocks (32 ms).
- The owner found more wideband than narrowband calls (owner, 2026-11-04). The exact split is unknown; the gateway log query in the notes gives it.

Options considered: (A) one 16 kHz model, upsample 8 kHz input, which needs band-limited training data; (B) one 8 kHz model, downsample 16 kHz input; (C) two models; (D) shared backbone with rate-specific frontend. Upsampling adds no information.

The engine design follows from this: it runs natively at 16 kHz and upsamples 8 kHz input internally.

## Decision

- A native 16 kHz model comes first. A native 8 kHz model is an optional later extension.
- The engine runs natively at 16 kHz. 8 kHz input is upsampled inside the engine by a fixed, causal, linear-phase finite impulse response half-band interpolator (2:1) with committed coefficients and the canonical `mul_add` reduction order of every kernel tier ([0026](0026-engine-determinism-and-kernel-tiers.md)). Any other input rate is an error.
- The input rate is set once at stream creation (for example `Stream::new(model, sample_rate)`) and fixed for the stream's lifetime.
- The streaming API accepts any block length (for example `Stream::push(samples)`) and buffers internally up to the model's stride.
- One resampler for 8 and 16 kHz conversion: `openvad-core` provides a 2:1 decimator and a 2:1 interpolator, exposed via `openvad-rs`. The engine, training and the benchmarks all use it. It is fixed and documented. Other rates in data preparation use ffmpeg, with the version recorded in the manifest (owner, 2026-11-04).
- The filter is fixed in the dataset construction phase (owner, 2026-11-06): a half-band decimator and interpolator, causal and linear-phase, 67 taps, at least 70 dB stopband, with committed coefficients and explicit `mul_add` in the canonical order, in the scalar tier only.
  - 63 taps and 70 dB were the starting values, and the design step could adjust them before the coefficients were committed. It moved to 67 taps: 63 reach only -69.6 dB, and 67 reach -73.3 dB.
  - A committed script designs the coefficients, and the hash of the coefficients goes in the dataset manifest.
  - `openvad-rs` exposes both on NumPy arrays through the `numpy` crate, because `Vec<f32>` from Python lists is too slow for dataset work.
  - Tests cover the frequency response against the stated stopband, the group delay, known vectors, and a Python and Rust comparison at `1e-6`.
- Nothing is stored at 8 kHz (owner, 2026-11-06). Band-limited training augmentation along the production narrowband path (16 kHz to 8 kHz, G.711, back to 16 kHz) runs at train time with this resampler, as defined in [0021](0021-telephone-channel-simulation.md).

## Consequences

- The resampler's group delay counts towards algorithmic latency and is reported.
- Production uses libsamplerate `sinc_best` instead. The filter difference is noted, and it may be tested as a robustness check.
- The resampler and the streaming API do not exist yet; each is added with its first user. The dataset construction is the resampler's first user ([dataset construction spec](../spec/dataset-construction.md)).
