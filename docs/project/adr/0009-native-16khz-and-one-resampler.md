# 0009: Native 16 kHz model and one resampler

Status: Accepted (2026-10-30), replaces the earlier 8 kHz-first decision

## Context

- Options were
    - (A) one 16 kHz model, upsample 8 kHz input (needs band-limited training data);
    - (B) one 8 kHz model, downsample 16 kHz input;
    - (C) two models;
    - (D) shared backbone with rate-specific frontend. Upsampling adds no information.
- The earlier proposal chose the main model by what the voicebot on the company cluster receives (likely G.711 at 8 kHz, option B) and ran "native 8 kHz vs. 16 kHz with upsampled narrowband" as one comparison.
- The resampler must be fixed and documented.

## Decision

- A native 16 kHz model comes first.
- Training uses band-limited augmentation that follows the production narrowband path: 16 kHz downsampled to 8 kHz, G.711 A-law and μ-law encode/decode, upsampled back to 16 kHz. The share of narrowband copies is decided in the data phase.
- A native 8 kHz model is an optional later extension.
- The engine runs natively at 16 kHz. 8 kHz input is upsampled inside the engine by a fixed, causal, linear-phase finite impulse response half-band interpolator (2:1) with committed coefficients and the same canonical `mul_add` reduction order in every kernel tier. Any other input rate is an error. Filter length and stopband are decided in the model-design phase.
- The input rate is set once at stream creation (e.g. `Stream::new(model, sample_rate)`) and fixed for the stream's lifetime.
- One resampler for everything: `vadigo-core` provides a 2:1 decimator and a 2:1 interpolator, exposed via `vadigo-rs`, used for training augmentation, deriving the 8 kHz test condition, the reference systems' input and the engine.
- The streaming API accepts any block length (`Stream::push(samples)`) and buffers internally up to the model's stride.

## Consequences

- The resampler's group delay counts towards algorithmic latency and is reported.
- Production uses libsamplerate `sinc_best` instead; the filter difference is noted, and it may be tested as a robustness check.
