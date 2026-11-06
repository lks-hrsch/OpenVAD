# 0004: Project scope

Status: Accepted (2026-10-30)

## Context

OpenVAD is a TU Dresden Master's thesis.
It must show how far a fully open voice activity detector gets on noisy telephone audio.

## Decision

- OpenVAD is a fully reproducible VAD: public data with documented labels, documented architecture, open training code and published weights.
  OpenVAD's own weights are published; datasets and third-party weights are never redistributed ([0024](0024-training-data-licence-policy.md)) (owner, 2026-11-04).
- Three research questions: (1) training data and labels, (2) model design and efficiency on CPU, (3) evaluation and generalisation against the reference VADs of [0003](0003-isolated-benchmark-adapters.md) on the benchmarks of [0020](0020-evaluation-benchmarks.md).
- Causal, frame-wise streaming.
- Training in PyTorch; inference hand-written in Rust (weights as safetensors, no ONNX Runtime, see [0001](0001-no-onnx-in-engine.md) and [0002](0002-safetensors-weights.md)), with parity tests against PyTorch and Python bindings via PyO3; no C API.
- Optimisation, quantisation and a native 8 kHz model are optional extensions.
- Headline results use at least 3 seeds and bootstrap confidence intervals over files.
- Thresholds are chosen on dev only.
- Every system except pyannote, which is a labelled non-causal upper bound, runs causally through one streaming interface and is fed identical chunks: the harness pushes the same fixed chunk into every system and each adapter buffers internally to its model's native call size.
  The OpenVAD resampler is used for every 8 and 16 kHz conversion; other rates in data preparation use ffmpeg ([0009](0009-native-16khz-and-one-resampler.md)) (owner, 2026-11-04) and outputs map onto the 10 ms grid ([spec](../spec/benchmark-predictions.md)).
- Algorithmic latency is reported per system, state is reset per file, and the speech prior is reported.
- The production voicebot runs on the company cluster. Machines and compute are listed in [build-and-hardware.md](../../notes/build-and-hardware.md).

## Consequences

- Every later ADR works inside this scope.
- Which data may be trained on is set by [0024](0024-training-data-licence-policy.md); the sources are listed in [0018](0018-datasets.md).
- Benchmark and metric detail lives in [0020](0020-evaluation-benchmarks.md).

## Open questions

- Does "thresholds on dev only" cover the second operating point that Fearless Steps gets from its own official Dev split ([0020](0020-evaluation-benchmarks.md))?
