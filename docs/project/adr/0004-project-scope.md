# 0004: Project scope

Status: Accepted (2026-10-30)

## Context

VADigo is a TU Dresden Master's thesis.
It must show how far a fully open voice activity detector gets on noisy telephone audio.

## Decision

- VADigo is a fully reproducible VAD: public data with documented labels, documented architecture, open training code, published weights.
- Three research questions: (1) training data and labels, (2) model design and efficiency on CPU, (3) evaluation and generalisation vs. selected existing VADs (WebRTC, Silero v6, TEN VAD, FireRedVAD) on our own test set and an independent real-world benchmark.
- Causal, frame-wise streaming.
- Training in PyTorch; inference hand-written in Rust (weights as safetensors, no ONNX Runtime, see [0001](0001-no-onnx-in-engine.md) and [0002](0002-safetensors-weights.md)), with parity tests against PyTorch and Python bindings via PyO3; no C API.
- Optimisation, quantisation and a native 8 kHz model are optional extensions.
- Compute: workstation (Ryzen 9 7900X + RTX 4090), laptop (Apple M3 Pro), TU Dresden HPC (Xeon Platinum 8470, EPYC 9334), and the company cluster (~Xeon Gold 6230), where the production voicebot runs.
- At least 3 seeds and bootstrap CIs over files for headline results.
- Thresholds are chosen on dev only.
- Every system, including the reference VADs, is fed identical chunks (same resampler and 10 ms grid): the harness pushes the same fixed chunk into every system and each adapter buffers internally to its model's native call size. Algorithmic latency is reported per system. State is reset per file. The speech prior is reported.
- No non-commercial data for training the released model.

## Consequences

- Every later ADR works inside this scope.
- Datasets with non-commercial licences can only appear in evaluation or related work (see [0018](0018-datasets.md)).
