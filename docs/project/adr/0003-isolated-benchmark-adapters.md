# 0003: Isolated benchmark adapters

Status: Accepted (2026-10-30)

## Context

The reference VADs have conflicting dependencies (pyannote, onnxruntime) and cannot share one environment.

## Decision

Each reference VAD runs as its own uv project under `benchmarking/adapters/<system>/`, with its own lock file. It is not a workspace member.
Each adapter pins the exact upstream version and documents the licence and upstream link.

Systems talk through files, not imports, as defined in the [benchmark prediction spec](../spec/benchmark-predictions.md).
Adapters read audio prepared offline and never import `openvad-rs`.

Reference systems:

- WebRTC VAD, the lower bound.
- Silero VAD v6.
- TEN VAD.
- FireRedVAD.
- pyannote, a clearly labelled non-causal upper bound.

OpenVAD itself (PyTorch and the Rust engine) runs through the same adapter interface.

TEN VAD has extra non-OSI conditions (Agora non-compete). Its adapter stays isolated and documents its licence.

Datasets and third-party weights are never redistributed; see [0024](0024-training-data-licence-policy.md).

## Consequences

- Dependency conflicts stay inside one adapter.
- Every system is compared through the same files.
- Each adapter has its own lock file to maintain.

## Open questions

- How does the Rust engine run through the adapter interface, given that adapters never import `openvad-rs`? Decided with the first adapter.
