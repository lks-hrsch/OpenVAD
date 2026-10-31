# 0003: Isolated benchmark adapters

Status: Accepted (2026-10-30)

## Context

The reference VADs have conflicting dependencies (NeMo, pyannote, onnxruntime). They cannot share one environment.

## Decision

Each reference VAD runs as its own uv project under `benchmarking/adapters/<system>/`. It is not a workspace member. Each adapter pins the exact upstream version and documents the licence and upstream link.

Systems talk through files, not imports, as defined in `docs/project/spec/benchmark-predictions.md`.

Reference systems: WebRTC VAD (lower bound), Silero VAD v6, TEN VAD, FireRedVAD. pyannote is a clearly labelled non-causal upper bound. VADigo itself (PyTorch and the Rust engine) runs through the same adapter interface.

TEN VAD has extra non-OSI conditions (Agora non-compete). Its adapter stays isolated and documents its licence.

Datasets and third-party weights are never redistributed. Scripts fetch them.

## Consequences

- Dependency conflicts stay inside one adapter.
- Every system is compared through the same files.
- Each adapter has its own lock file to maintain.
