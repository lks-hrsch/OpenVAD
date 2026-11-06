# 0011: Only the best model goes to Rust

Status: Accepted (2026-10-30)

## Context

Several backbones are candidates ([0012](0012-classifier-backbone.md)).
Each layer in the Rust engine is hand-written and parity-tested ([0001](0001-no-onnx-in-engine.md), [0002](0002-safetensors-weights.md)).

## Decision

Only the best model is ported to Rust.

## Consequences

- Comparison models live in PyTorch only.
- Which candidates are explored in PyTorch is open in [0012](0012-classifier-backbone.md).

## Open questions

- What does "best" mean: accuracy only, or accuracy at a given byte budget ([0008](0008-byte-size-budget.md))? No record defines it.
