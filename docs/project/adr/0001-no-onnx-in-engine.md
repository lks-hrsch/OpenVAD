# 0001: No ONNX in the engine

Status: Accepted (2026-10-30)

## Context

The thesis studies the model and its efficiency end to end.
The Rust inference engine (`inferencing/openvad-core`) must keep every operation inspectable and stay light on dependencies.

## Decision

The engine is a hand-written forward pass.
It does not use ONNX Runtime, tract, burn or candle.

ONNX Runtime is allowed in `benchmarking/` only: for the speed baseline (Silero VAD on ONNX Runtime with `intra_op_num_threads = 1`) and for adapters whose upstream system needs it ([0003](0003-isolated-benchmark-adapters.md)).

`just guard-engine` enforces this.
It fails if `cargo tree` for the workspace (which includes `openvad-rs`) lists `ort`, `tract`, `burn` or `candle`, or a crate whose name extends one of them with a hyphen, for example `tract-core`.

## Consequences

- Every operation in the engine is our own code and can be read, tested and profiled.
- We write and maintain each layer we need.
- Adding an inference crate to the workspace breaks `just guard-engine`.
