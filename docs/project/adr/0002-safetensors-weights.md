# 0002: safetensors weights

Status: Accepted (2026-10-30)

## Context

Weights trained in PyTorch must reach the Rust engine and the PyO3 bindings. pickle can run code on load. safetensors cannot, and its header format is trivial.

## Decision

Weights move from PyTorch to Rust only as safetensors files with documented tensor names. BatchNorm is folded before export.

Parity is first-class. Golden vectors exported from PyTorch are checked against the Rust engine and the bindings with an absolute tolerance of `1e-6`. The smoke fixture is `inferencing/fixtures/smoke/linear.safetensors`.

## Consequences

- Loading a weight file never executes code.
- Tensor names are part of the contract between training and inference.
- A change that moves outputs by more than `1e-6` fails the parity check.
