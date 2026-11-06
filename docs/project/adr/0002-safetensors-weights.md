# 0002: safetensors weights and PyTorch parity

Status: Accepted (2026-10-30)

## Context

Weights trained in PyTorch must reach the Rust engine and the PyO3 bindings.
pickle can run code on load; safetensors cannot, and its header format is trivial.
The engine is hand-written ([0001](0001-no-onnx-in-engine.md)), so every layer needs a check against PyTorch.

## Decision

- Weights move from PyTorch to Rust only as safetensors files, never pickle.
- Tensor names are documented, and BatchNorm is folded before export.
- Parity is first-class. Golden vectors exported from PyTorch (inputs, outputs and per-layer activations) are checked against the Rust engine and the bindings per layer, offline and streaming.
- The tolerance is an absolute `1e-6`, no looser.
- The smoke fixture is the committed test file `inferencing/fixtures/smoke/linear.safetensors`.

Golden vectors check parity with PyTorch. Golden checksums check bitwise determinism across tiers and machines ([0026](0026-engine-determinism-and-kernel-tiers.md)).

## Consequences

- Loading a weight file never executes code.
- Tensor names are part of the contract between training and inference.
- A change that moves outputs by more than `1e-6` fails the parity check, and parity tests must pass before any change to the model or the engine is done.

## Open questions

- Where are the tensor names documented, and is a naming scheme decided? No document defines them yet.
