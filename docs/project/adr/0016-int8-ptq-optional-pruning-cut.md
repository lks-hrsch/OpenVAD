# 0016: INT8 post-training quantisation optional, pruning cut

Status: Accepted (2026-10-30)

## Context

- FP32 is the parity reference.
- INT8 is the relevant format for production (Xeon Gold 6230 has AVX-512 VNNI).
- BF16 needs no QAT and is not supported on the Xeon Gold 6230.
- Unstructured sparsity gives the dense Rust GEMV no speed-up.

## Decision

- INT8 post-training quantisation (W8A32, then W8A8) stays an optional extension on the final model.
- QAT only if post-training quantisation loses noticeably. Keep LSTM cell state and gates in higher precision.
- Pruning is cut.

## Consequences

- BF16 may be studied in PyTorch only; the Rust engine has no BF16 kernels (see [build-and-hardware.md](../../notes/build-and-hardware.md)).
