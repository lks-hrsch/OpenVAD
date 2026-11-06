# 0016: INT8 post-training quantisation optional, pruning cut

Status: Accepted (2026-10-30)

## Context

- FP32 is the parity reference ([0002](0002-safetensors-weights.md)).
- INT8 is the relevant format for production: the Xeon Gold 6230 has AVX-512 VNNI.
- BF16 needs no QAT and is not supported on the Xeon Gold 6230.
- Unstructured sparsity gives the dense Rust GEMV no speed-up.

## Decision

- INT8 post-training quantisation (W8A32, then W8A8) stays an optional extension on the final model.
- QAT only if post-training quantisation loses noticeably.
- Keep LSTM cell state and gates in higher precision.
- Pruning is cut.
- BF16 may be studied in PyTorch only. The Rust engine has no BF16 kernels, and no FP16 or AMX kernels either, unless the owner agrees ([0026](0026-engine-determinism-and-kernel-tiers.md)).

## Open questions

- How is INT8 parity checked: which tolerance, against which reference?
- Does bitwise identity across kernel tiers apply to W8A32 as well? [0026](0026-engine-determinism-and-kernel-tiers.md) and the notes justify it for the integer path (exact integer accumulation); W8A32 has FP32 activations.
- Does the higher-precision rule apply if the final model is not an LSTM, for example a GRU or Mamba ([0012](0012-classifier-backbone.md) is Proposed)?
