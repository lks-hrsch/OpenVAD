# 0015: GPU training, CPU timing

Status: Accepted (2026-10-30)

## Context

The experiments need many runs × seeds.
The production target and the Rust engine are CPU only.

## Decision

- Train on GPU; run several small runs in parallel on one GPU.
- Do all timing, RTF and the Rust engine on CPU.

## Consequences

- Reproducibility needs `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds and pinned versions.
- Check that cuDNN-LSTM and mamba-ssm kernels are deterministic: two runs with the same seed, compare weight checksums.
- Claim bitwise reproducibility of training only per hardware class; otherwise within seed variance.

## Open questions

- Determinism of cuDNN-LSTM and mamba-ssm kernels.
