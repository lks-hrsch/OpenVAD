# 0015: GPU training, CPU timing

Status: Accepted (2026-10-30)

## Context

The experiments need many runs × seeds ([0017](0017-experiment-budget.md)).
The production target and the Rust engine are CPU only.
Training GPUs are the RTX 4090 (workstation, sm_89) and the NVIDIA H100 (TU Dresden HPC Capella, sm_90, blocked for training until the owner decides otherwise, [0024](0024-training-data-licence-policy.md)); see [build-and-hardware.md](../../notes/build-and-hardware.md).

## Decision

- Train on GPU; run several small runs in parallel on one GPU.
- Do all timing and RTF on CPU, never on GPU. The Rust engine runs on CPU.
- Reference CPU: the formal "RTF < 1 on one core" requirement is measured on the workstation CPU at the time of the final benchmark (Ryzen 9 7900X for now). All headline RTF numbers come from that one CPU, and a hardware change is dated in the reports. The company cluster is a production-like scenario under load, not a pass/fail criterion.
- On the TU Dresden HPC, CPU timing runs on Barnard (Sapphire Rapids CPU nodes), not on Capella (GPU nodes).

## Consequences

- Reproducibility needs `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds and pinned versions.
- Bitwise reproducibility of training is claimed only per hardware class; otherwise within seed variance.
- Check that cuDNN-LSTM and mamba-ssm kernels are deterministic: two runs with the same seed, compare weight checksums.

## Open questions

- Are the cuDNN-LSTM and mamba-ssm kernels deterministic? Answered by the check above, only for backbones that are used.
- Which audio do timing runs on Barnard and the company cluster use, given decision 6 of [0024](0024-training-data-licence-policy.md)?
- Does the workstation move to a Ryzen 9 9950X3D? If so, the hardware change is dated in the benchmark reports.
