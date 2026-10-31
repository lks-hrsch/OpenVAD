# 0008: Model size budget in bytes

Status: Accepted (2026-10-31), replaces the earlier hard limit

## Context

- The literature range is ~300 params (AtomicVAD) to ~0.6 M (FireRedVAD, Mamba-FVAD); Silero ≈ 463 k.
- L3 does not constrain anything (≥ 27.5 MB on every x86 machine).
- L2 = 1 MB/core (2 MB on the Xeon 8470) is the meaningful limit: ≈ 250 k params in FP32, ≈ 500 k in BF16, ≈ 1 M in INT8 (minus headroom for state and activations).
- For a single stream, memory bandwidth is not the bottleneck. The limit matters for p99 per-frame latency and when one core serves many calls.

## Decision

The byte budget is a target, not a hard limit: if possible, weights + state + activations fit 1 MB L2 per core (see [build-and-hardware.md](../../notes/build-and-hardware.md)).
The budget is bytes, not parameters, which couples it to the number format.

## Consequences

- A larger model only fits with a smaller number format.
- Measure it: sweep model width, plot per-frame latency against weight bytes, mark L1/L2.
