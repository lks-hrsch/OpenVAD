# 0008: Model size budget in bytes

Status: Accepted (2026-10-31)

## Context

- The literature range is about 300 parameters (AtomicVAD) to about 0.6 M (FireRedVAD, Mamba-FVAD); Silero has about 463 k.
- L3 does not constrain anything: it is at least 27.5 MB on every x86 machine.
- L2 is 1 MB per core (2 MB on the Xeon Platinum 8470) and is the meaningful limit. In 1 MB that is about 250 k parameters in FP32, about 500 k in BF16 (a PyTorch study only, [0016](0016-int8-ptq-optional-pruning-cut.md)) and about 1 M in INT8, minus headroom for state and activations.
- For a single stream, memory bandwidth is not the bottleneck. The limit matters for p99 per-frame latency and when one core serves many calls.

Machine details are in [build-and-hardware.md](../../notes/build-and-hardware.md).

## Decision

The byte budget is a target, not a hard limit: if possible, weights + state + activations fit 1 MB of L2 per core (2 MB on the Xeon Platinum 8470).
The budget is bytes, not parameters, which couples it to the number format.

## Consequences

- A larger model only fits with a smaller number format.
- State size depends on the backbone ([0012](0012-classifier-backbone.md) gives the owner's estimate).
- The target is checked by sweeping model width and plotting per-frame latency against weight bytes, with L1 and L2 marked.

## Open questions

- Is there a concrete byte number for the final model, or does the target stay "if possible" until the width sweep?
