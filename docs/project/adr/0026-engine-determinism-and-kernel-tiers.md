# 0026: Engine determinism and kernel tiers

Status: Accepted (2026-11-04)

## Context

The engine is hand-written ([0001](0001-no-onnx-in-engine.md)) and the wheel for each architecture must run correctly on every machine of that architecture in [build-and-hardware.md](../../notes/build-and-hardware.md): Zen 4, Zen 5 (possible upgrade), Apple M3 Pro, Sapphire Rapids and Cascade Lake.
Wide SIMD pays off at VAD size (hidden size 64 to 128) only with a batch API across concurrent streams, because per-call overhead dominates.
Cascade Lake lowers its clock under heavy 512-bit instructions, and the slowdown also affects other services on that core.

## Decision

### Determinism

Engine output is bitwise identical across all kernel tiers and all machines.

- Canonical reduction order: every dot product and reduction uses a fixed number of partial accumulators and a fixed final pairwise summation tree; tail handling is part of the definition. Scalar, NEON, AVX2 and AVX-512 implement exactly the same order.
- `f32::mul_add` (FMA) is used explicitly in every tier, scalar included. No fast-math, no reassociation.
- Transcendentals never use the platform libm. All tiers use the same choice: the pure-Rust `libm` crate per lane, or one shared polynomial with the same `mul_add` sequence.
- No FTZ/DAZ changes; MXCSR and FPCR stay at their defaults. If denormals become a performance problem, measure first and ask the owner.
- One stream runs on one thread. No reduction depends on the thread count.
- The 2:1 decimator and interpolator follow the same rules ([0009](0009-native-16khz-and-one-resampler.md)).
- INT8 outputs are also bitwise identical across tiers, because integer accumulation is exact. The notes require the +128 activation offset with a precomputed correction term, the non-saturating instruction, and a proof that the i32 accumulators cannot overflow.

Required tests:

1. Every tier available on the machine against `scalar`, bitwise.
2. Committed golden checksums for a fixed test file, verified on every machine (and CI runner, if CI is added).
3. PyTorch parity per layer at `1e-6`, offline and streaming ([0002](0002-safetensors-weights.md)). The resampler has no PyTorch mirror, so tests 1 and 2 cover it.

### Kernel tiers

In implementation order:

1. `scalar`: always present; the reference that defines the canonical result.
2. `neon` (aarch64).
3. `avx2` (`avx2` + `fma`).
4. `avx512` (`avx512f` + `avx512bw` + `avx512vl`).
5. `avx512vnni`: optional INT8 extension ([0016](0016-int8-ptq-optional-pruning-cut.md)).

- Tiers are selected at runtime by feature detection, never by `target-cpu` at compile time. Release artifacts (wheels) target the generic baselines.
- `target-cpu` is never committed. Local `target-cpu` experiments are fine, but must be stated with any reported number.
- `OPENVAD_KERNEL` forces a tier (for example `scalar`, `neon`, `avx2`, `avx512` or `auto`). An unsupported tier is an error, not a fallback. The selected tier is exposed via the API and logged, so every benchmark reports it.
- "Widest wins" is never hard-coded. `avx2` against `avx512` on the company cluster is measured under realistic full-node load, and the tier preference stays configurable.
- No BF16, FP16 or AMX kernels without asking the owner.
- The single-stream API stays extensible to a batched multi-stream step.

## Consequences

- The tiers, `OPENVAD_KERNEL`, golden checksums, the 2:1 resampler and the streaming API do not exist yet; each is added with its first user.

## Open questions

- Pure-Rust `libm` per lane, or one shared polynomial for the transcendentals? The notes allow either; accuracy of a polynomial is checked against `libm`.
- Which CPU model(s) does the company cluster have, and are the nodes homogeneous? This decides which tiers matter.
- Does bitwise identity apply to W8A32 ([0016](0016-int8-ptq-optional-pruning-cut.md))?

## References

- [build-and-hardware.md](../../notes/build-and-hardware.md): ISA matrix, kernel tiers and determinism rules.
