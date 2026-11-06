# Build and hardware

How the project is built and which hardware the code must run on and be optimised for; the rules condensed in `.claude/CLAUDE.md` apply to all code.

## Build setup

- **Nix flake + direnv** (`use flake`) is the dev entry point on the owner's own machines. The flake provides devShells for `x86_64-linux` (NixOS workstation) and `aarch64-darwin` (MacBook). `flake.lock`, `Cargo.lock` and `uv.lock` are committed.
  - devShell `default`: the Rust toolchain from `rust-toolchain.toml` (rust-overlay), maturin, uv, Python 3.12, just, typos, nixfmt and the git-hooks shell hook.
  - devShell `docs`: `default` plus the TeX Live set and the tudcd-scr classes (via `TEXMFHOME`). LaTeX is only in this shell.
  - Env: `UV_PYTHON` (the Nix Python 3.12), `UV_PYTHON_DOWNLOADS=never`, `UV_PROJECT_ENVIRONMENT=.venv`.
  - `.envrc`: `use flake`, `dotenv_if_exists .env`, `watch_file rust-toolchain.toml pyproject.toml`.
- **Nix is not available on the TU Dresden HPC and, until verified, assume not on the company cluster either.** Every build, training and benchmark step must also work without Nix: `rustup` with the version pinned in `rust-toolchain.toml` and standalone `uv` with the pinned Python. The flake only provides tools; no build logic may exist only in the flake.
- **Python:** uv workspace with the members `training`, `inferencing/openvad` and `inferencing/openvad-rs`; `tool.uv.environments` limits resolution to darwin arm64 and linux x86_64. PyTorch for training and evaluation comes as wheels via uv (CUDA build on Linux, CPU/MPS on macOS), not from nixpkgs.
  - Pinned in `uv.lock`: torch `2.14.1` (PyPI, macOS arm64) and `2.14.1+cu130` (index `pytorch-cu130`, Linux x86_64).
  - On NixOS, manylinux wheels need the usual shared libs and the NVIDIA driver libs. The repo solves this in the devShell with `LD_LIBRARY_PATH` (on Linux: `stdenv.cc.cc.lib`, `zlib` and `/run/opengl-driver/lib`), **not** with nix-ld.
- **GPUs for training:** RTX 4090 (workstation, sm_89) and NVIDIA H100 94 GB (TU Dresden HPC Capella, sm_90, blocked until the owner decides otherwise, [ADR 0024](../project/adr/0024-training-data-licence-policy.md)). Torch and CUDA versions are pinned in `uv.lock`. All timing and RTF measurements run on CPU, never on GPU.
- **Rust:** Cargo workspace with `openvad-core` (`inferencing/openvad-core`, hand-written forward pass; no ONNX Runtime, tract, burn or candle) and `openvad-rs` (`inferencing/openvad-rs`, PyO3 + maturin). PyO3 is the only binding. Weights only as `safetensors` with documented tensor names.
  - `rust-toolchain.toml`: channel `1.99.1`, profile minimal, components rustfmt, clippy, rust-analyzer and rust-src.
  - `.cargo/config.toml` sets only `PYO3_BUILD_EXTENSION_MODULE = "1"`.
- **Release artifacts** are wheels only. They are built for the **generic target baseline** (`x86_64-unknown-linux-gnu` default, `aarch64-apple-darwin`) and select SIMD kernels at runtime (see [ISA matrix and kernel tiers](#isa-matrix-and-kernel-tiers)). One artifact must run correctly on every machine in [Machines](#machines).
- The main model and the engine run natively at 16 kHz; 8 kHz input is upsampled inside the engine, see [ADR 0009](../project/adr/0009-native-16khz-and-one-resampler.md).

## Machines

| Role | Machine | CPU (microarchitecture) | Cores / clock | L1d / L2 / L3 | Memory | Notes |
|---|---|---|---|---|---|---|
| Dev, CPU benchmarks, GPU training | Workstation (NixOS) | AMD Ryzen 9 7900X (Zen 4) | 12C/24T, 4.7 / 5.6 GHz | 32 KB / 1 MB per core / 64 MB (2 × 32) | DDR5-5200, 2 ch, ~83 GB/s | + RTX 4090. **Possible upgrade: Ryzen 9 9950X3D** (Zen 5, 16C/32T, 4.3 / 5.7 GHz, 48 KB / 1 MB / 96 + 32 MB on asymmetric CCDs, native 512-bit AVX-512) |
| Dev, aarch64 target | MacBook | Apple M3 Pro | 5–6 P + 6 E cores, P up to 4.05 GHz | 128 KB (P) / 16 MB shared per P-cluster / no L3 (SLC only) | LPDDR5-6400, ~150 GB/s | P/E heterogeneity; macOS allows no core pinning |
| Large CPU jobs, CPU benchmarks | TU Dresden HPC **Barnard** | 2 × Intel Xeon Platinum 8470 (Sapphire Rapids) | 2 × 52C, 2.0 / 3.8 GHz | 48 KB / 2 MB per core / 105 MB per socket | DDR5-4800, 8 ch per socket | 512 GiB per node, Slurm; no training data or anything rendered from it without an owner decision ([ADR 0024](../project/adr/0024-training-data-licence-policy.md)) |
| GPU training (blocked until the owner decides otherwise, [ADR 0024](../project/adr/0024-training-data-licence-policy.md)) | TU Dresden HPC **Capella** | 2 × AMD EPYC 9334 (Zen 4) | 2 × 32C, 2.7 / 3.9 GHz | 32 KB / 1 MB per core / 128 MB per socket | DDR5-4800, 12 ch per socket | 4 × H100 94 GB, 768 GiB per node, Slurm; CPUs mainly feed the GPUs |
| **Production** | Company cluster | Intel Xeon Gold 6230 (Cascade Lake) **or larger, TODO verify** | 20C/40T, 2.1 / 3.9 GHz | 32 KB / 1 MB per core / 27.5 MB | DDR4-2933, 6 ch | Runs the production voicebot, many concurrent calls per node; deployment mechanism TODO |

## Production audio

Facts about the production voicebot's audio path (anonymised):

- It negotiates one of three SIP codecs: G.722 (16 kHz), PCMA (G.711 A-law, 8 kHz) or PCMU (G.711 μ-law, 8 kHz). Other codecs (L16, Opus, G.729) cannot be negotiated over SIP.
- Production has more wideband than narrowband calls (owner, 2026-11-04). The exact split is unknown; see [Open questions](#open-questions).
- Everything inside the voicebot runs at 16 kHz. Narrowband (8 kHz) calls are upsampled to 16 kHz with libsamplerate `sinc_best` (with about 3 dB of headroom applied first) before the VAD and speech recognition. Narrowband audio has nothing above 4 kHz even after upsampling.
- The current VAD is Silero at 16 kHz, fed 512-sample blocks (32 ms).
- A browser voice channel uses WebRTC Opus, decoded to 16 kHz PCM (not verified in the code).

The engine design follows from this: it runs natively at 16 kHz and upsamples 8 kHz input internally, see [ADR 0009](../project/adr/0009-native-16khz-and-one-resampler.md).

## ISA matrix and kernel tiers

| Feature | Zen 4 (7900X, EPYC 9334) | Zen 5 (9950X3D) | Apple M3 Pro | Sapphire Rapids (8470) | Cascade Lake (6230) |
|---|---|---|---|---|---|
| ISA level | x86-64-v4 | x86-64-v4 | ARMv8.6-A | x86-64-v4 | x86-64-v4 |
| Physical SIMD width | 512-bit split into 2 × 256 | native 512 | NEON 128 | native 512 | native 512, with AVX-512 frequency licences |
| FP32 FMA per core and cycle | 32 FLOP | 64 FLOP | 32 FLOP | 64 FLOP | 64 FLOP |
| AVX2 + FMA | yes | yes | – | yes | yes |
| AVX-512 F / BW / DQ / VL | yes | yes | – | yes | yes |
| AVX-512 VNNI (INT8) | yes | yes | – | yes | yes |
| AVX-VNNI (VEX) | no | yes | – | yes | no |
| NEON dotprod / i8mm | – | – | yes / yes | – | – |
| BF16 | yes | yes | yes | yes | **no** |
| FP16 arithmetic | no | no | yes | yes | no |
| Matrix unit | no | no | Apple AMX (undocumented) | Intel AMX | no |
| rustc `target-cpu` name | `znver4` | `znver5` | `apple-m3` | `sapphirerapids` | `cascadelake` |

Several values come from microarchitecture documentation, not vendor product pages: verify per machine (see [Open questions](#open-questions)).

Rules:

- Kernel tiers, in implementation order:
  1. `scalar`: always present; **the reference that defines the canonical result**.
  2. `neon` (aarch64; part of the baseline on Apple): for development on the Mac.
  3. `avx2` (`avx2` + `fma`): portable x86 path.
  4. `avx512` (`avx512f` + `avx512bw` + `avx512vl`): production candidate for the company cluster.
  5. `avx512vnni` INT8: optional extension (quantisation). It is the production-relevant low-precision path because VNNI is the only AI acceleration Cascade Lake has.
- **No BF16, FP16 or AMX kernels.** The production CPU (Cascade Lake) lacks them and they conflict with the FP32 reproducibility goal. Ask the owner before adding any. BF16 may still be studied in PyTorch ([ADR 0016](../project/adr/0016-int8-ptq-optional-pruning-cut.md)).
- The common denominator of all x86 machines is x86-64-v4 + AVX512_VNNI, but kernels are selected at **runtime** (`is_x86_feature_detected!`, `std::arch::is_aarch64_feature_detected!`, `#[target_feature(enable = "...")]` on the kernel functions), never via `target-cpu` at compile time. **Never commit `-C target-cpu=native`**: a binary built on Zen 4/5 can crash with SIGILL on Cascade Lake (VBMI, AVX-VNNI, ...). Local experiments with the `target-cpu` names above are fine but must not leak into committed config, and reported numbers must state it.
- Provide a forced-tier override for tests and benchmarks (e.g. env var `OPENVAD_KERNEL=scalar|neon|avx2|avx512|auto`). Requesting an unsupported tier is an error, not a silent fallback. The selected tier is exposed via the API and logged, so every benchmark reports it.
- Cascade Lake lowers its clock under heavy 512-bit instructions, and the slowdown also affects other services on that core. Whether `avx2` or `avx512` is faster on the company cluster must be **measured under realistic full-node load**. Keep the tier preference configurable; never hard-code "widest wins".
- At VAD size (hidden size 64–128) per-call overhead dominates; wide SIMD only pays off with a **batch API across concurrent streams** (matrix-matrix instead of matrix-vector). Design `openvad-core` so that a batched step can be added later without breaking the single-stream API.
- The model size budget is **bytes, not parameters**: weights + state + activations should fit the 1 MB/core L2 (2 MB on the 8470). Details in [ADR 0008](../project/adr/0008-byte-size-budget.md).

## Determinism rules (FP32 engine)

- Goal: **bitwise identical outputs across all kernel tiers and all machines in [Machines](#machines)** for the Rust engine. PyTorch-to-Rust parity is checked within an absolute tolerance of `1e-6` (ADR 0002) against golden vectors (inputs, outputs, per-layer activations).
- **Canonical reduction order:** every dot product and reduction uses a fixed number of partial accumulators (e.g. 16) and a fixed final pairwise summation tree; tail handling is part of the definition. Scalar (16 accumulators), NEON (4 × 4 lanes), AVX2 (2 × 8) and AVX-512 (1 × 16) implement exactly the same order.
- Use `f32::mul_add` / FMA intrinsics **explicitly and in every tier**, including scalar, so all tiers round identically. Rust never contracts mul+add on its own; do not rely on either behaviour implicitly. No fast-math intrinsics (`fadd_fast` etc.), no reassociation.
- **Transcendentals** (exp, tanh, sigmoid, log in the frontend): never the platform libm (macOS libm and glibc differ). Either the pure-Rust `libm` crate evaluated per lane, or one shared polynomial approximation written with the same `mul_add` sequence in every tier (accuracy checked against `libm`). The same choice in all tiers.
- No FTZ/DAZ changes; MXCSR/FPCR stay at their defaults. If denormals become a performance problem (decaying recurrent state), measure first and ask the owner before changing anything.
- One stream = one thread. No thread-count-dependent reductions anywhere in the engine.
- The 2:1 decimator and the 2:1 interpolator in `openvad-core` follow the same rules: committed coefficients, the canonical reduction order, bitwise identical across tiers. They are covered by the tier-vs-scalar bitwise tests and the golden checksums (there is no PyTorch mirror of the resampler).
- Required tests: (a) every tier available on the current machine against `scalar`, bitwise; (b) committed golden checksums for a fixed test file, verified on every machine (and CI runner, if CI is added); (c) PyTorch parity per layer, offline and streaming.
- **INT8 (optional):** integer accumulation is exact, so INT8 outputs must also be bitwise identical across tiers. x86 `vpdpbusd` is u8 × s8, NEON `sdot` is s8 × s8: use the +128 activation offset with a precomputed correction term, document it, use the non-saturating instruction and prove the i32 accumulators cannot overflow.
- GPU training determinism is covered in [ADR 0015](../project/adr/0015-gpu-training-cpu-timing.md). Bitwise claims for training hold only per hardware class.

## Benchmarking rules

- The harness pushes the same fixed chunk into every adapter, each adapter buffers internally to its model's native call size, and outputs map onto the 10 ms grid by causal hold; file format in the [benchmark prediction spec](../project/spec/benchmark-predictions.md).
- Resampled benchmark audio (the 8 kHz test condition and the per-system upsampled inputs) is produced offline by a data preparation script with the one resampler via `openvad-rs`. Adapters read the prepared files and never depend on `openvad-rs`, which keeps ADR 0003 isolation.
- Every benchmark report records: machine, CPU model, `lscpu` (Linux) or `sysctl machdep.cpu.brand_string hw.optional` (macOS), kernel tier, rustc / torch / onnxruntime versions, commit hash, CPU governor and boost state, thread count, pinning.
- Linux: pin with `taskset` or cgroups, one thread per stream. Report single-thread RTF, per-frame time p50/p99, algorithmic latency and peak RAM (full metric list in the work plan). On a 9950X3D, pin to one CCD and say which (V-cache CCD or not). On the TU Dresden HPC use Slurm with `--exclusive`; CPU timing on Barnard, not on Capella.
- macOS: pinning is impossible, so set the thread QoS to user-interactive, use more repetitions and report the spread; note P vs E cores.
- Speed baselines: Silero VAD on ONNX Runtime with `intra_op_num_threads = 1`; PyTorch CPU numbers with `torch.set_num_threads(1)`.
- Production-like scenario on the company cluster: N concurrent streams on a fully loaded node, `avx2` vs `avx512` tiers (and `avx512vnni` if INT8 exists).
- Decision (owner, 2026-10-30): the reference CPU for the formal "RTF < 1 on one core" requirement is the workstation CPU at the time of the final benchmark (Ryzen 9 7900X for now). All headline RTF numbers come from that one CPU; a hardware change is dated in the reports. The company cluster is a production-like scenario under load, not a pass/fail criterion.
- The resampler's group delay counts towards algorithmic latency and is reported.

## Repo vs owner prompt

Where the repo differed from the owner prompt, and what was recorded here:

- Crate names: the prompt's `vad-core` and `vad-py` are `openvad-core` and `openvad-rs`.
- No cbindgen and no C API crate (`vad-capi`): owner decision. PyO3 (`openvad-rs`) is the only binding, and release artifacts are wheels only.
- LaTeX is only in the `docs` devShell, not in `default`.
- The repo uses `LD_LIBRARY_PATH`, not nix-ld.
- No CI: there is no `.github/` directory, and the owner decided against CI for now.
- No `target-cpu` or `RUSTFLAGS` exists in `.cargo/config.toml`, `flake.nix` or `justfile`; nothing was removed.
- Parity tolerance stays at `1e-6` (ADR 0002); the looser tolerance in the prompt was not adopted.

## Open questions

- [ ] Exact CPU model(s) on the company cluster and whether the nodes are homogeneous (`lscpu` incl. flags). Decides the tiers and whether Ice Lake features (VBMI, IFMA) exist.
- [ ] Workstation upgrade to Ryzen 9 9950X3D yes/no. If yes, record the date of the hardware change in the benchmark reports.
- [ ] Extend the prediction spec's `meta.json` (`docs/project/spec/benchmark-predictions.md`) with kernel tier, commit, CPU governor and boost state, thread count, pinning, the upsampling applied and the algorithmic latency when the first adapter lands.
- [ ] Exact production split between G.722 and PCMA/PCMU calls: wideband calls outnumber narrowband ones (owner, 2026-11-04), but the numbers are unknown. The gateway logs the line "Audio formats are negotiated: <codec> with ClockRate <n>" on every call, so a log query would show it.
- [ ] Verify the cache sizes and feature flags (VNNI/BF16/AMX) of every machine in [ISA matrix and kernel tiers](#isa-matrix-and-kernel-tiers) with `lscpu` / `sysctl` (several values come from microarchitecture documentation, not vendor product pages).
