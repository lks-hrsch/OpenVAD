# VADigo

A fully reproducible, real-time voice activity detector (VAD) for noisy telephone audio, built as a Master's thesis at TU Dresden: open architecture, open training code, public training data only, open evaluation.
The goal is to show how far a fully open VAD gets, not to beat Silero VAD.

## Project context

TU Dresden Master Informatik thesis, 22 weeks.
The preparation phase (core literature, design discussion, frontend and Mamba literature) is done.
Its results live in the ADRs (`docs/project/adr/`) and `docs/shared/acronyms.tex`; read them before design work.

## Research questions

- RQ1: training data.
- RQ2: model design and efficiency.
- RQ3: evaluation and generalisation.

## Pipeline

- Data and training are Python and PyTorch (`training/`).
- The canonical PyTorch model lives in `inferencing/vadigo`. Training imports it; never duplicate model code in training.
- Inference is a hand-written forward pass in Rust (`inferencing/vadigo-core`). The engine never uses ONNX Runtime, tract, burn or candle; `just guard-engine` enforces this ([ADR 0001](../docs/project/adr/0001-no-onnx-in-engine.md)).
- Weights move from PyTorch to Rust only as safetensors with documented tensor names, BatchNorm folded ([ADR 0002](../docs/project/adr/0002-safetensors-weights.md)).
- Python bindings use PyO3 and maturin (`inferencing/vadigo-rs`).

## Parity

Parity is first-class. Parity tests must pass before any change to the model or the engine is done. See `inferencing/CLAUDE.md`.

## Evaluation

- Every system except pyannote runs causally through one common streaming interface. "Identical chunks" means the harness pushes the same fixed chunk into every adapter and each adapter buffers to its model's native call size. Outputs map onto the 10 ms grid by causal hold (the most recent probability already emitted, never a future one).
- Two test conditions: 16 kHz, and 8 kHz derived from the same test set with the one VADigo resampler, so bandwidth is the only variable.
- pyannote is a labelled non-causal upper bound.
- ONNX Runtime is allowed in `benchmarking/` only.
- See `benchmarking/CLAUDE.md` and the [benchmark prediction spec](../docs/project/spec/benchmark-predictions.md).

## Licensing

- No GPL or unlicensed third-party code in the release.
- Never commit data or weights. Datasets and third-party weights are never redistributed; scripts fetch them.
- The TEN VAD adapter stays isolated (Agora non-compete, [ADR 0003](../docs/project/adr/0003-isolated-benchmark-adapters.md)).

## Environment

- Data paths come from `VADIGO_DATA_DIR` (default `./data`).
- Machines:
  - NixOS x86_64 with CUDA, Ryzen 9 7900X + RTX 4090 (main dev),
  - macOS aarch64 M3 Pro,
  - TU Dresden HPC,
  - and a company cluster.
- The repo must work without Nix (uv, cargo). LaTeX is ok to just work on Nix. Nix and direnv are a convenience layer. Every build, training and benchmark step works with `rustup` (`rust-toolchain.toml`) and standalone uv; no build logic lives only in the flake.

## Build and hardware rules

Details and machine tables: `docs/notes/build-and-hardware.md`.

None of the kernel tiers, `VADIGO_KERNEL`, golden checksums, the 2:1 resampler or the streaming API exist yet; add each with its first user.

- PyTorch comes only as uv wheels pinned in `uv.lock`, never from nixpkgs.
- Release artifacts (wheels) target the generic baselines and select SIMD kernels at runtime.
- Never commit `target-cpu` (native or named). Local `target-cpu` experiments must be stated with any reported number.
- Kernel tiers: `scalar` (reference, defines the canonical result), `neon`, `avx2`, `avx512`, optional `avx512vnni` INT8.
- No BF16, FP16 or AMX kernels without asking the owner (BF16 only as a PyTorch study).
- `VADIGO_KERNEL` forces a tier. An unsupported tier is an error, not a fallback. The selected tier is exposed and logged.
- Never hard-code "widest wins". `avx2` vs `avx512` on the company cluster is measured under full-node load.
- Keep the single-stream API extensible to a batched multi-stream step.
- The size budget is bytes: if possible weights + state + activations fit 1 MB L2 per core.
- Determinism: engine output is bitwise identical across tiers and machines.
  - Canonical reduction order: fixed accumulators and summation tree in every tier.
  - Explicit `mul_add`/FMA in every tier; no fast-math or reassociation.
  - No platform libm for transcendentals.
  - No FTZ/DAZ changes without asking the owner.
- Required tests: every tier vs `scalar` bitwise, golden checksums, and PyTorch parity per layer at `1e-6` (ADR 0002), offline and streaming.
- Timing and RTF are measured on CPU only, never GPU. On the TU Dresden HPC, CPU timing runs on Barnard (Sapphire Rapids CPU nodes), not Capella (GPU nodes); headline RTF comes from the reference CPU (the workstation, see the notes).
- Every benchmark report records machine, CPU, flags, tier, versions, commit, governor/boost, threads and pinning (fields in the notes).
- The engine runs one stream on one thread; benchmarks pin one thread per stream on Linux.
- The engine runs natively at 16 kHz, upsamples 8 kHz internally with the one finite impulse response resampler, rejects other rates, takes the rate at stream creation and accepts any block length. Training uses that resampler via `vadigo-rs`; benchmark audio is prepared offline with it, and adapters never depend on `vadigo-rs`.
- Speed baselines: Silero on ONNX Runtime with `intra_op_num_threads = 1`; PyTorch with `torch.set_num_threads(1)`.

## Working rules

- KISS and YAGNI: build nothing without a present user; defer it and list it.
- Conventional Commits.
- Never use the em dash character.

## Language

- Everything written to the repo is English: code, identifiers, comments, docstrings, commit messages, issues, README, docs, notes, LaTeX thesis, paper, slides and generated text. This holds even when the owner writes in German.
- Reply in chat in english always.
- Use British spelling (normalisation, quantisation), matching the existing docs.

## Design records

- Design decisions are ADRs, one decision per record. Before writing or changing one, read `docs/project/adr/README.md` for the status values and the rule that only the owner changes a status.
- `docs/shared/acronyms.tex` is the single source of truth for acronyms. Every new acronym gets `short`, `long` and `definition` there.

## Repo map

- `inferencing/vadigo`: canonical PyTorch model.
- `inferencing/vadigo-core`: Rust inference engine.
- `inferencing/vadigo-rs`: PyO3 bindings.
- `inferencing/fixtures`: parity fixtures.
- `training/`: the `vadigo_train` package (smoke fixture and docs exports so far).
- `benchmarking/`: conventions only (`CLAUDE.md`), no adapters yet.
- `docs/`: thesis, ADRs and specs (`docs/project/`); see `docs/README.md`.
- `docs/notes`: build and hardware notes.
- `data/`: git-ignored data and weights, only `data/README.md` is committed.
- `justfile`, `flake.nix`, `Cargo.toml`, `pyproject.toml`, `rust-toolchain.toml`: build and environment.

## `just` recipes

- `just setup`: fetch submodules and install the Python workspace.
- `just fmt`: format all code.
- `just check`: lint and type-check everything, plus fixtures, the engine guard and tests.
- `just test`: run all tests.
- `just fixtures`: regenerate the parity fixture.
- `just check-fixtures`: fail if the committed fixture is stale.
- `just lint-py`: lint, format-check and type-check Python.
- `just test-rs`: run Rust tests.
- `just lint-rs`: check Rust formatting and clippy.
- `just guard-engine`: fail if any Rust crate pulls in a third-party inference runtime.
- `just thesis`: build the thesis PDF into `docs/thesis/build`.
- `just watch <doc>`: rebuild a document on every save.
- `just docs-generated`: regenerate numbers and figures under `docs/`.
- `just skills`: restore the Claude skills pinned in `skills-lock.json` (needs Node).
