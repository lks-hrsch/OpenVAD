# Repository setup: design

Status: draft for review (2026-10-30).

## Intent

Give VADigo a working skeleton for the thesis: toolchains, workspaces, docs build and Claude configuration, with just enough placeholder code that every build and check passes.
The repo must be fully usable without Nix (uv + cargo + TeX Live); Nix and direnv are a convenience layer.
Guiding rule from Lukas: **KISS and YAGNI**, now and during later programming. Anything without a present user is deferred, not stubbed.

## Non-goals

- No model, data pipeline, VAD adapter, config loader, run-metadata writer or fetch CLI. They are written when the code that needs them is written.
- No datasets or model weights downloaded.
- No CI, no push, no remote.
- No licence (code or docs) for now. `README.md` keeps "To be decided".

## Decisions

| Topic | Decision | Source |
| --- | --- | --- |
| Licence | none for now, no `license` fields | Lukas |
| Python | 3.12 everywhere (workspace and adapters) | Lukas |
| Training configs | one YAML file per run, typed dataclasses, strict loader (later) | Lukas |
| Training | plain PyTorch loop, no Lightning | Lukas |
| Tracking | TensorBoard (`SummaryWriter`) for live curves, plus `metrics.jsonl`, `run.json`, `config.yaml` per run (later) | Lukas + my call |
| nixpkgs | `nixos-26.05`, pinned by `flake.lock` | Lukas |
| Tables | polars, Parquet wherever possible, each Parquet file described by a sibling `.md` | Lukas |
| Large files | Git LFS for committed Parquet and own released weights; third-party weights and audio never committed | Lukas + my call |
| Audio | fetched by a CLI script into `$VADIGO_DATA_DIR` (gitignored `./data` or a NAS mount) | Lukas |
| Aufgabenstellung | never in the repo; placeholder page in the thesis | Lukas |
| tudcd-scr | git submodule at `docs/tudcd-scr`, pinned to `c436bf5` (LPPL 1.3c) | my call, Lukas asked |
| Author | Lukas Hirsch, no ORCID yet | Lukas |
| Approach | B: Nix first, every step verified in `nix develop`, one non-Nix check at the end | Lukas |

## Approaches considered

- **A. Plain first, Nix last:** guards the non-Nix promise at every step, but slower to set up.
- **B. Nix first (chosen):** faster; risk is a hidden Nix-only dependency. Mitigation: `just` recipes call plain `uv`/`cargo`/`latexmk` only, and a final check runs in an Ubuntu container with uv + rustup and no Nix.
- Apptainer image as the non-Nix path: dropped, a third environment to maintain.

## Scope now

```
VADigo/
├── .claude/{CLAUDE.md,settings.json}
├── .envrc, .env.example, flake.nix, flake.lock
├── Cargo.toml, Cargo.lock, rust-toolchain.toml
├── pyproject.toml, uv.lock, .python-version
├── justfile, README.md, .gitignore, .gitmodules, .editorconfig
├── data/README.md                     # layout, VADIGO_DATA_DIR, NAS use
├── training/                          # uv member `vadigo-train`; data code later in `vadigo_train.data`; exports the parity fixture
├── inferencing/                       # folders named after the package they hold
│   ├── CLAUDE.md                      # parity rules, tolerances, fixture regeneration
│   ├── vadigo/                        # uv member `vadigo` (canonical PyTorch model lives here later)
│   ├── vadigo-core/                   # crate; deps: safetensors only
│   ├── vadigo-rs/                     # crate + uv member `vadigo-rs` (PyO3 + maturin, abi3-py312)
│   └── fixtures/smoke/                # golden vectors, plain git, < 1 MB
├── benchmarking/CLAUDE.md             # isolation, fairness and licence rules
└── docs/
    ├── CLAUDE.md, README.md, .envrc
    ├── tudcd-scr/                     # submodule
    ├── shared/{references.bib,acronyms.tex,macros.tex,generated/numbers.tex}
    ├── images/generated/              # one generated figure
    ├── thesis/{main.tex,latexmkrc,frontmatter/,chapters/,appendix/}
    └── project/{design/,adr/,spec/}
```

### Rust

- Virtual workspace: `resolver = "3"`, edition 2024, `[workspace.package]`, `[workspace.dependencies]`, `[workspace.lints]` (clippy pedantic warn, `unsafe_code = "deny"`).
- `vadigo-core` depends on `safetensors` only. Plain `Vec<f32>` and slices, no ndarray: the forward pass is hand-written, so an array library adds a dependency without removing code.
- Toolchain pinned in `rust-toolchain.toml`, read by rust-overlay in Nix and by rustup without Nix.

### Python

- Virtual uv workspace root (no `[project]`); members `training`, `inferencing/vadigo`, `inferencing/vadigo-rs`. Inter-member deps use `{ workspace = true }`.
- `src/` layout, ruff (line length 100, rules `E,F,I,UP,B,NPY`), basedpyright (PyPI wheel bundles Node, so it runs on the HPC without extra installs; faster than mypy on torch), pytest. Dev tools in root `[dependency-groups] dev`.
- torch: PyPI wheel on macOS (CPU/MPS), CUDA wheel on Linux via `[[tool.uv.index]]` (explicit) + `[tool.uv.sources]` with a `sys_platform == 'linux'` marker. CPU-only Linux takes the CUDA wheel too (one download, no second index).
- torch pinned to `==2.14.1` on all platforms, Linux index `cu130` (`https://download.pytorch.org/whl/cu130`). Lukas asked for CUDA 13.0 or newer with the best driver compatibility: `cu130` needs the oldest CUDA 13 driver. Checked 2026-10-30: torch 2.15.0 exists only as `cu132` (and newer drivers), `cu130` ends at 2.14.1. Moving to `cu132` later is a one-line change plus `uv lock`.
- `tool.uv.environments` limits the lock to `darwin/arm64` and `linux/x86_64`, matching the Nix systems. (my call)
- Bindings: `vadigo_rs` = native `_native` + thin wrapper + `.pyi`. `tool.uv.cache-keys` lists the Rust sources so `uv sync` rebuilds; `maturin develop --uv` documented for fast iteration.
- Data paths come from `VADIGO_DATA_DIR` (default `./data`).

### Parity smoke test

`vadigo-train` writes `inferencing/fixtures/smoke/linear.safetensors` for a `Linear(4, 2)` with hard-coded small weights and input (no RNG): tensors `weight`, `bias`, `input`, `expected`.
`expected` is computed in float64 and cast to float32, so regeneration gives the same bytes on every platform; tests compare with the tolerance below. (my call)
Checked against `expected` with abs tolerance `1e-6` by:

- `cargo test -p vadigo-core` (Rust forward pass),
- pytest in `inferencing/vadigo` (torch forward pass),
- pytest in `inferencing/vadigo-rs` (`vadigo_rs` through PyO3).

`just fixtures` regenerates it; output must be byte-identical across reruns.

### Docs

- Thesis only, `tudcdbook` (class name on one line), LuaLaTeX + biblatex/biber, babel `english`, KOMA `scrlayer-scrpage` (no fancyhdr), Noto fonts via `OSFONTDIR` in Nix.
- Class options taken only from the pinned `doc/handbook.tex`, `doc/demo-*.tex` and `source/*.dtx`.
- Chapter files with one-line TODOs; frontmatter placeholders for title page fields, abstract, Kurzfassung, declaration, AI-use statement and the Aufgabenstellung, all marked "to be confirmed with Prüfungsamt/chair".
- One number from `shared/generated/numbers.tex` and one TikZ figure from `images/generated/`, both written by a tiny `just docs-generated` script (no matplotlib) to prove the results-to-docs path.
- One sentence per line. Output in `docs/thesis/build/` (gitignored).
- Non-Nix: TeX Live plus `l3build install` from the submodule, documented in `docs/README.md`.

### Specs and ADRs (Markdown, short)

- `adr/0001-no-onnx-in-engine.md`, `adr/0002-safetensors-weights.md`, `adr/0003-isolated-benchmark-adapters.md`.
- `spec/benchmark-predictions.md`: input = manifest + run config (sample rate, chunk size); output = one Parquet per file with 10 ms frame probabilities + `meta.json` (system, version, weights hash, runtime, threads, CPU model).

### Nix

- Systems `aarch64-darwin`, `x86_64-linux` (no `aarch64-linux`: no machine uses it).
- Inputs: nixpkgs `nixos-26.05`, rust-overlay, git-hooks.nix. `inputs.self.submodules = true` so the flake sees tudcd-scr.
- `devShells.default`: Rust toolchain (+ rust-analyzer, rust-src), maturin, uv, python312, just, git-lfs, typos, nixfmt, hook installer. Sets `UV_PYTHON`, `UV_PYTHON_DOWNLOADS=never`, `UV_PROJECT_ENVIRONMENT=.venv`; on Linux, `LD_LIBRARY_PATH` with `stdenv.cc.cc.lib`, `zlib`, `/run/opengl-driver/lib`.
- `devShells.docs`: default + TeX Live set, latexmk, biber, l3build, chktex, the tudcd-scr TEXMF derivation, Noto fonts.
- Hooks: ruff, ruff-format, rustfmt, nixfmt, typos, end-of-file, trailing whitespace, check-added-large-files (500 kB, LFS files exempt), `uv lock --check`.
- `formatter` = nixfmt, `checks` = hooks.
- `.envrc`: `use flake`, `dotenv_if_exists .env.local`, `watch_file rust-toolchain.toml pyproject.toml`. `docs/.envrc`: `use flake ..#docs`.

### just

`setup` (`git submodule update --init`, `git lfs install --local`, `uv sync`), `check` (ruff, basedpyright, clippy, tests, engine-deps guard: fails if `cargo tree -p vadigo-core` contains ort/tract/burn/candle), `test`, `fmt`, `fixtures`, `thesis`, `watch thesis`, `docs-generated`.

### Claude

- `.claude/CLAUDE.md` (< 150 lines): purpose, RQ1 training data, RQ2 model design/efficiency, RQ3 evaluation/generalisation, non-negotiables, repo map, `just` commands, KISS/YAGNI, never commit data/weights, no ONNX/tract/burn/candle in the engine, parity must pass before model/engine changes are done, Conventional Commits, English.
- Subdirectory `CLAUDE.md` in `docs/`, `inferencing/`, `benchmarking/`. No `.claude/rules/`: ruff and rustfmt already enforce style, so one mechanism per topic.
- `settings.json`: allow `just`, `uv run`, `uv sync`, `cargo build/test/clippy/fmt`, `nix build`, `nix flake check`, `latexmk`, `git status/diff/log`; deny reading `.env.local`, writing `data/datasets/**`, `data/models/**`.

## Deferred (proposed; strike any you want now)

| Item | Added when |
| --- | --- |
| `vadigo_train.data` subpackage, `benchmarking/vadigo-bench` member, polars | first data or scoring code |
| `inferencing/vadigo-cli`, criterion `benches/` | engine has a real forward pass |
| `benchmarking/adapters/{webrtc,TEMPLATE}`, `just bench-adapter` | first reference system is run |
| Git LFS patterns in `.gitattributes` | first committed Parquet or released weights |
| `VADIGO_CACHE_DIR` | caching code exists |
| Seed/determinism helpers, run-metadata writer, dirty-tree guard, config loader | training loop |
| `docs/paper/` (ACL), `docs/defence/`, `docs/talks/`, `docs/project/cards/` | when that document is started |
| `nix build .#thesis` package | needed only by CI, which is dropped |
| `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`, git-cliff | first tagged release (v0.1) |
| `typos.toml`, `.gitattributes` | first false positive / first LFS file |

## Commits on `chore/repo-setup`

1. `build(nix): add flake, dev shells, direnv and pre-commit hooks`
2. `chore: add repo basics, data layout and justfile`
3. `build(rust): add cargo workspace with vadigo-core`
4. `build(python): add uv workspace with vadigo and vadigo-train`
5. `feat(bindings): add vadigo-rs and parity smoke test`
6. `docs: add thesis skeleton with tudcd-scr`
7. `docs(project): add ADRs and benchmark prediction spec`
8. `chore(claude): add Claude configuration and development setup`

Each commit is verified inside Nix, with output shown: `nix flake check`, both shells via direnv, `uv sync --locked`, `cargo test`, `uv run pytest`, `python -c "import vadigo_rs"`, `just thesis` (no undefined references, no tudcd warnings), hooks on all files.
Last: an Ubuntu Docker container (`--platform linux/amd64`, the architecture of the HPC and the workstation; slower under emulation on the Mac, run once) with uv + rustup only runs `uv sync --locked`, `cargo test --locked`, `uv run pytest`.

## Verified at setup time (not assumed)

uv virtual workspace root and `cache-keys` syntax; `inputs.self.submodules` on the pinned Nix; PyO3 and maturin versions; TeX Live version in `nixos-26.05`; `luaotfload-tool --find="Noto Sans"`; tudcd-scr options.

## Left to the builder

Exact chapter list, placeholder wording, ruff rule details beyond the set above, recipe names beyond those listed.
