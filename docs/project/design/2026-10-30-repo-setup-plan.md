# Repository Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A OpenVAD skeleton where the Nix shells, the uv and cargo workspaces, the PyO3 bindings, the parity smoke test and the thesis build all work and pass.

**Architecture:** Nix first (approach B): every task is verified inside `nix develop`, and `just` recipes call only plain `uv`/`cargo`/`latexmk`, so a final Ubuntu container without Nix proves the non-Nix path.
One fixture (`Linear(4, 2)` in safetensors) is checked by torch, by the Rust engine and by the Python bindings.

**Tech Stack:** Nix flakes (nixos-26.05, rust-overlay, git-hooks.nix), direnv, just, uv, Python 3.12, torch 2.14.1, numpy, safetensors, Rust 2024 + safetensors crate, PyO3 + maturin, LuaLaTeX + biber + tudcd-scr.

**Spec:** `docs/project/design/2026-10-30-repo-setup.md`

## Global Constraints

- KISS and YAGNI: no file, dependency, option or recipe without a present user. Deferred items stay deferred (spec, "Deferred" table).
- No licence files and no `license` fields anywhere.
- No em dash character in any file or commit message.
- English only. Conventional Commits. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `chore/repo-setup`. No push, no remote.
- Python: `requires-python = ">=3.12,<3.13"` in every member, `.python-version` = `3.12`.
- torch `==2.14.1`; Linux from index `https://download.pytorch.org/whl/cu130` (explicit), macOS from PyPI.
- uv lock environments: `sys_platform == 'darwin' and platform_machine == 'arm64'`, `sys_platform == 'linux' and platform_machine == 'x86_64'`.
- Nix systems: `aarch64-darwin`, `x86_64-linux`.
- Rust: edition 2024, `resolver = "3"`. `openvad-core` depends on `safetensors` only. No `ort`, `tract`, `burn`, `candle` anywhere in the Rust workspace.
- Fixture: `inferencing/fixtures/smoke/linear.safetensors`, plain git, < 1 MB, abs tolerance `1e-6`.
- `.tex` files: one sentence per line. Only tudcd-scr options documented in the pinned `doc/handbook.tex`, `doc/demo-*.tex`, `source/*.dtx`.
- Author: Lukas Hirsch.
- Deviation from spec: Git LFS is deferred with its patterns, so `git-lfs` is not in the shell and `just setup` has no `git lfs install`.

## Review Focus

1. Fresh clone without `--recursive`: `just thesis` must stop with "run `git submodule update --init`", not a LaTeX "file not found". Test in Task 6.
2. Editing Rust code: the next `uv sync` must rebuild `openvad-rs`, not keep the stale extension. Test in Task 5.
3. Regenerating the fixture on any machine must not change its bytes. Test in Task 3, and `just check` runs it.
4. Running pytest from a member directory instead of the root must still find the fixture. Test in Tasks 3 and 5.
5. Committing a file over 500 kB (audio, weights) must be blocked. Test in Task 2.

---

### Task 0: Commit design and plan

- [ ] **Step 1: Commit**

```bash
git add docs/project/design/
git commit -m "docs(project): add repository setup design and plan"
```

### Task 1: Nix flake, dev shell, direnv, hooks

**Files:**
- Create: `flake.nix`, `flake.lock`, `rust-toolchain.toml`, `.envrc`

**Interfaces:**
- Produces: `devShells.default` (Rust toolchain from `rust-toolchain.toml`, maturin, uv, python312, just, typos, nixfmt); env `UV_PYTHON` (Nix python3.12), `UV_PYTHON_DOWNLOADS=never`, `UV_PROJECT_ENVIRONMENT=.venv`; on Linux `LD_LIBRARY_PATH` with `stdenv.cc.cc.lib`, `zlib`, `/run/opengl-driver/lib`. Keep these env vars in one `let` attrset so Task 6 can reuse them (mkShell `inputsFrom` does not carry env vars).
- Produces: `formatter` = nixfmt, `checks.<system>.pre-commit`.

- [ ] **Step 1: Look up and record versions:** latest stable Rust release (pin exact, e.g. `channel = "1.NN.0"`), the git-hooks.nix hook name for nixfmt in the locked revision. Components: `rustfmt`, `clippy`, `rust-analyzer`, `rust-src`.
- [ ] **Step 2: Write `flake.nix`:** plain `nixpkgs.lib.genAttrs` over the two systems, no flake-parts, no flake-utils. `inputs.self.submodules = true;` (needed by Task 6; verify the installed Nix accepts it). Hooks: ruff, ruff-format, rustfmt (using the rust-overlay toolchain), nixfmt, typos, end-of-file-fixer, trim-trailing-whitespace, check-added-large-files `--maxkb=500`, and a local `uv-lock` hook (`uv lock --check`, files `(^|/)(pyproject\.toml|uv\.lock)$`, `pass_filenames = false`). Call `git-hooks.lib.${system}.run` twice: all hooks for the shell hook, all but `uv-lock` for `checks` (the Nix sandbox has no network).
- [ ] **Step 3: Write `.envrc`:** `use flake`, `dotenv_if_exists .env.local`, `watch_file rust-toolchain.toml pyproject.toml`.
- [ ] **Step 4: Verify**

Run: `nix flake lock && nix flake check && direnv allow && nix develop -c bash -c 'cargo --version && uv --version && python --version && just --version && echo $UV_PYTHON'`
Expected: check passes; Python `3.12.x`; cargo shows the pinned version; `UV_PYTHON` is a `/nix/store/...python3` path.

- [ ] **Step 5: Commit** `build(nix): add flake, dev shell, direnv and pre-commit hooks`

### Task 2: Repo basics and justfile

**Files:**
- Create: `.gitignore`, `.editorconfig`, `.env.example`, `data/README.md`, `justfile`

**Interfaces:**
- Produces: recipes `default` (`just --list`), `setup` (`git submodule update --init`, `uv sync --all-packages`), `fmt`, `check`, `test`. Later tasks add their recipes to `check`/`test`.

- [ ] **Step 1: Write `.gitignore`:** `.venv/`, `target/`, `.direnv/`, `__pycache__/`, `*.egg-info/`, `*.so`, `.env.local`, `/data/*` with `!/data/README.md`, `docs/**/build/`, `docs/thesis/frontmatter/aufgabenstellung.pdf`, `.pre-commit-config.yaml` (generated by git-hooks.nix).
- [ ] **Step 2: Write `.env.example`** with `OPENVAD_DATA_DIR=./data` and a comment that a NAS mount path works too; `data/README.md` with the layout (`datasets/raw`, `datasets/processed`, `models/checkpoints`, `cache`), "never committed", "audio is fetched by a script later".
- [ ] **Step 3: Test the large-file hook**

Run: `head -c 600000 /dev/urandom > big.bin && git add big.bin && git commit -m test; echo "exit=$?"; git reset -q big.bin; rm big.bin`
Expected: commit refused by `check-added-large-files`, `exit=1`.

- [ ] **Step 4: Verify** `just --list` shows the recipes; `pre-commit run --all-files` passes.
- [ ] **Step 5: Commit** `chore: add repo basics, data layout and justfile`

### Task 3: uv workspace, `openvad`, `openvad-train`, fixture

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.python-version`
- Create: `training/pyproject.toml`, `training/src/openvad_train/__init__.py`, `training/src/openvad_train/export_smoke.py`, `training/tests/test_export_smoke.py`
- Create: `inferencing/openvad/pyproject.toml`, `inferencing/openvad/src/openvad/__init__.py`, `inferencing/openvad/tests/test_load_linear.py`
- Create: `inferencing/fixtures/smoke/linear.safetensors` (generated)
- Modify: `justfile`

**Interfaces:**
- Produces: `openvad_train.export_smoke.main(argv: list[str] | None = None) -> None`, CLI `python -m openvad_train.export_smoke <out.safetensors>`; module constants `WEIGHT` (2x4), `BIAS` (2), `INPUT` (3x4) as nested lists.
- Produces: fixture tensors, all float32: `weight` [2, 4], `bias` [2], `input` [3, 4], `expected` [3, 2].
- Produces: `openvad.load_linear(path: str | os.PathLike[str]) -> torch.nn.Linear` (reads `weight`, `bias` with `safetensors.torch.load_file`).
- Root: virtual workspace (no `[project]`), members `training`, `inferencing/openvad` (Task 5 adds `inferencing/openvad-rs`); torch source and index as in Global Constraints; `[dependency-groups] dev` = pytest, ruff, basedpyright; `[tool.ruff]` line-length 100, target `py312`, select `E,F,I,UP,B,NPY`; `[tool.basedpyright] typeCheckingMode = "standard"`; `[tool.pytest.ini_options] addopts = "--import-mode=importlib"`, testpaths = the member `tests/` dirs.
- Member deps: `openvad-train` = numpy, safetensors (no torch yet); `openvad` = torch, safetensors. Build backend: uv's default for `uv init --lib` in the locked uv version.

Fixture values (exact in float32, so bytes match on every platform):

```python
WEIGHT = [[0.5, -0.25, 1.0, 0.75], [-1.0, 0.5, 0.25, -0.5]]
BIAS = [0.125, -0.25]
INPUT = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.0, 2.0], [0.25, -0.75, 1.5, -2.0]]
```

- [ ] **Step 1: Write the failing tests**

```python
# training/tests/test_export_smoke.py
def test_export_writes_expected(tmp_path):
    out = tmp_path / "linear.safetensors"
    main([str(out)])
    t = safetensors.numpy.load_file(out)
    assert {k: v.dtype for k, v in t.items()} == dict.fromkeys(["weight", "bias", "input", "expected"], np.float32)
    assert t["expected"].tolist() == [[6.125, -1.5], [1.0, 0.0], [0.4375, 0.5]]

def test_export_is_byte_identical(tmp_path):
    main([str(tmp_path / "a.safetensors")]); main([str(tmp_path / "b.safetensors")])
    assert (tmp_path / "a.safetensors").read_bytes() == (tmp_path / "b.safetensors").read_bytes()

# inferencing/openvad/tests/test_load_linear.py
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"

def test_torch_matches_fixture():
    t = safetensors.torch.load_file(FIXTURE)
    out = load_linear(FIXTURE)(t["input"])
    torch.testing.assert_close(out, t["expected"], atol=1e-6, rtol=0)
```

- [ ] **Step 2: Verify the Python layout:** confirm against the uv docs that a root without `[project]` is a valid workspace root, then `uv lock` and `uv sync --all-packages`.
- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest`
Expected: FAIL (`main` / `load_linear` not defined, fixture missing).

- [ ] **Step 4: Implement `main` and `load_linear`:** `expected` = float64 `INPUT @ WEIGHT.T + BIAS`, cast to float32; save with `safetensors.numpy.save_file`.
- [ ] **Step 5: Add recipes and generate the fixture:** `fixtures` = `uv run python -m openvad_train.export_smoke inferencing/fixtures/smoke/linear.safetensors`; `check-fixtures` = `just fixtures && git diff --exit-code inferencing/fixtures`; `lint-py` = `uv run ruff check`, `uv run ruff format --check`, `uv run basedpyright`; add `lint-py`, `check-fixtures` to `check` and `uv run pytest` to `test`. Run `just fixtures`.
- [ ] **Step 6: Verify**

Run: `uv sync --locked --all-packages && uv run pytest && (cd inferencing/openvad && uv run pytest) && just lint-py && ls -l inferencing/fixtures/smoke/`
Expected: all tests PASS from root and from the member dir; lint clean; fixture well under 1 MB. On Linux, `uv run python -c "import torch; print(torch.__version__)"` shows `2.14.1+cu130`.

- [ ] **Step 7: Commit** `build(python): add uv workspace, openvad, openvad-train and parity fixture`

### Task 4: Cargo workspace and `openvad-core`

**Files:**
- Create: `Cargo.toml`, `Cargo.lock`, `inferencing/openvad-core/Cargo.toml`, `inferencing/openvad-core/src/lib.rs`, `inferencing/openvad-core/tests/parity_smoke.rs`
- Modify: `justfile`

**Interfaces:**
- Consumes: fixture tensors from Task 3.
- Produces: `pub fn linear(input: &[f32], weight: &[f32], bias: &[f32], in_features: usize) -> Vec<f32>` (row-major; `input` is `rows * in_features`, `weight` is `out_features * in_features`, `out_features = bias.len()`; panics on mismatched lengths).
- Produces: `pub fn load_f32(bytes: &[u8], name: &str) -> Result<(Vec<usize>, Vec<f32>), Error>` and `pub enum Error { SafeTensors(safetensors::SafeTensorError), NotF32(String) }` with `Display` + `std::error::Error`.
- Root `Cargo.toml`: `[workspace]` members `["inferencing/openvad-core"]` (Task 5 adds `inferencing/openvad-rs`), `resolver = "3"`; `[workspace.package]` edition 2024, `authors = ["Lukas Hirsch"]`, `publish = false`; `[workspace.dependencies]` safetensors; `[workspace.lints]` `clippy.pedantic = { level = "warn", priority = -1 }`, `rust.unsafe_code = "deny"`.

- [ ] **Step 1: Write the failing test**

```rust
// tests/parity_smoke.rs
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures/smoke/linear.safetensors");

#[test]
fn linear_matches_pytorch_fixture() {
    let bytes = std::fs::read(FIXTURE).unwrap();
    let (_, w) = load_f32(&bytes, "weight").unwrap();
    let (_, b) = load_f32(&bytes, "bias").unwrap();
    let (shape, x) = load_f32(&bytes, "input").unwrap();
    let (_, expected) = load_f32(&bytes, "expected").unwrap();
    let out = linear(&x, &w, &b, shape[1]);
    assert_eq!(out.len(), expected.len());
    for (o, e) in out.iter().zip(&expected) { assert!((o - e).abs() <= 1e-6, "{o} vs {e}"); }
}

#[test]
fn load_f32_rejects_missing_tensor() {
    let bytes = std::fs::read(FIXTURE).unwrap();
    assert!(load_f32(&bytes, "nope").is_err());
}
```

- [ ] **Step 2: Run** `cargo test -p openvad-core`. Expected: FAIL (functions not defined).
- [ ] **Step 3: Implement** `linear` and `load_f32`: decode with `f32::from_le_bytes` over `chunks_exact(4)`, no `unsafe`, no other crates.
- [ ] **Step 4: Add recipes:** `test-rs` = `cargo test --workspace --locked`; `lint-rs` = `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked`; `guard-engine` fails if `cargo tree -p openvad-core -e normal` lists `ort`, `tract`, `burn` or `candle`. Add to `check`/`test`.
- [ ] **Step 5: Verify**

Run: `just test-rs && just lint-rs && just guard-engine && cargo tree -p openvad-core -e normal`
Expected: 2 tests PASS; no clippy errors; tree shows only `safetensors` and its deps.

- [ ] **Step 6: Commit** `build(rust): add cargo workspace and openvad-core with parity smoke test`

### Task 5: `openvad-rs` bindings

**Files:**
- Create: `inferencing/openvad-rs/Cargo.toml`, `inferencing/openvad-rs/pyproject.toml`, `inferencing/openvad-rs/src/lib.rs`, `inferencing/openvad-rs/python/openvad_rs/__init__.py`, `inferencing/openvad-rs/python/openvad_rs/_native.pyi`, `inferencing/openvad-rs/python/openvad_rs/py.typed`, `inferencing/openvad-rs/tests/test_parity_smoke.py`
- Modify: `Cargo.toml` (member), `pyproject.toml` (member), `uv.lock`, `Cargo.lock`

**Interfaces:**
- Consumes: `openvad_core::linear` (Task 4), fixture (Task 3).
- Produces: Python `openvad_rs.linear(input: Sequence[float], weight: Sequence[float], bias: Sequence[float], in_features: int) -> list[float]`, implemented in native module `openvad_rs._native` and re-exported by `__init__.py`.
- Crate: lib name `_native`, `crate-type = ["cdylib"]`, PyO3 with `abi3-py312`; maturin `module-name = "openvad_rs._native"`, `python-source = "python"`.

- [ ] **Step 1: Look up:** current PyO3 and maturin versions, whether the locked PyO3 still needs the `extension-module` feature, and the `tool.uv.cache-keys` syntax for globs and files outside the member dir.
- [ ] **Step 2: Write the failing test**

```python
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"

def test_bindings_match_fixture():
    t = safetensors.numpy.load_file(FIXTURE)
    out = openvad_rs.linear(t["input"].ravel().tolist(), t["weight"].ravel().tolist(), t["bias"].tolist(), 4)
    np.testing.assert_allclose(out, t["expected"].ravel(), atol=1e-6, rtol=0)
```

- [ ] **Step 3: Run** `uv run pytest inferencing/openvad-rs`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement** the `#[pyfunction] linear` wrapper and the `_native` `#[pymodule]`; `.pyi` with the signature above; `cache-keys` covering this member's `pyproject.toml`, `Cargo.toml`, `src/**/*.rs`, and `../openvad-core/Cargo.toml`, `../openvad-core/src/**/*.rs`.
- [ ] **Step 5: Test the rebuild (Review Focus 2)**

Run: add a temporary `#[pyfunction] fn probe() -> u32 { 1 }` to `_native`, `uv sync --all-packages`, `uv run python -c "import openvad_rs._native as n; print(n.probe())"`; change it to `2`, sync, run again; then remove `probe`, sync.
Expected: prints `1`, then `2`.

- [ ] **Step 6: Verify**

Run: `uv sync --locked --all-packages && uv run python -c "import openvad_rs; print(openvad_rs.linear)" && uv run pytest && (cd inferencing/openvad-rs && uv run pytest) && just check`
Expected: import works; all tests PASS from root and member dir; `just check` passes (`cargo test --workspace` must not fail to link the cdylib; if it does, set `test = false`, `doctest = false` on the `openvad-rs` lib target).

- [ ] **Step 7: Commit** `feat(bindings): add openvad-rs PyO3 bindings with parity smoke test`

### Task 6: Thesis skeleton with tudcd-scr

**Files:**
- Create: submodule `docs/tudcd-scr` at `c436bf5`, `.gitmodules`
- Modify: `flake.nix` (tudcd derivation, `devShells.docs`), `flake.lock`, `justfile`
- Create: `docs/.envrc`, `docs/README.md`, `docs/CLAUDE.md`
- Create: `docs/shared/references.bib`, `docs/shared/acronyms.tex`, `docs/shared/macros.tex`, `docs/shared/generated/numbers.tex` (generated), `docs/images/generated/smoke-expected.tex` (generated)
- Create: `docs/thesis/main.tex`, `docs/thesis/latexmkrc`, `docs/thesis/frontmatter/{titlepage,abstract,kurzfassung,declaration,ai-use,aufgabenstellung}.tex`, `docs/thesis/chapters/{01-introduction,02-background,03-related-work,04-data,05-model,06-evaluation,07-results,08-conclusion}.tex`, `docs/thesis/appendix/a-reproducibility.tex`
- Create: `training/src/openvad_train/export_docs.py`, `training/tests/test_export_docs.py`

**Interfaces:**
- Consumes: fixture (Task 3), shared env attrset (Task 1).
- Produces: `openvad_train.export_docs.main(argv: list[str] | None = None) -> None`, CLI `python -m openvad_train.export_docs <docs_dir>`: reads the fixture, writes `shared/generated/numbers.tex` with `\newcommand{\SmokeExpectedFirst}{6.125}` and `images/generated/smoke-expected.tex` (plain TikZ bars of the six `expected` values, no pgfplots unless the TeX set already has it). Both files start with `% Generated by openvad_train.export_docs. Do not edit.` (moves to `openvad-bench` when that exists).
- Produces: Nix derivation `tudcd-scr` (`stdenvNoCC`, src `./docs/tudcd-scr`, runs `l3build unpack`, installs generated `.cls`/`.sty` and logos into `$out/tex/latex/tudcd-scr/`); `devShells.docs` = `inputsFrom` default + TeX Live set (koma-script, fontspec, luaotfload, babel-english, biblatex, biber, acro, siunitx, pdfpages, tikz, l3build, latexmk, chktex, latexindent; extend only on "file not found") + `TEXMFHOME = tudcd-scr` + `OSFONTDIR` to the Noto fonts.
- Produces: recipes `thesis` (fails with "run `git submodule update --init`" if `docs/tudcd-scr/source/tudcd.ins` is missing, then `latexmk` in `docs/thesis`), `watch doc` (`latexmk -pvc`), `docs-generated`.

- [ ] **Step 1: Add the submodule** `git submodule add https://github.com/tud-cd/tudcd-scr docs/tudcd-scr && git -C docs/tudcd-scr checkout c436bf5`.
- [ ] **Step 2: Read** `docs/tudcd-scr/doc/handbook.tex`, `doc/demo-*.tex` and `source/*.dtx`. Note the documented `tudcdbook` class options, title page keys, and font setup; use nothing else.
- [ ] **Step 3: Write the failing test** for `export_docs`

```python
def test_export_docs_writes_number_and_figure(tmp_path):
    main([str(tmp_path)])
    numbers = (tmp_path / "shared/generated/numbers.tex").read_text()
    assert numbers.startswith("% Generated by openvad_train.export_docs. Do not edit.")
    assert r"\newcommand{\SmokeExpectedFirst}{6.125}" in numbers
    assert (tmp_path / "images/generated/smoke-expected.tex").read_text().count(r"\fill") == 6
```

- [ ] **Step 4: Run** `uv run pytest training`. Expected: FAIL.
- [ ] **Step 5: Implement** `export_docs`, add the Nix derivation and docs shell, `docs/.envrc` (`use flake ..#docs`), `latexmkrc` (`$pdf_mode = 4`, `$out_dir = 'build'`, biber), `main.tex` (`\documentclass[...]{tudcdbook}` on one line, babel `english`, biblatex, `scrlayer-scrpage` only if needed, `\input` of shared files). Title: "Reproducible Real-Time Voice Activity Detection under Noisy Telephone Conditions", author Lukas Hirsch. Each chapter: heading plus one `% TODO` line. Introduction cites the one bib entry (Silero VAD, `@misc`, GitHub URL) and uses `\SmokeExpectedFirst` and the generated figure. Frontmatter files are placeholders marked `TODO: confirm with Prüfungsamt/chair`; `aufgabenstellung.tex` includes `aufgabenstellung.pdf` via `\IfFileExists` and otherwise prints a placeholder. `docs/README.md`: build with Nix (`cd docs && direnv allow`, `just thesis`) and without (TeX Live, `cd docs/tudcd-scr && l3build install`), one sentence per line, never edit `generated/`. `docs/CLAUDE.md`: LuaLaTeX, read the handbook before any tudcd option, one sentence per line, never edit `generated/`.
- [ ] **Step 6: Test the missing-submodule path (Review Focus 1)** (run after the Step 8 commit: the clone only sees committed files)

Run: `d=$(mktemp -d) && git clone -q . "$d" && (cd "$d" && just thesis; echo "exit=$?")`
Expected: message "run `git submodule update --init`", `exit` non-zero. Delete the clone afterwards.

- [ ] **Step 7: Verify**

Run: `cd docs && direnv allow && nix develop ..#docs -c bash -c 'kpsewhich tudcdbook.cls && luaotfload-tool --find="Noto Sans" && luaotfload-tool --find="Noto Serif"' && just docs-generated && just thesis && ! grep -Ei 'undefined|warning.*tudcd' thesis/build/main.log && ls -l thesis/build/main.pdf && uv run pytest training`
Expected: class and both fonts found; PDF exists; grep finds nothing; tests PASS. Add a `typos.toml` only if `typos` flags German words.

- [ ] **Step 8: Commit** `docs: add thesis skeleton with tudcd-scr`

### Task 7: ADRs and benchmark prediction spec

**Files:**
- Create: `docs/project/adr/0001-no-onnx-in-engine.md`, `docs/project/adr/0002-safetensors-weights.md`, `docs/project/adr/0003-isolated-benchmark-adapters.md`, `docs/project/spec/benchmark-predictions.md`

Each ADR: Status, Context, Decision, Consequences, under 40 lines.
The spec states input (manifest + run config: sample rate, chunk size), output (one Parquet per audio file, columns `frame` (int, 10 ms index) and `speech_prob` (float32), plus `meta.json` with `system`, `version`, `weights_sha256`, `runtime`, `threads`, `cpu_model`), and that pyannote is labelled non-causal.

- [ ] **Step 1: Write the four files.**
- [ ] **Step 2: Verify** `pre-commit run --all-files` passes.
- [ ] **Step 3: Commit** `docs(project): add ADRs and benchmark prediction spec`

### Task 8: Claude configuration and README

**Files:**
- Create: `.claude/CLAUDE.md`, `.claude/settings.json`, `inferencing/CLAUDE.md`, `benchmarking/CLAUDE.md`
- Modify: `README.md` (add "Development setup")

- [ ] **Step 1: Write** `.claude/CLAUDE.md` with the content listed in the spec ("Claude" section), under 150 lines, plus the repo map and the `just` recipes as they now exist. `inferencing/CLAUDE.md`: parity rule, `1e-6` tolerance, `just fixtures` / `just check-fixtures`. `benchmarking/CLAUDE.md`: adapters are isolated uv projects outside the workspace, communicate by files (link the spec), same chunk size, 10 ms grid and resampling for every system, ONNX Runtime allowed only here, TEN VAD's extra conditions (Agora non-compete) keep its adapter isolated, never redistribute third-party weights or data.
- [ ] **Step 2: Write `settings.json`:** allow `Bash(just:*)`, `Bash(uv run:*)`, `Bash(uv sync:*)`, `Bash(cargo build:*)`, `Bash(cargo test:*)`, `Bash(cargo clippy:*)`, `Bash(cargo fmt:*)`, `Bash(nix build:*)`, `Bash(nix flake check:*)`, `Bash(latexmk:*)`, `Bash(git status:*)`, `Bash(git diff:*)`, `Bash(git log:*)`; deny `Read(./.env.local)`, `Edit(./data/datasets/**)`, `Write(./data/datasets/**)`, `Edit(./data/models/**)`, `Write(./data/models/**)`. Check the permission rule syntax against current Claude Code docs.
- [ ] **Step 3: Add "Development setup" to `README.md`:** with Nix (`direnv allow`, `just setup`, `just check`) and without (rustup, uv, `git clone --recursive`, `uv sync --all-packages`, `cargo test --workspace`, `uv run pytest`; TeX Live for docs). Short, no other README changes.
- [ ] **Step 4: Verify** `python -m json.tool .claude/settings.json` and `pre-commit run --all-files` pass; `wc -l .claude/CLAUDE.md` < 150.
- [ ] **Step 5: Commit** `chore(claude): add Claude configuration and development setup`

### Task 9: Non-Nix check and final report

- [ ] **Step 1: Run in a clean container (no Nix)**

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/src:ro ubuntu:24.04 bash -euc '
  apt-get update -qq && apt-get install -y -qq git curl ca-certificates build-essential >/dev/null
  git clone -q /src /work && cd /work
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"
  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
  uv sync --locked --all-packages && cargo test --workspace --locked && uv run pytest'
```

Expected: all three succeed. Any failure is a hidden Nix dependency: fix it in the owning task's files and commit `fix: ...`.

- [ ] **Step 2: Final checks:** `grep -rn $'\xe2\x80\x94' --exclude-dir={.git,target,.venv,tudcd-scr} .` finds nothing; `pre-commit run --all-files`, `nix flake check`, `just check` pass; `git log --oneline main..` shows one commit per task.
- [ ] **Step 3: Report** to Lukas: tree of created files, what was verified (with output), what was not (TeX without Nix, the HPC driver), decisions to review.
