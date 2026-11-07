# docs

The Master's thesis lives here, written in LuaLaTeX on the TU Dresden class bundle [tudcd-scr](https://github.com/tud-cd/tudcd-scr).

## Layout

- `thesis/`: the thesis, built with `latexmk` into `thesis/build/main.pdf`.
- `shared/`: bibliography, acronyms and macros.
- `shared/generated/` and `images/generated/`: numbers and figures written by `just docs-generated`.
- `notes/`: build and hardware, dataset, forced-alignment and speech-quality notes (Markdown).
- `project/`: ADRs, specs and design documents.
- `superpowers/`: the historical datasets design and plan of 2026-10-31, superseded by ADRs 0023 and 0024 and the current fetchers.
- `tudcd-scr/`: the class bundle, a git submodule pinned to `c436bf5`. The docs shell takes the same classes from the `tudcd-scr` input of `flake.nix`, which must name the same commit: when you move the submodule, change the rev in `flake.nix` and run `nix flake lock` too.

Never edit files under `generated/` by hand.
Change the script that writes them and run `just docs-generated` instead.

## Build with Nix

Run `cd docs && direnv allow` once to enter the docs shell, which brings TeX Live and the unpacked tudcd-scr classes.
Then, still inside `docs/`, run `just thesis`, or `just watch thesis` to rebuild on every save.

## Build without Nix (best effort)

The thesis only has to build in the Nix docs shell; machines without Nix, such as the TU Dresden HPC, need not build it.

Install a TeX Live newer than 2023.3 (tudcd-scr needs it) with LuaLaTeX, biber and latexmk.
Run `git submodule update --init` and then `cd docs/tudcd-scr && l3build install` to install the classes into your `TEXMFHOME`.
Then run `just thesis` from the repository root.
