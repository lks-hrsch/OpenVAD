# Tools

Third-party tools that run next to OpenVAD and never enter the release: the Montreal Forced Aligner (MFA) and Kaldi, which labels are built from ([ADR 0025](../docs/project/adr/0025-alignment-labelling.md), [dataset construction spec](../docs/project/spec/dataset-construction.md#mfa-in-tools)).
Only people who align need any of this. `just setup` and `uv sync` never touch it.

## Layout

| Path | What |
| :-- | :-- |
| `montreal-forced-aligner` | git submodule, tag `v3.4.3` (2026-10-08) |
| `kaldi` | git submodule, commit `f4007661023b98b8081fd875029f0dee62242fd1` |
| `mfa` | uv project (`openvad-mfa`) with its own `uv.lock`: editable MFA, `kalpy-kaldi==0.10.5` and the runtime packages MFA needs but does not declare (`pgvector`, `psycopg2-binary`) |
| `kaldi/build`, `kaldi/dist` | build output of `just build-mfa`, git-ignored |

`mfa` is a project of its own, not a dependency group of the root workspace.
A group with an editable path source on `tools/montreal-forced-aligner` made `uv sync --all-packages` fail in a clone whose submodule was not fetched (`does not appear to be a Python project`), so every `just setup` and `uv run` would have needed the submodules.
This is fallback 1 of the spec; `uv lock` itself never tried to build `kalpy-kaldi`.
Run it with `uv run --project tools/mfa ...`, or `just mfa <args>`, which also puts the Kaldi binaries on `PATH`.

## Pins

- MFA: tag `v3.4.3`, checked with `git ls-remote --tags` on the upstream repository (`93411fc`). Never `main`, which is heading to 4.0.
- Kaldi: `f4007661023b98b8081fd875029f0dee62242fd1` (merge of PR #4974, 2025-07-22). kalpy has no Kaldi pin of its own: its CI (`.github/workflows/main.yml`) installs `environment.yml` from conda-forge, which asks for `kaldi=*=*cpu*`. The conda-forge `kaldi-feedstock` recipe (`recipe/meta.yaml`, read on 2026-11-07) builds version `5.5.1172` from exactly this commit, so it is the Kaldi that `kalpy-kaldi` 0.10.x is built and tested against. The commit exists in `kaldi-asr/kaldi` (GitHub API).
- OpenFst `1.8.4`, the version Kaldi's `tools/Makefile` defaults to and `kalpy-kaldi` compiles against (`-DOPENFST_VER=18400`). The recipe downloads `openfst-1.8.4.tar.gz` from openfst.org and checks SHA-256 `a8ebbb6f3d92d07e671500587472518cfc87cb79b9a654a5a8abb2d0eb298016` (computed on 2026-11-07, the site publishes none).

The submodules are full clones, not `shallow = true`: the flake sets `self.submodules`, and Nix refuses an initialised shallow submodule (`is a shallow Git repository, so 'revCount' is not available`), which would break `nix develop` and direnv.

## Requirements

`just build-mfa` checks these first and names each one that is missing: `git`, `curl`, `tar`, `patch`, `make`, `cmake`, `uv`, a C++ compiler (`$CXX`, else `c++`), `initdb` and `pg_ctl`.

- `initdb` and `pg_ctl` come from a PostgreSQL server. MFA keeps its corpus in PostgreSQL and runs `CREATE EXTENSION` for `vector` ([pgvector](https://github.com/pgvector/pgvector)), `pg_trgm` and `pg_stat_statements`, so the server needs pgvector and the contrib modules.
- Linux also needs `OPENBLAS_ROOT`, a prefix with `include/cblas.h`, `include/lapacke.h` and `lib/libopenblas`. macOS uses the Accelerate framework.
- `nix develop .#align` provides all of it (cmake, curl, PostgreSQL with pgvector, `OPENBLAS_ROOT` on Linux). Nothing in the recipe depends on Nix.

## Build

```sh
just build-mfa
just mfa version        # 3.4.3
```

`just build-mfa` is re-runnable.
It fetches the two submodules, builds OpenFst 1.8.4 (with FAR, PDT and MPDT) and Kaldi (`configure --shared`, then `make`; Kaldi's default `-O1`; plus `libkaldi-online`, see below) and links the build into `tools/kaldi/dist`, the `KALDI_ROOT` that `kalpy-kaldi` wants (`include/kaldi`, `lib`, plus `bin` with the OpenFst and Kaldi executables).
It then builds pynini (see below) and runs `uv sync --inexact --project tools/mfa` with `KALDI_ROOT` set, which compiles `kalpy-kaldi` with CMake, and installs the pynini wheel into that environment.
A stamp file in `dist` is written after each step, so a finished step is skipped and an interrupted one is redone; `make` resumes where it stopped.
After a new Kaldi commit the recipe rebuilds Kaldi and reinstalls `kalpy-kaldi`.
To start over, delete `tools/kaldi/build` and `tools/kaldi/dist` and run `git -C tools/kaldi clean -fdx`.

Four things differ from the conda route that MFA documents, each forced by something found while building:

- **pynini is built from its sdist with a patch.** `kalpy-kaldi` includes `_pywrapfst.h` and casts `pywrapfst` objects through its struct types (`VectorFstObject` and others). Only conda-forge's pynini feedstock generates that header, with its patch `0002-Add-API-declarations.patch`, which makes the Cython classes `public`. The recipe downloads that patch at feedstock commit `9aed7bb122e77fc1d73a2c62070f5b003e5e956d` and the pynini 2.1.7 sdist, checks both SHA-256 sums, builds a wheel against the OpenFst in `dist` and copies the header into `dist/include`. The PyPI wheel would also bundle a second OpenFst next to Kaldi's, and kalpy passes FST objects between the two. The wheel is installed after `uv sync --inexact`, so `pynini` is not in `tools/mfa/pyproject.toml`.
- **`libkaldi-online` is compiled without PortAudio.** Kaldi's make build only builds it with `make ext` and PortAudio; its CMake build (what conda runs) uses `-DKALDI_NO_PORTAUDIO`, and so does the recipe.
- **Two sources are added to `libkaldi-nnet2`.** Kaldi's `nnet2/Makefile` omits `combine-nnet-a.cc` and `shrink-nnet.cc`, which its CMake build globs in; `kalpy-kaldi` needs `nnet2::ShrinkNnet` and fails to import without it.
- **Kaldi is built with `configure` and `make`, not CMake.** Its CMake build hard-codes `-lcblas -llapack`, which macOS does not have, and fetches its own OpenFst 1.7.2 fork, while `kalpy-kaldi` compiles against OpenFst 1.8.4.

The directory is `dist`, not `install`, because Kaldi ships a file called `INSTALL` and macOS file systems are case-insensitive.

MFA checks that the Kaldi binaries (`compute-mfcc-feats`, `gmm-align-compiled` and others) and `fstcompile` are on `PATH` before most commands, and `mfa` also needs a PostgreSQL server; `just mfa` sets `PATH`, and MFA starts its own server on first use.

## Licences

Checked in each repository on 2026-11-07. All four run as tools, are not linked into OpenVAD and do not enter the release.

| Tool | Licence | Checked in |
| :-- | :-- | :-- |
| MFA 3.4.3 | MIT | `LICENSE` of the submodule |
| Kaldi `f400766` | Apache-2.0 | `COPYING` of the submodule (the GitHub licence field says `NOASSERTION` only because the file starts with a notice about copyright headers) |
| kalpy and `kalpy-kaldi` 0.10.5 | MIT | `LICENSE` of `MontrealCorpusTools/kalpy`, sdist metadata on PyPI |
| pynini 2.1.7 | Apache-2.0 | `LICENSE` and `PKG-INFO` of the sdist |
| OpenFst 1.8.4 | Apache-2.0 | `COPYING` of the tarball |

No GPL and no unlicensed component.
