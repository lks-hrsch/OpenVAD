# Tools

Third-party tools that run next to OpenVAD and never enter the release: the Montreal Forced Aligner (MFA) and Kaldi, which labels are built from ([ADR 0025](../docs/project/adr/0025-alignment-labelling.md), [dataset construction spec](../docs/project/spec/dataset-construction.md#mfa-in-tools)).
Only people who align need any of this: `just setup`, `uv sync` and the Nix shells never fetch the submodules, which only `just build-mfa` does.

## Layout

| Path | What |
| :-- | :-- |
| `montreal-forced-aligner` | git submodule, tag `v3.4.3` (2026-10-08) |
| `kaldi` | git submodule, commit `f4007661023b98b8081fd875029f0dee62242fd1` |
| `mfa` | uv project (`openvad-mfa`) with its own `uv.lock`: editable MFA, `kalpy-kaldi==0.10.5` and the runtime packages MFA needs but does not declare (`pgvector`, `psycopg2-binary`, `jinja2`), and the Japanese and Mandarin tokenisers (`sudachipy` below 0.7, whose 0.7 rejects the `NOOOVBOW2` category in MFA 3.4.3's `char.def`, `sudachidict-core`, `spacy-pkuseg`, `hanziconv`, `dragonmapper`); `spacy-pkuseg` downloads its model to `~/.pkuseg` on first use |
| `kaldi/build`, `kaldi/dist` | build output of `just build-mfa`, untracked (`ignore = untracked` on the submodule keeps it out of `git status`) |

`mfa` is a project of its own, not a dependency group of the root workspace.
A group with an editable path source on `tools/montreal-forced-aligner` made `uv sync --all-packages` fail in a clone whose submodule was not fetched (`does not appear to be a Python project`), so every `just setup` and `uv run` would have needed the submodules.
This is fallback 1 of the spec; `uv lock` itself never tried to build `kalpy-kaldi`.
Run it with `uv run --project tools/mfa ...`, or `just mfa <args>`, which also puts the Kaldi binaries on `PATH`.

## Pins

- MFA: tag `v3.4.3`, checked with `git ls-remote --tags` on the upstream repository (`93411fc`). Never `main`, which is heading to 4.0.
- Kaldi: `f4007661023b98b8081fd875029f0dee62242fd1` (merge of PR #4974, 2025-07-22). kalpy has no Kaldi pin of its own: its CI (`.github/workflows/main.yml`) installs `environment.yml` from conda-forge, which asks for `kaldi=*=*cpu*`. The conda-forge `kaldi-feedstock` recipe (`recipe/meta.yaml`, read on 2026-11-07) builds version `5.5.1172` from exactly this commit, so it is the Kaldi that `kalpy-kaldi` 0.10.x is built and tested against. The commit exists in `kaldi-asr/kaldi` (GitHub API).
- OpenFst `1.8.4`, the version Kaldi's `tools/Makefile` defaults to and `kalpy-kaldi` compiles against (`-DOPENFST_VER=18400`). The recipe downloads `openfst-1.8.4.tar.gz` from openfst.org and checks SHA-256 `a8ebbb6f3d92d07e671500587472518cfc87cb79b9a654a5a8abb2d0eb298016` (computed on 2026-11-07, the site publishes none).

The submodules are shallow (`shallow = true`, fetched with `--depth 1` by the pinned SHA). That is safe for Nix because `flake.nix` no longer sets `self.submodules` (it did, and Nix then refused an initialised shallow submodule: `is a shallow Git repository, so 'revCount' is not available`). A shallow checkout has the tag `v3.4.3` only while the pin is the tip of MFA's `main`; otherwise `setuptools_scm` would write a `0.1.dev1+g<sha>` version. So `build-mfa` fetches the tag (`git fetch --depth 1 origin tag v3.4.3`) and stops unless it is the pinned commit; when you move the pin, change `mfa_tag` in the `justfile` with it.
Kaldi's `src/configure` starts with `#!/bin/bash`, as do its makefiles' `SHELL`, and stock NixOS has no `/bin/bash`. Only on such a machine does the recipe pass `SHELL=<bash on PATH>` to every Kaldi `make` and rewrite that one shebang in the submodule, marked `git update-index --skip-worktree` and restored at the start of the next run, so moving the pin does not fail on it.

## Requirements

`just build-mfa` checks these first and names each one that is missing: `git`, `curl`, `shasum`, `getconf`, `tar`, `patch`, `make`, `cmake`, `sqlite3`, `uv`, a C++ compiler (`$CXX`, else `c++`), `initdb` and `pg_ctl`.

- `initdb` and `pg_ctl` come from a PostgreSQL server and must be on `PATH` (Debian and Ubuntu keep them in `/usr/lib/postgresql/<version>/bin`). MFA keeps its corpus in PostgreSQL and runs `CREATE EXTENSION` for `vector` ([pgvector](https://github.com/pgvector/pgvector)), `pg_trgm` and `pg_stat_statements`, so the server needs pgvector and the contrib modules (`postgresql-16-pgvector` on Ubuntu 24.04). `sqlite3` is the command-line tool, which MFA runs.
- `shasum` comes with Perl (Debian: `perl`). `getconf` comes with libc.
- A Python 3.12 that has its headers (`python3-dev`), or let uv download its own with `UV_PYTHON_PREFERENCE=only-managed`; the pynini and kalpy-kaldi builds need `Python.h`.
- Linux also needs `OPENBLAS_ROOT`, a prefix whose `lib/` holds an **LP64** `libopenblas.so` (32-bit integers; not an ILP64 build, which Kaldi's `int` arguments would overrun) and `libgfortran.so` (Kaldi links `-lopenblas -lgfortran`), with `cblas.h` and `lapacke.h` under its `include/` or on the compiler's include path. `just build-mfa` test-links both libraries before it builds anything. On Debian and Ubuntu (`libopenblas-dev`, `liblapacke-dev`, `gfortran`):

  ```sh
  mkdir -p ~/openblas-root/lib
  ln -sf /usr/lib/$(gcc -dumpmachine)/libopenblas.so $(gcc -print-file-name=libgfortran.so) ~/openblas-root/lib/
  export OPENBLAS_ROOT=~/openblas-root
  ```

  LAPACK 3.9.1 and later headers declare routines with character arguments with hidden string lengths, which Kaldi's `matrix/cblas-wrappers.h` does not pass; on Linux the recipe puts a `lapacke.h` in front of the real one that adds them. macOS uses the Accelerate framework.
- `just build-mfa` compiles a one-line check against `cblas.h` and stops if `OPENBLAS_USE64BITINT` is defined, so an ILP64 OpenBLAS cannot get as far as Kaldi. nixpkgs' `openblas` is ILP64 on x86_64, which is why the align shell uses `openblasCompat` (`-DINTERFACE64=FALSE`, no `OPENBLAS_USE64BITINT` in its `openblas_config.h`).
- Memory: compiling OpenFst needs about 2 GB per job. With 8 GB of RAM, set `JOBS=3` (default: all cores), or the compiler is killed.
- `nix develop .#align` provides all of it (cmake, curl, sqlite, PostgreSQL with pgvector, `OPENBLAS_ROOT` on Linux). Nothing in the recipe depends on Nix.

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

## Using it

- Once, on a new Linux machine such as the workstation, before trusting labels from it, run Kaldi's own matrix test, which exercises the BLAS and LAPACK calls (add `SHELL="$(command -v bash)"` to the `make` if there is no `/bin/bash`):

  ```sh
  make -C tools/kaldi/src/matrix matrix-lib-test EXTRA_CXXFLAGS="-I$PWD/tools/kaldi/build/lapack-shim"
  tools/kaldi/src/matrix/matrix-lib-test    # must end with "Tests succeeded." and exit 0
  ```

  On macOS drop the `EXTRA_CXXFLAGS`. A wrong integer width or a bad LAPACK call shows up there as a failed check or a crash.

- `uv sync --project tools/mfa` without `--inexact` uninstalls the pynini wheel, which is not in `tools/mfa/pyproject.toml`; `mfa` then fails at import. `just build-mfa` puts it back and always syncs with `--inexact`.
- A clean `mfa align` exit does not mean every clip aligned. Clips that fail feature generation (silence, noise, a transcript far too long for the audio, a missing `.lab`) get no TextGrid and at most one log warning and a row with blank likelihoods in `alignment_analysis.csv`; a corrupt audio file aborts the whole run with exit 1 and no output. A batch must therefore validate its audio first and compare the clip IDs it sent with the TextGrids it got.
- Words missing from the dictionary keep their spelling in the words tier and get `spn` in the phones tier, so counting words proves nothing about them; look for `spn`.
- On macOS (Accelerate, `german_mfa`), aligning needed none of `baumwelch` and OpenGrm `ngram`, which the build does not provide; it did need `fstcompile` and the Kaldi executables on `PATH`.

## Licences

Checked in each repository on 2026-11-07. All five run as tools, are not linked into OpenVAD and do not enter the release.

| Tool | Licence | Checked in |
| :-- | :-- | :-- |
| MFA 3.4.3 | MIT | `LICENSE` of the submodule |
| Kaldi `f400766` | Apache-2.0 | `COPYING` of the submodule (the GitHub licence field says `NOASSERTION` only because the file starts with a notice about copyright headers) |
| kalpy and `kalpy-kaldi` 0.10.5 | MIT | `LICENSE` of `MontrealCorpusTools/kalpy`, sdist metadata on PyPI |
| pynini 2.1.7 | Apache-2.0 | `LICENSE` and `PKG-INFO` of the sdist |
| OpenFst 1.8.4 | Apache-2.0 | `COPYING` of the tarball |

No GPL and no unlicensed component. The environment in `tools/mfa/.venv` also holds two LGPL libraries, `psycopg2-binary` (LGPL with exceptions) and `soxr` (LGPL-2.1-or-later, a dependency of librosa); like the rest they run as tools and are not in the release.
