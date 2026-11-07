# .env holds local secrets such as MDC_API_KEY (git-ignored)
set dotenv-load

default:
    just --list

# fetch the thesis submodule and install the Python workspace (tools/ is fetched by build-mfa)
setup:
    git submodule update --init docs/tudcd-scr
    uv sync --all-packages

# format all code
fmt:
    uv run ruff format
    cargo fmt --all

# lint and type-check
check: lint-py check-fixtures lint-rs guard-engine test

# run all tests
test: test-rs
    uv run pytest

# regenerate the parity fixture
fixtures:
    uv run python -m openvad_train.export_smoke inferencing/fixtures/smoke/linear.safetensors

# fail if the committed fixture is stale
check-fixtures:
    just fixtures
    git diff --exit-code inferencing/fixtures

# lint, format-check and type-check Python
lint-py:
    uv run ruff check
    uv run ruff format --check
    uv run basedpyright

# run Rust tests
test-rs:
    cargo test --workspace --locked

# check Rust formatting and clippy
lint-rs:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --locked -- -D warnings

# regenerate the resampler coefficients
design-resampler:
    uv run python -m openvad_train.design_resampler

# fail if any Rust crate pulls in a third-party inference runtime
guard-engine:
    out=$(cargo tree --workspace --locked -e normal,build,dev) && ! echo "$out" | grep -E '(^|[^A-Za-z0-9_-])(ort|tract|burn|candle)(-[A-Za-z0-9_-]+)? v[0-9]'

# build the thesis PDF into docs/thesis/build
thesis:
    @test -f docs/tudcd-scr/source/tudcd.ins || { echo 'docs/tudcd-scr is empty: run `git submodule update --init docs/tudcd-scr`' >&2; exit 1; }
    cd docs/thesis && latexmk

# rebuild a document on every save, e.g. `just watch thesis`
watch doc:
    cd docs/{{doc}} && latexmk -pvc

# regenerate numbers and figures under docs/
docs-generated:
    uv run python -m openvad_train.export_docs docs

# fetch one dataset into $OPENVAD_DATA_DIR/datasets/raw (see data/README.md)
fetch dataset:
    uv run python -m openvad_train.fetch {{dataset}}

# run one dataset construction stage (see docs/project/spec/dataset-construction.md)
dataset stage *args:
    uv run python -m openvad_train.dataset {{stage}} {{args}}

# build Kaldi into tools/kaldi/dist and the MFA environment in tools/mfa (see tools/README.md)
build-mfa:
    #!/usr/bin/env bash
    set -euo pipefail
    cxx="${CXX:-c++}"
    missing=0
    for tool in git curl shasum getconf tar patch make cmake sqlite3 uv initdb pg_ctl "$cxx"; do
        if ! command -v "$tool" >/dev/null; then
            echo "build-mfa: $tool not found on PATH" >&2
            missing=1
        fi
    done
    [ "$missing" = 0 ] || { echo "build-mfa: install what is missing, see tools/README.md" >&2; exit 1; }
    if [ "$(uname)" = Linux ]; then
        if [ -z "${OPENBLAS_ROOT:-}" ]; then
            echo "build-mfa: set OPENBLAS_ROOT to a prefix with include/cblas.h, include/lapacke.h, lib/libopenblas and libgfortran" >&2
            exit 1
        fi
        # Kaldi's configure links -lopenblas -lgfortran; find out now rather than after OpenFst is built.
        echo 'int main() {}' | "$cxx" -x c++ - -L"$OPENBLAS_ROOT/lib" -lopenblas -lgfortran -o /dev/null \
            || { echo "build-mfa: $cxx cannot link -lopenblas -lgfortran from $OPENBLAS_ROOT/lib" >&2; exit 1; }
    fi

    # Kaldi wants /bin/bash (the shebang of configure, SHELL in its makefiles), which e.g. NixOS lacks. Without it,
    # every Kaldi make gets SHELL=<bash on PATH> and configure's shebang is rewritten. configure is a tracked
    # file, so the rewrite is marked skip-worktree and undone here, before the submodule can move.
    if [ -e tools/kaldi/.git ]; then
        git -C tools/kaldi update-index --no-skip-worktree src/configure
        git -C tools/kaldi checkout -- src/configure
    fi
    git submodule update --init --depth 1 tools/kaldi tools/montreal-forced-aligner
    # A shallow checkout has no tags unless the pin is the tip of main, and setuptools_scm would then write
    # 0.1.dev1+g<sha> as the MFA version. Fetch the tag and stop unless it is the pinned commit.
    mfa_tag=v3.4.3
    git -C tools/montreal-forced-aligner rev-parse -q --verify "$mfa_tag^{commit}" >/dev/null \
        || git -C tools/montreal-forced-aligner fetch --depth 1 origin tag "$mfa_tag"
    [ "$(git -C tools/montreal-forced-aligner rev-parse "$mfa_tag^{commit}")" = "$(git rev-parse :tools/montreal-forced-aligner)" ] \
        || { echo "build-mfa: tag $mfa_tag of tools/montreal-forced-aligner is not the pinned commit; move the pin and mfa_tag together" >&2; exit 1; }
    kmake() {
        if [ -x /bin/bash ]; then make "$@"; else make SHELL="$(command -v bash)" "$@"; fi
    }
    if [ ! -x /bin/bash ]; then
        sed -i.orig "1s|.*|#!$(command -v bash)|" tools/kaldi/src/configure
        rm tools/kaldi/src/configure.orig
        chmod +x tools/kaldi/src/configure
        git -C tools/kaldi update-index --skip-worktree src/configure
    fi
    dist="$PWD/tools/kaldi/dist"
    work="$PWD/tools/kaldi/build"
    jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
    fst=1.8.4
    fst_sha256=a8ebbb6f3d92d07e671500587472518cfc87cb79b9a654a5a8abb2d0eb298016
    kaldi_commit=$(git -C tools/kaldi rev-parse HEAD)
    rebuilt=
    mkdir -p "$dist" "$work"

    # fetch URL SHA256 DEST: keep DEST if it matches the checksum; otherwise download to a temporary name and
    # move it into place once the checksum passes, so an interrupted, corrupt or outdated file is replaced.
    fetch() {
        if [ -f "$3" ] && echo "$2  $3" | shasum -a 256 -c - >/dev/null 2>&1; then return 0; fi
        curl -fsSL -o "$3.part" "$1"
        echo "$2  $3.part" | shasum -a 256 -c - || { rm -f "$3.part"; exit 1; }
        mv "$3.part" "$3"
    }

    # Each stamp is written last and names the upstream versions it was built from, so a half-finished
    # step is redone and a finished one is skipped. A step that runs also removes the stamps of the steps
    # after it (OpenFst, Kaldi, pynini, kalpy-kaldi), so they are redone against it.
    pynini=2.1.7
    feedstock=9aed7bb122e77fc1d73a2c62070f5b003e5e956d
    if [ ! -f "$dist/.openfst-$fst" ]; then
        rm -f "$dist"/.openfst-* "$dist"/.kaldi-* "$dist"/.pynini-* "$dist"/.kalpy-*
        tarball="$work/openfst-$fst.tar.gz"
        fetch "https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$fst.tar.gz" "$fst_sha256" "$tarball"
        tar -xzf "$tarball" -C "$work"
        (cd "$work/openfst-$fst" && ./configure --prefix="$dist" --enable-far --enable-ngram-fsts \
            --enable-lookahead-fsts --enable-pdt --enable-mpdt --with-pic CXX="$cxx" LIBS=-ldl && make -j"$jobs" && make install)
        touch "$dist/.openfst-$fst"
    fi

    kaldi_stamp="$dist/.kaldi-$kaldi_commit-openfst$fst"
    if [ ! -f "$kaldi_stamp" ]; then
        rm -f "$dist"/.kaldi-* "$dist"/.kalpy-*
        blas=()
        [ -z "${OPENBLAS_ROOT:-}" ] || blas=(--mathlib=OPENBLAS --openblas-root="$OPENBLAS_ROOT")
        # LAPACK 3.9.1 and later headers (Ubuntu 24.04, nixpkgs OpenBLAS 0.3.33) declare the routines that take
        # character arguments with hidden string lengths, and Kaldi's matrix/cblas-wrappers.h calls them without
        # ("too few arguments to function stptri_"). A lapacke.h in front of the real one passes them.
        kflags=
        if [ -n "${OPENBLAS_ROOT:-}" ]; then
            mkdir -p "$work/lapack-shim"
            {
                echo '#include_next <lapacke.h>'
                echo '#ifdef LAPACK_FORTRAN_STRLEN_END'
                for fn in stptri_ dtptri_ sgesvd_ dgesvd_; do echo "#define $fn(...) $fn(__VA_ARGS__, 1, 1)"; done
                for fn in ssptri_ dsptri_ ssptrf_ dsptrf_; do echo "#define $fn(...) $fn(__VA_ARGS__, 1)"; done
                echo '#endif'
            } > "$work/lapack-shim/lapacke.h"
            kflags="-I$work/lapack-shim"
        fi
        (cd tools/kaldi/src && ./configure --shared --use-cuda=no --fst-root="$dist" --fst-version="$fst" ${blas[@]+"${blas[@]}"} \
            && kmake -j"$jobs" EXTRA_CXXFLAGS="$kflags")
        # Kaldi's nnet2/Makefile leaves out combine-nnet-a.cc and shrink-nnet.cc, which its CMake build globs in
        # and kalpy-kaldi needs (nnet2::ShrinkNnet), so add them to the library.
        printf 'include Makefile\nOBJFILES += combine-nnet-a.o shrink-nnet.o\nkaldi-nnet2.a: combine-nnet-a.o shrink-nnet.o\n' > "$work/nnet2.mk"
        kmake -C tools/kaldi/src/nnet2 -f "$work/nnet2.mk" -j"$jobs" EXTRA_CXXFLAGS="$kflags"
        # kalpy-kaldi links libkaldi-online, which Kaldi builds only with PortAudio (make ext); its own CMake
        # build compiles it with -DKALDI_NO_PORTAUDIO, so do the same and link only the Kaldi libraries.
        ext=so
        [ "$(uname)" != Darwin ] || ext=dylib
        deps=
        for lib in decoder lat hmm feat transform gmm tree util matrix base; do deps="$deps ../$lib/libkaldi-$lib.$ext"; done
        kmake -C tools/kaldi/src/online -j"$jobs" EXTRA_CXXFLAGS="$kflags -DKALDI_NO_PORTAUDIO=1" EXTRA_LDLIBS="$deps"
        # kalpy wants KALDI_ROOT/{include/kaldi,lib}; link the in-tree build, whose rpath is src/lib.
        mkdir -p "$dist/include" "$dist/lib" "$dist/bin"
        ln -sfn ../../src "$dist/include/kaldi"
        ln -sf "$PWD"/tools/kaldi/src/lib/* "$dist/lib/"
        for exe in tools/kaldi/src/*bin/*; do
            [ -f "$exe" ] && [ -x "$exe" ] && case "$exe" in *.*) ;; *) ln -sf "$PWD/$exe" "$dist/bin/" ;; esac
        done
        touch "$kaldi_stamp"
    fi

    # kalpy-kaldi casts pywrapfst objects through _pywrapfst.h, which only a pynini with the conda-forge
    # "Add API declarations" patch generates; the PyPI pynini lacks it. So pynini is built here from its
    # sdist plus that patch, against the OpenFst in dist, and installed into the env after the sync.
    pynini_dir="$work/pynini-$pynini"
    pynini_stamp="$dist/.pynini-$pynini-feedstock${feedstock:0:7}-openfst$fst"
    if [ ! -f "$pynini_stamp" ]; then
        rm -f "$dist"/.pynini-* "$dist"/.kalpy-*
        rm -rf "$pynini_dir" "$work"/wheels
        fetch https://files.pythonhosted.org/packages/c6/b6/09a0efe843df37d9c80a9a1c190c7bb28d8920cee401e2a61c46f1328ec1/pynini-$pynini.tar.gz \
            fc06cc02adcbd4b14eb94ae9f2a0949d110612d2d4ee3a7ac5dc97b45fff8e33 "$work/pynini.tar.gz"
        fetch https://raw.githubusercontent.com/conda-forge/pynini-feedstock/$feedstock/recipe/patches/0002-Add-API-declarations.patch \
            846df2906b5be1b84085fde562d2093e2149aeb9ddc8677f63169a74ee7a331e "$work/pynini-api.patch"
        tar -xzf "$work/pynini.tar.gz" -C "$work"
        (cd "$pynini_dir" && patch -p0 < "$work/pynini-api.patch")
        CPPFLAGS="-I$dist/include" LDFLAGS="-L$dist/lib -Wl,-rpath,$dist/lib" \
            uv build --wheel --out-dir "$work/wheels" "$pynini_dir"
        cp "$pynini_dir/extensions/_pywrapfst.h" "$dist/include/_pywrapfst.h"
        touch "$pynini_stamp"
    fi

    # uv keeps an installed kalpy-kaldi and reuses its cached wheel even for --reinstall-package, so without the
    # kalpy stamp (also missing after an interrupted run) it syncs with --no-cache and compiles the package again,
    # against the Kaldi and OpenFst of this run. (uv cache clean would wait for every running uvx tool.)
    kalpy_stamp="$dist/.kalpy-$kaldi_commit-openfst$fst-pynini$pynini"
    [ -f "$kalpy_stamp" ] || rebuilt=1
    # Its CMake build needs pybind11's config dir, which only the pip package has, inside the package.
    pybind11_dir=$(uvx --from pybind11==3.1.0 pybind11-config --cmakedir)
    KALDI_ROOT="$dist" CMAKE_ARGS="-Dpybind11_DIR=$pybind11_dir" CMAKE_BUILD_PARALLEL_LEVEL="$jobs" \
        LDFLAGS="-Wl,-rpath,$dist/lib" \
        uv sync --inexact --project tools/mfa ${rebuilt:+--reinstall-package kalpy-kaldi --no-cache}
    venv=$(uv run --project tools/mfa --no-sync python -c 'import sys; print(sys.prefix)')
    uv pip install --python "$venv/bin/python" --no-deps --reinstall "$work"/wheels/pynini-*.whl
    touch "$kalpy_stamp"

# run mfa from tools/mfa with the Kaldi binaries on PATH (after build-mfa)
mfa *args:
    PATH="$PWD/tools/kaldi/dist/bin:$PATH" uv run --project tools/mfa --no-sync mfa {{args}}

# restore the Claude skills pinned in skills-lock.json (needs Node; experimental command)
skills:
    npx skills experimental_install
