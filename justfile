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
    @test -f docs/tudcd-scr/source/tudcd.ins || { echo 'docs/tudcd-scr is empty: run `git submodule update --init`' >&2; exit 1; }
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
    for tool in git curl tar patch make cmake uv initdb pg_ctl "$cxx"; do
        if ! command -v "$tool" >/dev/null; then
            echo "build-mfa: $tool not found on PATH" >&2
            missing=1
        fi
    done
    [ "$missing" = 0 ] || { echo "build-mfa: install what is missing, see tools/README.md" >&2; exit 1; }
    if [ "$(uname)" = Linux ] && [ -z "${OPENBLAS_ROOT:-}" ]; then
        echo "build-mfa: set OPENBLAS_ROOT to a prefix with include/cblas.h, include/lapacke.h and lib/libopenblas" >&2
        exit 1
    fi

    git submodule update --init tools/kaldi tools/montreal-forced-aligner
    dist="$PWD/tools/kaldi/dist"
    work="$PWD/tools/kaldi/build"
    jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
    fst=1.8.4
    fst_sha256=a8ebbb6f3d92d07e671500587472518cfc87cb79b9a654a5a8abb2d0eb298016
    kaldi_commit=$(git -C tools/kaldi rev-parse HEAD)
    rebuilt=
    mkdir -p "$dist" "$work"

    # Each stamp is written last, so a half-finished step is redone and a finished one is skipped.
    if [ ! -f "$dist/.openfst-$fst" ]; then
        tarball="$work/openfst-$fst.tar.gz"
        [ -f "$tarball" ] || curl -fsSL -o "$tarball" "https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$fst.tar.gz"
        echo "$fst_sha256  $tarball" | shasum -a 256 -c -
        tar -xzf "$tarball" -C "$work"
        (cd "$work/openfst-$fst" && ./configure --prefix="$dist" --enable-far --enable-ngram-fsts \
            --enable-lookahead-fsts --enable-pdt --enable-mpdt --with-pic CXX="$cxx" LIBS=-ldl && make -j"$jobs" && make install)
        touch "$dist/.openfst-$fst"
    fi

    if [ ! -f "$dist/.kaldi-$kaldi_commit" ]; then
        rebuilt=1
        rm -f "$dist"/.kaldi-*
        blas=()
        [ -z "${OPENBLAS_ROOT:-}" ] || blas=(--mathlib=OPENBLAS --openblas-root="$OPENBLAS_ROOT")
        (cd tools/kaldi/src && ./configure --shared --use-cuda=no --fst-root="$dist" --fst-version="$fst" ${blas[@]+"${blas[@]}"} \
            && make -j"$jobs")
        # Kaldi's nnet2/Makefile leaves out combine-nnet-a.cc and shrink-nnet.cc, which its CMake build globs in
        # and kalpy-kaldi needs (nnet2::ShrinkNnet), so add them to the library.
        printf 'include Makefile\nOBJFILES += combine-nnet-a.o shrink-nnet.o\nkaldi-nnet2.a: combine-nnet-a.o shrink-nnet.o\n' > "$work/nnet2.mk"
        make -C tools/kaldi/src/nnet2 -f "$work/nnet2.mk" -j"$jobs"
        # kalpy-kaldi links libkaldi-online, which Kaldi builds only with PortAudio (make ext); its own CMake
        # build compiles it with -DKALDI_NO_PORTAUDIO, so do the same and link only the Kaldi libraries.
        ext=so
        [ "$(uname)" != Darwin ] || ext=dylib
        deps=
        for lib in decoder lat hmm feat transform gmm tree util matrix base; do deps="$deps ../$lib/libkaldi-$lib.$ext"; done
        make -C tools/kaldi/src/online -j"$jobs" EXTRA_CXXFLAGS=-DKALDI_NO_PORTAUDIO=1 EXTRA_LDLIBS="$deps"
        # kalpy wants KALDI_ROOT/{include/kaldi,lib}; link the in-tree build, whose rpath is src/lib.
        mkdir -p "$dist/include" "$dist/lib" "$dist/bin"
        ln -sfn ../../src "$dist/include/kaldi"
        ln -sf "$PWD"/tools/kaldi/src/lib/* "$dist/lib/"
        for exe in tools/kaldi/src/*bin/*; do
            [ -f "$exe" ] && [ -x "$exe" ] && case "$exe" in *.*) ;; *) ln -sf "$PWD/$exe" "$dist/bin/" ;; esac
        done
        touch "$dist/.kaldi-$kaldi_commit"
    fi

    # kalpy-kaldi casts pywrapfst objects through _pywrapfst.h, which only a pynini with the conda-forge
    # "Add API declarations" patch generates; the PyPI pynini lacks it. So pynini is built here from its
    # sdist plus that patch, against the OpenFst in dist, and installed into the env after the sync.
    pynini=2.1.7
    pynini_dir="$work/pynini-$pynini"
    if [ ! -f "$dist/.pynini-$pynini" ]; then
        feedstock=9aed7bb122e77fc1d73a2c62070f5b003e5e956d
        rm -rf "$pynini_dir" "$work"/wheels
        curl -fsSL -o "$work/pynini.tar.gz" https://files.pythonhosted.org/packages/c6/b6/09a0efe843df37d9c80a9a1c190c7bb28d8920cee401e2a61c46f1328ec1/pynini-$pynini.tar.gz
        echo "fc06cc02adcbd4b14eb94ae9f2a0949d110612d2d4ee3a7ac5dc97b45fff8e33  $work/pynini.tar.gz" | shasum -a 256 -c -
        curl -fsSL -o "$work/pynini-api.patch" https://raw.githubusercontent.com/conda-forge/pynini-feedstock/$feedstock/recipe/patches/0002-Add-API-declarations.patch
        echo "846df2906b5be1b84085fde562d2093e2149aeb9ddc8677f63169a74ee7a331e  $work/pynini-api.patch" | shasum -a 256 -c -
        tar -xzf "$work/pynini.tar.gz" -C "$work"
        (cd "$pynini_dir" && patch -p0 < "$work/pynini-api.patch")
        CPPFLAGS="-I$dist/include" LDFLAGS="-L$dist/lib -Wl,-rpath,$dist/lib" \
            uv build --wheel --out-dir "$work/wheels" "$pynini_dir"
        cp "$pynini_dir/extensions/_pywrapfst.h" "$dist/include/_pywrapfst.h"
        touch "$dist/.pynini-$pynini"
    fi

    # kalpy-kaldi is cached by uv as a wheel, so force it to be rebuilt against a new Kaldi.
    # Its CMake build needs pybind11's config dir, which only the pip package has, inside the package.
    pybind11_dir=$(uvx --from pybind11==3.1.0 pybind11-config --cmakedir)
    KALDI_ROOT="$dist" CMAKE_ARGS="-Dpybind11_DIR=$pybind11_dir" CMAKE_BUILD_PARALLEL_LEVEL="$jobs" \
        LDFLAGS="-Wl,-rpath,$dist/lib" \
        uv sync --inexact --project tools/mfa ${rebuilt:+--reinstall-package kalpy-kaldi}
    venv=$(uv run --project tools/mfa --no-sync python -c 'import sys; print(sys.prefix)')
    uv pip install --python "$venv/bin/python" --no-deps --reinstall "$work"/wheels/pynini-*.whl

# run mfa from tools/mfa with the Kaldi binaries on PATH (after build-mfa)
mfa *args:
    PATH="$PWD/tools/kaldi/dist/bin:$PATH" uv run --project tools/mfa --no-sync mfa {{args}}

# restore the Claude skills pinned in skills-lock.json (needs Node; experimental command)
skills:
    npx skills experimental_install
