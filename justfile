# .env holds local secrets such as MDC_API_KEY (git-ignored)
set dotenv-load

default:
    just --list

# fetch submodules and install Python workspace
setup:
    git submodule update --init
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

# restore the Claude skills pinned in skills-lock.json (needs Node; experimental command)
skills:
    npx skills experimental_install
