# Contributing to bio2zarr

## Development setup

We use [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/sgkit-dev/bio2zarr.git
cd bio2zarr
uv sync --group dev
```

## Running tests

```bash
uv run pytest
```

### BGEN exporter interoperability

With a compatible `vcztools` exporter installed in the test environment, run
the optional cross-repository BGEN suite explicitly:

```bash
uv run --no-sync python -m pytest integration -v --no-cov
```

## Linting

We use [prek](https://github.com/prek-dev/prek) for pre-commit linting,
configured in `prek.toml`. Install it as a pre-commit hook:

```bash
uv run prek install
```

Run all checks manually:

```bash
uv run --only-group=lint prek -c prek.toml run --all-files
```

If local results differ from CI, run `uv run prek cache clean`.

## Building documentation

```bash
cd docs && make
```

## Pull requests

- Create a branch from `main`
- Ensure all CI checks pass
- Add a changelog entry if appropriate
