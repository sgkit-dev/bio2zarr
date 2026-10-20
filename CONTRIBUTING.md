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

The required cross-repository BGEN suite lives in
`integration/test_bgen_roundtrip.py`, separately from the unit suite. It tests
the `vcztools` exporter against the `bio2zarr` importer without adding a runtime
dependency between the projects. Missing packages or required exporter APIs
are failures, not skipped tests.

Use a dedicated environment and an explicitly verified exporter wheel:

```bash
UV_PROJECT_ENVIRONMENT=.venv-integration uv sync --locked --group test --no-default-groups --python 3.12
uv pip install --python=.venv-integration '/absolute/path/to/verified-vcztools.whl[bgen]'
UV_PROJECT_ENVIRONMENT=.venv-integration uv run --no-sync python -m pytest integration -v --no-cov
```

Do not run `uv sync` between installing the exporter and running these tests:
it can remove the test-only exporter. The suite executes 24 actual CLI
roundtrips (four independent BGEN datasets, both Zarr versions, and 1/16/32-bit
probabilities), four full/subset roundtrips, an embedded `NA` identifier case,
and a sample-sidecar preflight preservation case. It checks quantization
bounds, independently derived dosage, metadata, missing/fill NaN bits, sample
order, and reuse of native `.metafile` and SQLite `.bgi` indexes.

The unreleased exporter must be installed from a verified wheel or immutable
source revision, never an invented registry version. Remote integration CI
requires a reachable, approved immutable exporter source or artifact; the
local wheel command above does not establish hosted-CI coverage.

The required `Required BGEN interoperability` CI job builds exporter commit
`8c7c06396471f7f47d812dfbd5e68c354a857190` from the approved
[`jpfeuffer/vcztools` fork](https://github.com/jpfeuffer/vcztools).
This is an immutable staging dependency for the exporter changes in
[sgkit-dev/vcztools#440](https://github.com/sgkit-dev/vcztools/pull/440),
not a merged or released exporter dependency.
It installs that wheel in a dedicated Python 3.12 environment with published
`bgen-limix` 4.15.0, explicitly checks required APIs, and rejects any result
other than 30 executed tests with zero skips, failures, or errors. The
`bgen-interoperability` artifact contains the built exporter wheel, its SHA-256,
the source commit, package versions, and the JUnit report. Ordinary unit CI
continues to run without the exporter.

The corresponding locally verified wheel is
`vcztools-0.2.1.dev9-cp312-cp312-macosx_15_0_arm64.whl` (SHA-256
`b5238f14b6f8bbe0fe1117c2164c090609608c3516e4010020972da72f119318`).

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
