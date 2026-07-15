# Getting started

## Requirements

- Python 3.12+
- (recommended) a dedicated virtual environment — conda, venv, or similar.

## Install

```bash
# Runtime + dev tooling (ruff, black, mypy, pytest, pre-commit)
pip install -e ".[dev]"

# Also install documentation tooling
pip install -e ".[dev,docs]"
```

## Verify the install

```bash
make test        # pytest, full suite + coverage
make lint         # ruff check + black --check
make typecheck    # mypy (strict mode)
```

Or directly with `pytest`/`ruff`/`black`/`mypy` if you are not using `make`
(e.g. on Windows without a `make` binary).

## Enable pre-commit hooks

```bash
pre-commit install
```

This runs ruff, black, mypy, and basic hygiene checks (trailing whitespace,
YAML/TOML validity, merge conflict markers) on every commit.

## Build the documentation locally

```bash
mkdocs serve
```
