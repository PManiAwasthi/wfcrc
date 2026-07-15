# wfcrc

**Worst-case Family Conformal Risk Control** — a reproducible research repository.

Built incrementally against a frozen set of engineering specifications
(Framework Specification, Mathematical Specification v2.1, Algorithm
Specification, Implementation Blueprint, Experiment Blueprint). See
[`docs/`](docs/index.md) for full documentation.

## Milestone status

- **MS1 (this milestone): core infrastructure.** Deterministic hashing and
  atomic serialization (`utils.io`), numerically stable primitives
  (`utils.numerics`), seeded RNG fanout (`utils.seeds`), structured JSONL
  logging (`utils.logging`), a content-addressed cache (`utils.cache`), git
  commit / environment fingerprinting (`utils.reproducibility`), and a
  strict, typed, hashable, layered configuration system (`config`). **No
  mathematical logic** (no ambiguity families, losses, or calibration) is
  implemented yet — that is intentional; see
  [docs/architecture.md](docs/architecture.md).
- MS2–MS5: datasets, prediction sets, losses, ambiguity families,
  calibration, verification, metrics, visualization, experiment runner —
  not yet started.

## Quick start

```bash
pip install -e ".[dev]"
make test        # pytest (unit suite + coverage)
make lint         # ruff + black --check
make typecheck    # mypy --strict
```

See [docs/getting-started.md](docs/getting-started.md) for details, and
[docs/configuration.md](docs/configuration.md) /
[docs/reproducibility.md](docs/reproducibility.md) for how the configuration
and reproducibility systems work.

## Requirements

Python 3.12+. See [`pyproject.toml`](pyproject.toml) for pinned dependency
ranges.

## License

See [`LICENSE`](LICENSE).
