# wfcrc

**Worst-case Family Conformal Risk Control** — a reproducible research repository.

## Milestone status

This repository is built incrementally against a frozen set of engineering
specifications. The current milestone is **MS1: core infrastructure**.

| Milestone | Scope | Status |
|---|---|---|
| MS1 | `utils` (io, numerics, seeds, logging, cache), `config`, CI skeleton | ✅ complete |
| MS2 | datasets, prediction sets, losses, loss tables | not started |
| MS3 | ambiguity families, calibration | not started |
| MS4 | verification, metrics | not started |
| MS5 | visualization, experiment runner | not started |

MS1 contains **no mathematical logic** — no ambiguity families, no losses, no
calibration procedure. It exists to give every later milestone a
deterministic, reproducible, well-tested foundation: content-addressed
hashing and caching, stable numerics, seeded RNG fanout, structured logging,
and a strict, typed, hashable configuration system.

## Where to go next

- [Getting started](getting-started.md) — install the package and run the test suite.
- [Configuration](configuration.md) — how layered YAML configs are validated and hashed.
- [Reproducibility](reproducibility.md) — how determinism is guaranteed across MS1.
- [Architecture](architecture.md) — module dependency graph and design principles.
- API reference — generated from docstrings, in the navigation sidebar.
