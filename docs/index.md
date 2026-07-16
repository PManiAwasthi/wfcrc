# wfcrc

**Worst-case Family Conformal Risk Control** — a reproducible research repository.

## Milestone status

This repository is built incrementally against a frozen set of engineering
specifications. **MS1-MS5 are complete and frozen** (see `CLAIMS_TRACEABILITY.md`
and `PROJECT_CONTEXT.md` for the exact scope of each and every disclosed
deviation from the frozen specifications).

| Milestone | Scope | Status |
|---|---|---|
| MS1 | `utils` (io, numerics, seeds, logging, cache), `config`, CI skeleton | ✅ complete |
| MS2 | ambiguity families, losses, calibration core | ✅ complete |
| MS3 | prediction sets, verification, calibration pipeline | ✅ complete |
| MS4 | datasets contracts, metrics, experiment execution | ✅ complete |
| MS5 | visualization (figures), experiment runner, `make reproduce` | ✅ complete |

MS1 contains **no mathematical logic** — no ambiguity families, no losses, no
calibration procedure. It exists to give every later milestone a
deterministic, reproducible, well-tested foundation: content-addressed
hashing and caching, stable numerics, seeded RNG fanout, structured logging,
and a strict, typed, hashable configuration system. MS2-MS5 build the
worst-case-over-family conformal risk control procedure itself, its
verification/metrics/visualization layers, and a config-driven runner with
checkpointing, sweeps, resume, and a golden-file reproducibility harness —
on top of that foundation.

## Where to go next

- [Getting started](getting-started.md) — install the package and run the test suite.
- [Configuration](configuration.md) — how layered YAML configs are validated and hashed.
- [Reproducibility](reproducibility.md) — how determinism is guaranteed, from MS1's hashing/seeding/caching primitives through MS5's run manifests and `make reproduce`.
- [Architecture](architecture.md) — module dependency graph and design principles.
- API reference — generated from docstrings, in the navigation sidebar.
