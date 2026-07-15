# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows the milestone sequence defined by the frozen
Implementation Blueprint (MS1–MS5) rather than conventional semantic
versioning while pre-1.0.

## [0.1.0] — 2026-07-15 — MS1: core infrastructure

### Added

- `wfcrc.utils.io`: canonical content hashing (`content_hash`), atomic file
  writes, JSON/`.npz` array serialization, numpy-aware encoding.
- `wfcrc.utils.numerics`: numerically stable `logsumexp`,
  `weighted_logsumexp`, `clamp`, `safe_div`, `quantile` (float64 only).
- `wfcrc.utils.seeds`: deterministic RNG fanout (`set_global_seed`,
  `derive_seed`, `rng_for`) with no hidden global RNG mutation.
- `wfcrc.utils.logging`: structured, golden-diffable JSONL run logging.
- `wfcrc.utils.cache`: content-addressed, read-through cache built on
  `utils.io`.
- `wfcrc.utils.reproducibility`: git commit + environment fingerprint
  capture for future run manifests.
- `wfcrc.config`: typed, immutable, strictly validated, hashable, layered
  YAML configuration system (`load_config`, `Config.hash`/`.to_yaml`/`.get`).
- `wfcrc.exceptions`: structured exception hierarchy rooted at `WFCRCError`.
- Project infrastructure: `pyproject.toml` (Python 3.12+), Ruff, Black,
  MyPy (strict), Pytest + coverage, pre-commit, GitHub Actions CI, MkDocs
  documentation skeleton.
- `requirements/lock.txt`: fully pinned dependency closure for exact
  environment reproduction (`make install-locked` / `make lock`).
- 145 unit tests across `utils`/`config`, 100% line and branch coverage.

### Changed (post-audit cleanup, same milestone)

- `wfcrc.utils.cache.make_key` now uses the full, untruncated 256-bit
  SHA-256 digest (`CACHE_KEY_HASH_WIDTH`) instead of `content_hash`'s
  shorter default width, to keep cache-key collision probability
  negligible across long-running research sweeps.
- `wfcrc.utils.numerics.safe_div` now returns a `float64` scalar for
  scalar inputs (previously always returned a 0-d array), matching the
  scalar/array return convention already used by `logsumexp`, `clamp`,
  and `quantile`. No numerical behavior changed.
- Corrected the `Repository`/`repo_url` metadata in `pyproject.toml` and
  `mkdocs.yml` to the actual git remote.

### Documentation

- Added `docs/architecture.md`, `docs/configuration.md`,
  `docs/reproducibility.md`, `docs/getting-started.md`, and API reference
  pages.
- `docs/architecture.md` now documents the naming divergence between the
  frozen Implementation Blueprint's `src/wfcrc/{data,sets,losses,families,
  calibration,verify,metrics,viz,runner}` layout and this repository's
  actual (pre-existing, unmodified) flat layout and directory names.
- Added `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.

### Not implemented (explicitly out of MS1 scope)

- No mathematical logic: no ambiguity families, prediction sets, losses,
  calibration procedure, verification, metrics, visualization, or
  experiment runner. `make reproduce` exists only as a documented stub.
