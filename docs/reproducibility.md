# Reproducibility

MS1 establishes four independent guarantees that every later milestone
builds on:

## 1. Deterministic hashing

[`wfcrc.utils.io.content_hash`][wfcrc.utils.io.content_hash] canonicalizes
any JSON-serializable object (sorted keys, fixed separators, ASCII-only,
numpy-aware) before hashing, so semantically equal objects — regardless of
dict key order, process, or platform — hash identically. This underlies
both [`Config.hash`][wfcrc.config.schema.Config.hash] and
[`wfcrc.utils.cache.make_key`][wfcrc.utils.cache.make_key].

## 2. Seeded, non-global RNG fanout

[`wfcrc.utils.seeds`][wfcrc.utils.seeds] derives independent, reproducible
`numpy.random.Generator` instances per named component from a single global
seed — without ever mutating numpy's process-global RNG state:

```python
from wfcrc.utils.seeds import set_global_seed, rng_for

set_global_seed(42)
split_rng = rng_for("calibration.split")
```

The same `(name, seed)` pair always derives the same seed
([`derive_seed`][wfcrc.utils.seeds.derive_seed]), so a run is bit-reproducible
given a fixed global seed. Project lint policy forbids bare `numpy.random.*`
calls anywhere outside this module — every stochastic component must go
through `rng_for`.

## 3. Content-addressed caching

[`wfcrc.utils.cache.Cache`][wfcrc.utils.cache.Cache] is a read-through cache
keyed by a hash of *all* of a computation's inputs
([`make_key`][wfcrc.utils.cache.make_key]), so a cache entry can never be
silently served for different inputs — a stale hit would require a hash
collision. `make_key` uses the full, untruncated 256-bit SHA-256 digest
(`CACHE_KEY_HASH_WIDTH`), not `content_hash`'s shorter human-facing default,
because cache keys accumulate across long-running research sweeps where a
truncated digest's collision probability is no longer negligible. Writes are
atomic ([`wfcrc.utils.io.atomic_write`][wfcrc.utils.io.atomic_write]): a
crash mid-write never leaves a partially written cache entry.

## 4. Structured, diffable run logs

[`wfcrc.utils.logging.get_logger`][wfcrc.utils.logging.get_logger] writes
one JSON-lines event per line, with a fixed field order and the timestamp
isolated in its own field — so two runs with identical inputs produce
byte-identical event streams once timestamps are stripped ("golden-diffable"),
letting CI compare log output across runs deterministically.

## 5. Pinned environment

`requirements/lock.txt` (repository root) is the fully pinned
dependency closure for `pip install -e ".[dev,docs]"` — every package at an
exact version, generated via `pip freeze --exclude-editable`. It is the
artifact that lets a specific past result be reproduced in the *exact*
environment that produced it, per the Implementation Blueprint's
reproducibility protocol (§17: "pinned environment") and the MS1
Implementation Specification (C7: "pyproject.toml + lockfile (pinned
deps)").

```bash
make install-locked   # reproduce the exact pinned environment
make lock              # regenerate the lockfile after a deliberate upgrade
```

`pyproject.toml` intentionally keeps *loose* compatible-release ranges
(`numpy>=1.26,<3`, etc.) rather than the lockfile's exact pins — this is a
deliberate split, not an inconsistency:

- **CI** installs from the ranges (`pip install -e ".[dev]"`). This is what
  actually catches a breaking upstream release early, which is the point of
  continuous integration.
- **The lockfile** is for exactly reproducing one specific environment on
  demand (e.g. to re-run an old experiment), not for CI's day-to-day
  regression signal.

The lockfile was generated on the primary development platform (Windows,
Python 3.12); a small number of pure-Python packages may resolve
differently on other platforms, which is why CI does not consume the
lockfile directly.

## Provenance capture (manifests)

[`wfcrc.utils.reproducibility`][wfcrc.utils.reproducibility] captures the
remaining piece of provenance — *what code and environment* produced a
result — via [`get_git_commit`][wfcrc.utils.reproducibility.get_git_commit]
and [`get_environment_fingerprint`][wfcrc.utils.reproducibility.get_environment_fingerprint].
Combined with a config hash and a seed, this is everything a run manifest
needs to fully explain a result: *parameters* (config hash), *randomness*
(seed), and *code/environment* (git commit + fingerprint).

`wfcrc.runner.runner.Manifest` (MS5) is the run manifest this section
anticipated: every `ExperimentRunner.run` call writes one atomically to
`<run_dir>/manifest.json`, combining a config-hash/seed/family fingerprint
with this module's git-commit and environment fingerprint, plus the
resulting calibration diagnostics and metrics. `make reproduce`
(`scripts/reproduce.py`) is no longer a stub: it re-runs a small, fixed-seed
synthetic reference experiment through `ExperimentRunner` and diffs its
manifest against the committed `tests/fixtures/reproduce_golden.json`
within a `1e-9` absolute tolerance — see
[Architecture](architecture.md#ms5-update-current-subpackage-status) and
`CLAIMS_TRACEABILITY.md` §12 item 10 for why the reference experiment is
synthetic rather than a real dataset.
