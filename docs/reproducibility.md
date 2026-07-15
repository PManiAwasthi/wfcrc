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
collision. Writes are atomic
([`wfcrc.utils.io.atomic_write`][wfcrc.utils.io.atomic_write]): a crash
mid-write never leaves a partially written cache entry.

## 4. Structured, diffable run logs

[`wfcrc.utils.logging.get_logger`][wfcrc.utils.logging.get_logger] writes
one JSON-lines event per line, with a fixed field order and the timestamp
isolated in its own field — so two runs with identical inputs produce
byte-identical event streams once timestamps are stripped ("golden-diffable"),
letting CI compare log output across runs deterministically.

## Provenance capture (manifests)

[`wfcrc.utils.reproducibility`][wfcrc.utils.reproducibility] captures the
remaining piece of provenance — *what code and environment* produced a
result — via [`get_git_commit`][wfcrc.utils.reproducibility.get_git_commit]
and [`get_environment_fingerprint`][wfcrc.utils.reproducibility.get_environment_fingerprint].
Combined with a config hash and a seed, this is everything a future run
manifest needs to fully explain a result: *parameters* (config hash),
*randomness* (seed), and *code/environment* (git commit + fingerprint).

MS1 does not yet write run manifests or implement `make reproduce` end to
end — that lands with the experiment runner (later milestones). The
`reproduce` target exists today only as a documented stub in the `Makefile`.
