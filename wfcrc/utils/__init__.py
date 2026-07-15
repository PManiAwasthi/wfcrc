"""Deterministic engineering primitives shared by every wfcrc module.

Submodules:
    io: Canonical content hashing and crash-safe (atomic) serialization.
    numerics: Numerically stable scalar/array primitives (float64 only).
    seeds: Deterministic RNG fanout from a single global seed.
    logging: Structured, diffable JSON-lines run logging.
    cache: Generic content-addressed read-through cache built on ``io``.
    reproducibility: Git commit + environment fingerprint capture for
        run manifests.
"""

from __future__ import annotations
