"""Deterministic RNG fanout from a single global seed.

The only stochastic component in the WF-CRC procedure is the A/B
calibration split; this module makes that split (and any other seeded
component) reproducible given a fixed global seed, without ever mutating
numpy's process-global RNG state. Callers must obtain a
:class:`numpy.random.Generator` from :func:`rng_for` rather than using bare
``numpy.random`` functions (enforced by project lint policy, not by this
module).
"""

from __future__ import annotations

import logging

import numpy as np

from wfcrc.constants import MAX_SEED, MIN_SEED
from wfcrc.exceptions import ReproducibilityError
from wfcrc.utils.io import content_hash

__all__ = ["derive_seed", "rng_for", "set_global_seed"]

_logger = logging.getLogger(__name__)

_SEED_MODULUS = MAX_SEED - MIN_SEED + 1

#: The process's current global seed, set via :func:`set_global_seed`.
_global_seed: int | None = None


def _validate_seed(seed: int, *, arg_name: str = "seed") -> None:
    """Validate that ``seed`` is a plain, in-range integer.

    Args:
        seed: Candidate seed value.
        arg_name: Name to use in the raised error message.

    Raises:
        ReproducibilityError: If ``seed`` is not an ``int`` (``bool`` is
            rejected despite being a subclass of ``int``) or is outside
            ``[MIN_SEED, MAX_SEED]``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReproducibilityError(f"{arg_name} must be an int, got {type(seed).__name__}")
    if not (MIN_SEED <= seed <= MAX_SEED):
        raise ReproducibilityError(f"{arg_name} must be in [{MIN_SEED}, {MAX_SEED}], got {seed}")


def set_global_seed(seed: int) -> None:
    """Set the process-wide seed that :func:`rng_for` derives component seeds from.

    This does **not** touch numpy's global RNG state (no ``np.random.seed``
    call) — it only records the seed for later use by :func:`derive_seed`
    and :func:`rng_for`.

    Args:
        seed: Non-negative integer seed.

    Raises:
        ReproducibilityError: If ``seed`` is not a valid integer seed.
    """
    _validate_seed(seed)
    global _global_seed
    _global_seed = seed
    _logger.info("global seed set to %d", seed)


def derive_seed(name: str, base: int) -> int:
    """Deterministically derive a component-specific seed from ``(name, base)``.

    The same ``(name, base)`` pair always yields the same derived seed
    (within and across processes); different names yield independent seeds
    with high probability (hash-based, collision-resistant).

    Args:
        name: Identifier of the component requesting a seed (e.g.
            ``"calibration.split"``).
        base: The base seed to derive from (typically the global seed).

    Returns:
        A derived integer seed in ``[MIN_SEED, MAX_SEED]``.

    Raises:
        ReproducibilityError: If ``name`` is empty or ``base`` is not a
            valid integer seed.
    """
    _validate_seed(base, arg_name="base")
    if not name:
        raise ReproducibilityError("name must be a non-empty string")

    digest = content_hash({"name": name, "base": base}, width=16)
    derived = int(digest, 16) % _SEED_MODULUS + MIN_SEED
    _logger.info("derived seed for '%s' (base=%d): %d", name, base, derived)
    return derived


def rng_for(name: str) -> np.random.Generator:
    """Return a freshly seeded, independent RNG for the named component.

    Args:
        name: Identifier of the component requesting an RNG (e.g.
            ``"calibration.split"``).

    Returns:
        A :class:`numpy.random.Generator` seeded deterministically from the
        current global seed and ``name``.

    Raises:
        ReproducibilityError: If :func:`set_global_seed` has not been called
            yet, or ``name`` is empty.
    """
    if _global_seed is None:
        raise ReproducibilityError("global seed not set; call set_global_seed() first")
    derived = derive_seed(name, _global_seed)
    return np.random.default_rng(derived)
