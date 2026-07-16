"""Shared fixtures for loss-evaluator tests: synthetic nested mask sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def growing_sets(n_steps: int, shape: tuple[int, ...] = (4, 4)) -> list[NDArray[np.bool_]]:
    """A sequence of boolean masks that strictly grows (dilates) `n_steps` times.

    Element `i` includes all elements of element `i-1`. Simulates a
    `λ`-indexed set family that grows with `λ` (the pairing FNR/Miscoverage
    require), without depending on any real `PredictionSetConstructor`.
    """
    total = int(np.prod(shape))
    flat_order = np.arange(total)
    sets = []
    for step in range(n_steps):
        count = round(total * step / (n_steps - 1)) if n_steps > 1 else total
        mask = np.zeros(total, dtype=np.bool_)
        mask[flat_order[:count]] = True
        sets.append(mask.reshape(shape))
    return sets


def shrinking_sets(n_steps: int, shape: tuple[int, ...] = (4, 4)) -> list[NDArray[np.bool_]]:
    """A sequence of boolean masks that strictly shrinks (erodes) `n_steps` times.

    Element `i` is a subset of element `i-1`. Simulates a `λ`-indexed set
    family that shrinks with `λ` (the pairing FPR requires).
    """
    return list(reversed(growing_sets(n_steps, shape)))
