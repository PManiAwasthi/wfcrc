"""``ThresholdSearch`` — monotone binary search for the WF-CRC threshold.

Per the Algorithm Specification (§7 step 7, §8): "`g` is non-increasing in
`λ`; `λ̂ = min{λ ∈ Λ : g(λ) <= alpha}` by binary search; if none, `λ̂ = λ_max`
(empty-selection flag)" (F-1, Algorithm Spec §14).

This module assumes — does not verify — that `g` is non-increasing over
`grid` (that precondition is checked upstream, e.g. via
:meth:`wfcrc.losses.base.LossEvaluator.assert_monotone` on the loss table
that produced `g`); verifying it here would require evaluating `g`
everywhere, defeating the point of a binary search.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["ThresholdSearch"]


class ThresholdSearch:
    """Binary search for the smallest `λ` on a grid with `g(λ) <= alpha`."""

    def search(
        self,
        g: Callable[[float], float],
        grid: NDArray[np.float64],
        alpha: float,
        default: float,
    ) -> float:
        """Find `λ̂ = min{λ ∈ grid : g(λ) <= alpha}`, or `default` if none.

        Args:
            g: A criterion function, assumed non-increasing over `grid`.
            grid: Strictly increasing `λ`-grid to search over.
            alpha: Target risk level.
            default: Value to return if no grid point satisfies
                `g(λ) <= alpha` (conventionally `λ_max = grid[-1]`, per
                Algorithm Spec F-1's empty-selection fallback).

        Returns:
            The smallest `λ ∈ grid` with `g(λ) <= alpha`, or `default` if
            the criterion is infeasible even at the most conservative grid
            point.

        Raises:
            ValueError: If `grid` is empty or not strictly increasing.
        """
        if grid.size == 0:
            raise ValueError("grid must be non-empty")
        if not np.all(np.diff(grid) > 0):
            raise ValueError("grid must be strictly increasing")

        if g(float(grid[-1])) > alpha:
            return default

        lo, hi = 0, grid.size - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if g(float(grid[mid])) <= alpha:
                hi = mid
            else:
                lo = mid + 1
        return float(grid[lo])
