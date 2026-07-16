"""``PredictionSetConstructor`` — the abstract nested prediction-set contract.

Per the Mathematical Specification v2.1 (D1) and the Algorithm Specification
(P-1): a nested prediction-set family `{C_λ}` satisfies
`λ ≤ λ' ⇒ C_λ(x) ⊆ C_{λ'}(x)`. Per the Implementation Blueprint (§6) and the
MS2 Implementation Specification (C3/M5), a `PredictionSetConstructor`
builds `C_λ(x)` from a per-example score and exposes a shared,
score-agnostic nestedness contract check.

This module fixes only that contract; concrete constructions
(:class:`~wfcrc.prediction_sets.classification.ThresholdSets`,
:class:`~wfcrc.prediction_sets.segmentation.MorphologicalSets`) live in
their own modules.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["PredictionSetConstructor"]


class PredictionSetConstructor(ABC):
    """Abstract base class for a nested prediction-set family `C_λ(x)`.

    Concrete subclasses must implement :meth:`construct` and :meth:`name`;
    :meth:`assert_nested` is a shared, concrete contract-check usable by any
    subclass (Implementation Blueprint §6: `assert_nested(grid)->bool`).
    """

    @abstractmethod
    def construct(self, score: ArrayLike, lam: float) -> NDArray[np.bool_]:
        """Build `C_λ(x)` from a per-example score at one threshold `λ`.

        Args:
            score: The per-example quantity this constructor builds `C_λ`
                from (e.g. a per-class score vector for
                :class:`~wfcrc.prediction_sets.classification.ThresholdSets`,
                or a boolean seed mask for
                :class:`~wfcrc.prediction_sets.segmentation.MorphologicalSets`).
                Arbitrary shape; dimension-agnostic.
            lam: The threshold `λ`.

        Returns:
            `C_λ(x)` as a boolean array, the same shape as (a boolean cast
            of) `score`.

        Raises:
            ValueError: If `score`/`lam` are invalid for this constructor
                (e.g. wrong dtype, or `lam` outside this constructor's
                supported domain).
            wfcrc.exceptions.SetConstructionError: If this constructor's
                configuration names a construction the frozen specification
                does not define a formula for.
        """

    @abstractmethod
    def name(self) -> str:
        """Return this constructor's short registry name (e.g. ``"threshold"``).

        Returns:
            A short, stable, lowercase identifier for this constructor.
        """

    def assert_nested(self, score: ArrayLike, lambda_grid: ArrayLike, *, tol: float = 0.0) -> bool:
        """Check that `{C_λ}` is nested (non-decreasing set inclusion) over `lambda_grid`.

        This is the P-1 contract check (Algorithm Specification §5, §20):
        for a strictly increasing `λ_1 < λ_2 < ... < λ_T`, construct
        `C_{λ_1}(x), ..., C_{λ_T}(x)` and verify
        `C_{λ_j}(x) ⊆ C_{λ_{j+1}}(x)` for every consecutive pair.

        Args:
            score: The per-example quantity to build `C_λ` from (see
                :meth:`construct`).
            lambda_grid: Strictly increasing `λ`-grid to check nesting
                across.
            tol: Reserved for subclasses that construct from continuous
                (non-boolean) intermediate state and need a numerical
                tolerance; the base boolean-subset check itself is exact
                and ignores `tol`.

        Returns:
            ``True`` if every consecutive pair is nested, ``False``
            otherwise.

        Raises:
            ValueError: If `lambda_grid` is empty or not strictly
                increasing.
        """
        del tol  # exact boolean-subset check; no tolerance needed at this level
        grid = np.asarray(lambda_grid, dtype=np.float64)
        if grid.size == 0:
            raise ValueError("lambda_grid must be non-empty")
        if grid.size > 1 and not np.all(np.diff(grid) > 0):
            raise ValueError("lambda_grid must be strictly increasing")

        sets = [self.construct(score, float(lam)) for lam in grid]
        return all(np.all(prev <= curr) for prev, curr in itertools.pairwise(sets))
