"""``ThresholdSets`` — the Least Ambiguous set-valued Classifier (LAC).

Frozen definition (`Paper 1 - MS2 IMPLEMENTATION SPEC.md`, §C3 item 5):
"LAC: include class `k` iff `score_k ≥ 1-λ`". This is the classification
instantiation of the nested prediction-set family `C_λ` (Mathematical
Specification v2.1 D1, Algorithm Specification P-1): the per-class score is
a vector in `[0,1]` (e.g. a softmax/calibrated per-class probability), `λ`
ranges over `[0,1]`, and the inclusion threshold `1-λ` is non-increasing in
`λ`, so `C_λ` grows monotonically with `λ` (P-1 nesting).

At `λ=0` the threshold is `1`, so only classes scored at exactly `1.0` are
included (the empty set for any strictly-sub-certain score); at `λ=1` the
threshold is `0`, so every non-negative-scored class is included (the full
set) — the `λ_min`/`λ_max` empty/full-set boundaries the MS2 spec's edge
cases name explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.prediction_sets.base import PredictionSetConstructor

__all__ = ["ThresholdSets"]


class ThresholdSets(PredictionSetConstructor):
    """LAC prediction sets: `C_λ(x) = {k : score_k ≥ 1 - λ}`.

    `score` is a 1-D array of per-class scores in `[0, 1]`; `λ` must be in
    `[0, 1]` (the domain over which `1 - λ` is itself a valid score
    threshold).
    """

    def construct(self, score: ArrayLike, lam: float) -> NDArray[np.bool_]:
        """Build the LAC set `{k : score_k ≥ 1 - λ}`.

        Args:
            score: 1-D array of per-class scores, each in `[0, 1]`.
            lam: The threshold `λ`, must be in `[0, 1]`.

        Returns:
            A 1-D boolean array, same shape as `score`, `True` at index `k`
            iff `score_k ≥ 1 - λ`.

        Raises:
            ValueError: If `score` is not 1-D, contains a value outside
                `[0, 1]`, or `lam` is outside `[0, 1]`.
        """
        arr = np.asarray(score, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"score must be 1-D, got shape {arr.shape}")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("score must have every entry in [0, 1]")
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        return arr >= (1.0 - lam)

    def name(self) -> str:
        """Return the registry name ``"threshold"``."""
        return "threshold"
