"""Known-weight ambiguity family.

Frozen alternative branch (Algorithm Specification §7', "Known-weight
family (weights `w(.)`, `E_P[w]=1`)"): no dual, no split — the calibrator
replaces the empirical mean by the normalized weighted mean
`sum_i w_i L[i,lambda] / sum_i w_i` inside the threshold rule, using the
full calibration set. This module only encodes the known weights; the
weighted-CRC computation itself is `wfcrc.calibration`'s responsibility.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from wfcrc.ambiguity.base import AmbiguityFamily
from wfcrc.config.schema import FamilyType
from wfcrc.exceptions import FamilyError

__all__ = ["KnownWeightFamily"]


class KnownWeightFamily(AmbiguityFamily):
    """Known-weight ambiguity family with weights `w(X_i)`, `E_P[w] = 1` (A4)."""

    def __init__(self, weights: Sequence[float], *, mean_tol: float = 1e-6) -> None:
        """Initialize the known-weight family.

        Args:
            weights: One weight per calibration example, `w(X_i) >= 0`.
            mean_tol: Allowed absolute deviation of `mean(weights)` from
                `1.0` (A4 requires `E_P[w] = 1` exactly for a *known*
                weight function; a small tolerance only absorbs
                floating-point round-off in how the weights were computed
                upstream, not a relaxation of the assumption).

        Raises:
            FamilyError: If `weights` is empty, contains a non-finite or
                negative value, or its mean deviates from `1.0` by more
                than `mean_tol`.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.size == 0:
            raise FamilyError("known-weight family requires at least one weight")
        if not np.all(np.isfinite(w)):
            raise FamilyError("weights must be finite")
        if np.any(w < 0.0):
            raise FamilyError("weights must be nonnegative")
        mean_w = float(np.mean(w))
        if abs(mean_w - 1.0) > mean_tol:
            raise FamilyError(f"weights must satisfy E_P[w] = 1 (A4); got mean(weights) = {mean_w}")
        self._weights = w

    @property
    def family_type(self) -> FamilyType:
        """Return ``"known_weight"``."""
        return "known_weight"

    def weights(self) -> NDArray[np.float64]:
        """Return the known weights.

        Returns:
            A copy of the weight array (defensive: mutating the returned
            array does not affect this family's internal state).
        """
        return self._weights.copy()
