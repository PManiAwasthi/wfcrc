"""False Negative Rate (FNR) loss.

Frozen definition (Research Vault, `05 Loss Functions/False Negative Rate
Loss.md`): `l = 1 - |Y ∩ C_λ(X)| / |Y|` — the fraction of true-positive
elements of `Y` missed by the predicted set. Equivalently
`l = |Y \\ C_λ(X)| / |Y|`.

Monotonicity pairing: this loss is non-increasing in `λ` when paired with a
set family that **grows** (dilates) as `λ` increases — the frozen source
notes "non-increasing in λ (bigger sets miss fewer)". Building that
paired set family (`sets`/`prediction_sets`) is out of MS2 scope.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wfcrc.losses.base import LossEvaluator

__all__ = ["FNRLoss"]


class FNRLoss(LossEvaluator):
    """False Negative Rate loss: `l = 1 - |Y ∩ C_λ(X)| / |Y|`.

    Convention for an empty label (`|Y| = 0`, no true-positive elements):
    the loss is defined as ``0.0`` — vacuously, there is nothing to miss.
    This convention is not stated numerically in the frozen source and is
    the standard, unambiguous reading of "fraction missed" when the
    denominator is zero.
    """

    def evaluate(self, predicted_set: NDArray[np.bool_], label: NDArray[np.bool_]) -> float:
        """Compute the false negative rate of `predicted_set` against `label`.

        Args:
            predicted_set: Boolean array representing `C_λ(X)`.
            label: Boolean array representing `Y`, same shape.

        Returns:
            `|Y \\ C_λ(X)| / |Y|`, or ``0.0`` if `|Y| = 0`. Always in
            `[0, 1] = [0, self.upper_bound()]`.

        Raises:
            ValueError: If the arrays are not boolean or shapes mismatch.
        """
        self._validate_shapes(predicted_set, label)
        num_positive = int(np.sum(label))
        if num_positive == 0:
            return 0.0
        num_missed = int(np.sum(label & ~predicted_set))
        return num_missed / num_positive

    def upper_bound(self) -> float:
        """Return `B = 1.0` (FNR is a rate loss)."""
        return 1.0

    def name(self) -> str:
        """Return the registry name ``"fnr"``."""
        return "fnr"
