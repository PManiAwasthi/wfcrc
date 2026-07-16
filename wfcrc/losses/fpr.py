"""False Positive Rate (FPR) loss.

**Provenance note (documented gap-fill, not an invented formula):** the
frozen sources name this loss and describe it only informally — the MS2
Implementation Specification (`Paper 1 - MS2 IMPLEMENTATION SPEC.md`, §C4)
says "FPR/accepted-FP proportion" — but no document in the Research Vault
spells out a closed-form expression (unlike FNR and Miscoverage, which are
given exact formulas in `05 Loss Functions/`). The formula implemented here
is the standard, textbook False Positive Rate — `FP / (FP + TN)` — applied
to the same `(predicted_set, label)` set representation the frozen FNR loss
uses, and it is the unique reading consistent with "accepted-FP proportion":
`l = |C_λ(X) \\ Y| / |Y^c|`, the fraction of true-negative elements wrongly
accepted into the predicted set. This mirrors the frozen FNR formula's
structure exactly (`|Y \\ C_λ(X)| / |Y|`) with the roles of `Y` and its
complement `Y^c` swapped.

Monotonicity pairing: unlike FNR, this loss is non-increasing in `λ` only
when paired with a set family that **shrinks** (erodes) as `λ` increases —
a smaller accepted set can only exclude more true negatives, never fewer.
Pairing this loss with a *growing* set family (as FNR requires) would make
it non-decreasing, violating A2/P-2; building the correct paired set family
is out of MS2 scope (see `wfcrc.losses.base.LossEvaluator.assert_monotone`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wfcrc.losses.base import LossEvaluator

__all__ = ["FPRLoss"]


class FPRLoss(LossEvaluator):
    """False Positive Rate loss: `l = |C_λ(X) \\ Y| / |Y^c|`.

    Convention for an all-positive label (`|Y^c| = 0`, no true-negative
    elements): the loss is defined as ``0.0`` — vacuously, there is nothing
    to falsely include.
    """

    def evaluate(self, predicted_set: NDArray[np.bool_], label: NDArray[np.bool_]) -> float:
        """Compute the false positive rate of `predicted_set` against `label`.

        Args:
            predicted_set: Boolean array representing `C_λ(X)`.
            label: Boolean array representing `Y`, same shape.

        Returns:
            `|C_λ(X) \\ Y| / |Y^c|`, or ``0.0`` if `|Y^c| = 0`. Always in
            `[0, 1] = [0, self.upper_bound()]`.

        Raises:
            ValueError: If the arrays are not boolean or shapes mismatch.
        """
        self._validate_shapes(predicted_set, label)
        num_negative = int(np.sum(~label))
        if num_negative == 0:
            return 0.0
        num_false_positive = int(np.sum(~label & predicted_set))
        return num_false_positive / num_negative

    def upper_bound(self) -> float:
        """Return `B = 1.0` (FPR is a rate loss)."""
        return 1.0

    def name(self) -> str:
        """Return the registry name ``"fpr"``."""
        return "fpr"
