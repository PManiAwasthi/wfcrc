"""Miscoverage loss.

Frozen definition (Research Vault, `05 Loss Functions/Miscoverage
Loss.md`): `l(C_λ(X), Y) = 1{ Y ∉ C_λ(X) }` — its expectation is the
miscoverage probability. Read set-valued for segmentation: `Y ∉ C_λ(X)`
means the predicted set does not fully contain the label, i.e.
`Y \\ C_λ(X) ≠ ∅` (at least one true-positive element is missed).

Monotonicity pairing: non-increasing in `λ` under the same **growing**
(dilating) set family FNR requires — a fully-covered label stays covered
as the set only grows. Building that paired set family is out of MS2 scope.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wfcrc.losses.base import LossEvaluator

__all__ = ["MiscoverageLoss"]


class MiscoverageLoss(LossEvaluator):
    """Miscoverage loss: `l(C_λ(X), Y) = 1{ Y ∉ C_λ(X) }`.

    A binary (0/1) indicator loss: ``1.0`` if any true-positive element of
    `Y` is missing from `predicted_set`, else ``0.0``.
    """

    def evaluate(self, predicted_set: NDArray[np.bool_], label: NDArray[np.bool_]) -> float:
        """Compute the miscoverage indicator of `predicted_set` against `label`.

        Args:
            predicted_set: Boolean array representing `C_λ(X)`.
            label: Boolean array representing `Y`, same shape.

        Returns:
            ``1.0`` if `Y ⊄ C_λ(X)` (some label element is missed), else
            ``0.0``. Always in `{0, 1} ⊆ [0, self.upper_bound()]`.

        Raises:
            ValueError: If the arrays are not boolean or shapes mismatch.
        """
        self._validate_shapes(predicted_set, label)
        missed = label & ~predicted_set
        return 1.0 if bool(np.any(missed)) else 0.0

    def upper_bound(self) -> float:
        """Return `B = 1.0` (miscoverage is a `{0,1}` indicator loss)."""
        return 1.0

    def name(self) -> str:
        """Return the registry name ``"miscoverage"``."""
        return "miscoverage"
