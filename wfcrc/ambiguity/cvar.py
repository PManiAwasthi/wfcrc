"""CVaR (superquantile) ambiguity family.

Frozen closed-form dual (Algorithm Specification §7, "CVaR (β)"):

    theta = eta;  c(eta) = eta;  t(z; eta) = (1/beta) * (z - eta)_+

`theta_hat_A(lambda) = eta_hat`, the `(1-beta)`-quantile of
`{L[i,lambda] : i in A}` — a closed-form argmin, no iterative solve
required (this is the standard CVaR dual: the minimizing `eta` of
`eta + (1/beta) E[(Z-eta)_+]` is exactly the `(1-beta)`-quantile of `Z`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.ambiguity.base import DualAmbiguityFamily
from wfcrc.config.schema import FamilyType
from wfcrc.exceptions import FamilyError
from wfcrc.utils.numerics import quantile

__all__ = ["CVaRFamily"]


class CVaRFamily(DualAmbiguityFamily):
    """CVaR ambiguity family with tail parameter `beta`.

    Attributes:
        beta: CVaR tail parameter, `0 < beta < 1` (smaller `beta` is a
            larger, more conservative ambiguity set).
    """

    def __init__(self, beta: float) -> None:
        """Initialize the CVaR family.

        Args:
            beta: CVaR tail parameter, must satisfy `0 < beta < 1`.

        Raises:
            FamilyError: If `beta` is outside `(0, 1)`.
        """
        if not (0.0 < beta < 1.0):
            raise FamilyError(f"CVaR beta must satisfy 0 < beta < 1, got {beta}")
        self.beta = beta

    @property
    def family_type(self) -> FamilyType:
        """Return ``"cvar"``."""
        return "cvar"

    def estimate_dual(self, losses_a_col: NDArray[np.float64]) -> float:
        """Estimate `eta_hat`, the `(1 - beta)`-quantile of `losses_a_col`.

        Args:
            losses_a_col: `{L[i,lambda] : i in A}` for a single `lambda`.

        Returns:
            `eta_hat` (a plain `float`; CVaR's dual parameter is scalar).

        Raises:
            FamilyError: If `losses_a_col` is empty.
            ValueError: If `losses_a_col` contains NaN/inf (propagated from
                :func:`wfcrc.utils.numerics.quantile`).
        """
        if len(losses_a_col) == 0:
            raise FamilyError("cannot estimate CVaR dual from an empty loss column")
        return float(quantile(losses_a_col, q=1.0 - self.beta))

    def c(self, theta: Any) -> float:
        """Return `c(eta) = eta`.

        Args:
            theta: `eta_hat`, as returned by :meth:`estimate_dual`.

        Returns:
            `eta`.
        """
        return float(theta)

    def t(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
        """Return `t(z; eta) = (1/beta) * (z - eta)_+`.

        Args:
            z: Scalar or array of loss values.
            theta: `eta_hat`, as returned by :meth:`estimate_dual`.

        Returns:
            `(1/beta) * max(z - eta, 0)`, same shape as `z`.
        """
        eta = float(theta)
        z_arr = np.asarray(z, dtype=np.float64)
        return np.maximum(z_arr - eta, 0.0) / self.beta
