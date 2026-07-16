"""``AmbiguityFamily`` — the abstract contract encoding a worst-case family `Q`.

Per the Mathematical Specification v2.1 (A5, D4) and the Algorithm
Specification (§7, §7'), an ambiguity family is either:

- a **dual family** (CVaR, KL) admitting a finite-dimensional convex dual
  representation `sup_{Q∈Q} E_Q[L] = inf_theta {c(theta) + E_P[t(L; theta)]}`
  (A5), exposing `estimate_dual`/`c`/`t`/`transform`/`btil`; or
- an **alternative-branch family** (finite-group, known-weight) with no
  dual — it exposes only the raw structure (`groups()`/`weights()`) that
  :mod:`wfcrc.calibration` uses to run per-group or weighted conformal risk
  control directly (Algorithm Spec §7').

This module fixes only the contract; all mathematics (dual estimation,
transforms) lives in the concrete family modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.config.schema import FamilyType
from wfcrc.exceptions import FamilyError

__all__ = ["AmbiguityFamily", "DualAmbiguityFamily"]


class AmbiguityFamily(ABC):
    """Minimal identity contract shared by every ambiguity family.

    Every concrete family (dual or alternative-branch) exposes
    `family_type`, matching the frozen supported set
    `{"cvar", "kl", "finite_group", "known_weight"}`
    (:data:`wfcrc.config.schema.FamilyType`).
    """

    @property
    @abstractmethod
    def family_type(self) -> FamilyType:
        """Return this family's type tag.

        Returns:
            One of ``"cvar"``, ``"kl"``, ``"finite_group"``,
            ``"known_weight"``.
        """


class DualAmbiguityFamily(AmbiguityFamily):
    """Shared contract for families with a finite-dimensional convex dual (A5).

    Concrete subclasses (:class:`~wfcrc.ambiguity.cvar.CVaRFamily`,
    :class:`~wfcrc.ambiguity.kl.KLFamily`) implement :meth:`estimate_dual`,
    :meth:`c`, and :meth:`t`; :meth:`transform` and :meth:`btil` are
    concrete, shared compositions of those three (Algorithm Spec §7:
    `L̃[i,λ] = c(theta) + t(L[i,λ]; theta)`, `B̃ = c(theta) + t(B; theta)`).

    The dual parameter `theta` is intentionally opaque (`Any`) at this
    level — its shape is family-specific (a scalar for CVaR, a 2-tuple for
    KL) — matching the Implementation Blueprint's own abstract API.
    """

    @abstractmethod
    def estimate_dual(self, losses_a_col: NDArray[np.float64]) -> Any:
        """Estimate the dual parameter `theta` from block-A losses at one `λ`.

        Args:
            losses_a_col: `{L[i,λ] : i ∈ A}` for a single `λ` — the
                dual-estimation block's loss column.

        Returns:
            The family-specific dual parameter `theta` (opaque to callers).

        Raises:
            FamilyError: If the dual cannot be estimated (e.g. an empty
                input column).
        """

    @abstractmethod
    def c(self, theta: Any) -> float:
        """Evaluate the dual's constant term `c(theta)`.

        Args:
            theta: A dual parameter previously returned by
                :meth:`estimate_dual`.

        Returns:
            `c(theta)`.
        """

    @abstractmethod
    def t(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
        """Evaluate the dual's transform `t(z; theta)`, elementwise over `z`.

        Args:
            z: Scalar or array of loss values (or the loss bound `B`).
            theta: A dual parameter previously returned by
                :meth:`estimate_dual`.

        Returns:
            `t(z; theta)`, same shape as `z`.
        """

    def transform(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
        """Compute the dual-transformed loss `c(theta) + t(z; theta)`.

        Args:
            z: Scalar or array of loss values (or the loss bound `B`).
            theta: A dual parameter previously returned by
                :meth:`estimate_dual`.

        Returns:
            The transformed value(s), same shape as `z`.

        Raises:
            FamilyError: If any resulting value is non-finite (F-3,
                Algorithm Spec §14: unbounded transform).
        """
        result = np.asarray(self.c(theta) + self.t(z, theta), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise FamilyError(
                f"unbounded transform: transform(z, theta) produced a non-finite "
                f"value for theta={theta!r}"
            )
        return result

    def btil(self, theta: Any, loss_bound: float) -> float:
        """Compute `B̃(theta) = c(theta) + t(B; theta)`, the transformed-loss bound.

        Args:
            theta: A dual parameter previously returned by
                :meth:`estimate_dual`.
            loss_bound: `B`, the untransformed loss's upper bound.

        Returns:
            `B̃(theta)`, a finite scalar (transform is non-decreasing in
            `z`, so this is evaluated at `z = B`, per Algorithm Spec §7).

        Raises:
            FamilyError: If the transform is unbounded at `z = B`.
        """
        return float(self.transform(loss_bound, theta))
