"""KL-divergence (f-divergence ball) ambiguity family.

Frozen dual (Algorithm Specification §7, "f-divergence (KL ball, radius
rho)"):

    theta = (eta, mu), eta > 0;  c(eta,mu) = mu + eta*rho
    t(z; eta,mu) = eta * (exp((z - mu)/eta) - 1)
    theta_hat_A(lambda) = argmin_{eta>0,mu} [ c + (1/n_A) sum_A t ]

`mu` is profiled out in closed form via log-sum-exp (Algorithm Spec §7,
§13), reducing the 2-D solve to a 1-D convex minimization over `eta`:

    mu*(eta) = eta * ( logsumexp(z/eta) - log(n_A) )
    h(eta)   = eta * ( logsumexp(z/eta) - log(n_A) + rho )   [minimize over eta > 0]

(Standard KL-DRO duality: `sup_{Q: KL(Q||P)<=rho} E_Q[Z]
= inf_{eta>0} { eta*log(E_P[exp(Z/eta)]) + eta*rho }`; substituting the
first-order condition for the profiled `mu` gives `h` above.) `eta` is
clamped to `>= eta_min > 0` (Algorithm Spec §13) to avoid `exp` overflow
as `eta -> 0`; the 1-D minimization uses :func:`wfcrc.utils.numerics.logsumexp`
for numerical stability and a dependency-free bracket-and-golden-section
search (no new numerical library — `h` is strictly convex on `(0, infinity)`
given `rho > 0`, so a plain unimodal-function minimizer is exact up to
`tol`, matching the "1-D convex solve" the Algorithm Spec calls for; this
is an implementation-detail solver choice, not a change to the mathematics
it solves).

**Fixed-eta fallback (Algorithm Spec §15, F-4 "dual solver non-convergence"):**
on a degenerate/near-constant `losses_a_col` (including the `n_A=1`
singleton case), `h` is minimized strictly at the `eta_min` boundary
(`h(eta) = c + eta*rho` there, monotone increasing) rather than at a
genuine interior optimum. Per §15 ("use a fixed data-independent eta;
still valid by weak duality; more conservative sets"), when this boundary
condition is detected, `estimate_dual` discards the data-driven `eta` and
substitutes `fallback_eta` (a fixed constant, not derived from the data or
even from `B` — a genuine constructor-time hyperparameter, defaulting to
`1.0`, the loss bound shared by every frozen concrete loss in
:mod:`wfcrc.losses`). `mu` is still computed via the *same* closed-form
profile step at that fixed `eta` (Math Spec §5: weak duality holds for
*any* `eta > 0` paired with its closed-form-optimal `mu`, so fixing `eta`
alone — without also fixing `mu` — preserves validity while keeping the
bound as tight as a fixed `eta` allows). This is not a new optimization
method: the existing golden-section solver is unchanged; only the
post-hoc substitution at its boundary result is added.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.ambiguity.base import DualAmbiguityFamily
from wfcrc.config.schema import FamilyType
from wfcrc.constants import DEFAULT_ETA_MIN
from wfcrc.exceptions import FamilyError
from wfcrc.utils.numerics import logsumexp

__all__ = ["KLDualParams", "KLFamily"]

#: Golden-section search convergence tolerance on the eta interval width.
_GOLDEN_TOL = 1e-10
#: Maximum golden-section iterations.
_GOLDEN_MAX_ITER = 200
#: Geometric growth factor used when expanding the bracket search.
_BRACKET_GROWTH = 2.0
#: Maximum bracket-expansion doublings before falling back to the last interval.
_BRACKET_MAX_EXPAND = 200
#: 1/golden ratio, used by the golden-section search.
_INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class KLDualParams:
    """The KL family's 2-parameter dual variable `theta = (eta, mu)`.

    Attributes:
        eta: The KL dual's scale parameter, `eta > 0`.
        mu: The profiled-out shift parameter.
    """

    eta: float
    mu: float


def _bracket_and_minimize(h: Callable[[float], float], lo: float) -> float:
    """Minimize a convex, unimodal `h` over `[lo, infinity)`.

    Args:
        h: A convex function, finite at `lo`.
        lo: Hard lower bound on the search domain (`eta_min`), `lo > 0`.

    Returns:
        `argmin_{x >= lo} h(x)`, accurate to :data:`_GOLDEN_TOL`.
    """
    # The first probe point MUST be adjacent to lo (not some fixed point
    # like 1.0): a convex function's values at two DISTANT points say
    # nothing about monotonicity in between (h(lo) <= h(far) does not
    # imply h is non-decreasing on [lo, far] -- it could dip well below
    # h(lo) in between and still recover to h(far) >= h(lo) at the far
    # end). Only a comparison against an adjacent point is a valid local
    # monotonicity check under convexity, which then extends globally.
    a, h_a = lo, h(lo)
    b, h_b = lo * _BRACKET_GROWTH, h(lo * _BRACKET_GROWTH)
    if h_b >= h_a:
        # By convexity, h is non-decreasing on all of [lo, infinity):
        # the constrained minimum is at the boundary lo.
        return lo

    for _ in range(_BRACKET_MAX_EXPAND):
        c = b * _BRACKET_GROWTH
        h_c = h(c)
        if h_c >= h_b:
            return _golden_section_minimize(h, a, c)
        a, h_a = b, h_b
        b, h_b = c, h_c

    # Fallback: rho too small / h too flat to bracket within the expansion
    # budget. Search the last known-decreasing interval.
    return _golden_section_minimize(h, a, b)


def _golden_section_minimize(h: Callable[[float], float], lo: float, hi: float) -> float:
    """Golden-section search for the minimizer of a unimodal `h` on `[lo, hi]`.

    Args:
        h: A unimodal function on `[lo, hi]`.
        lo: Interval lower bound.
        hi: Interval upper bound.

    Returns:
        The interval midpoint after convergence (or after
        :data:`_GOLDEN_MAX_ITER` iterations), approximating `argmin h`.
    """
    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    h_c, h_d = h(c), h(d)
    for _ in range(_GOLDEN_MAX_ITER):
        if (b - a) < _GOLDEN_TOL:
            break
        if h_c < h_d:
            b, d, h_d = d, c, h_c
            c = b - _INV_PHI * (b - a)
            h_c = h(c)
        else:
            a, c, h_c = c, d, h_d
            d = a + _INV_PHI * (b - a)
            h_d = h(d)
    return (a + b) / 2.0


class KLFamily(DualAmbiguityFamily):
    """KL-divergence-ball ambiguity family with radius `rho`.

    Attributes:
        rho: KL-ball radius, `rho > 0`.
        eta_min: Lower clamp bound on the dual scale parameter `eta`
            (Algorithm Spec §13); defaults to
            :data:`wfcrc.constants.DEFAULT_ETA_MIN`.
        fallback_eta: The fixed, data-independent `eta` substituted when
            dual estimation hits the `eta_min` boundary (Algorithm Spec
            §15, F-4); defaults to `1.0`.
    """

    def __init__(
        self,
        rho: float,
        *,
        eta_min: float = DEFAULT_ETA_MIN,
        fallback_eta: float = 1.0,
    ) -> None:
        """Initialize the KL family.

        Args:
            rho: KL-ball radius, must satisfy `rho > 0`.
            eta_min: Lower clamp bound on `eta`, must satisfy `eta_min > 0`.
            fallback_eta: Fixed `eta` used by the §15 fallback when dual
                estimation hits the `eta_min` boundary; must satisfy
                `fallback_eta > 0`. Chosen independently of any
                calibration data (a deployment-time constant, analogous to
                `eta_min`), per §15's "fixed data-independent eta".

        Raises:
            FamilyError: If `rho <= 0`, `eta_min <= 0`, or
                `fallback_eta <= 0`.
        """
        if not (rho > 0.0):
            raise FamilyError(f"KL rho must satisfy rho > 0, got {rho}")
        if not (eta_min > 0.0):
            raise FamilyError(f"eta_min must be > 0, got {eta_min}")
        if not (fallback_eta > 0.0):
            raise FamilyError(f"fallback_eta must be > 0, got {fallback_eta}")
        self.rho = rho
        self.eta_min = eta_min
        self.fallback_eta = fallback_eta

    @property
    def family_type(self) -> FamilyType:
        """Return ``"kl"``."""
        return "kl"

    def estimate_dual(self, losses_a_col: NDArray[np.float64]) -> KLDualParams:
        """Estimate `theta_hat = (eta_hat, mu_hat)` by 1-D convex minimization.

        If the minimizer lands exactly at the `eta_min` boundary — which,
        by convexity, happens if and only if `h` is non-decreasing on all
        of `[eta_min, infinity)` (a degenerate/near-constant `losses_a_col`,
        including the `n_A=1` singleton case) — this is treated as dual
        non-convergence (F-4) and the §15 fixed-`eta` fallback is applied:
        `eta_hat` is replaced by `self.fallback_eta` and `mu_hat` is
        recomputed via the same closed-form profile step at that fixed
        `eta`.

        Args:
            losses_a_col: `{L[i,lambda] : i in A}` for a single `lambda`.

        Returns:
            The estimated :class:`KLDualParams`.

        Raises:
            FamilyError: If `losses_a_col` is empty.
            ValueError: If `losses_a_col` contains NaN or `+inf`
                (propagated from :func:`wfcrc.utils.numerics.logsumexp`).
        """
        z = np.asarray(losses_a_col, dtype=np.float64)
        if z.size == 0:
            raise FamilyError("cannot estimate KL dual from an empty loss column")
        log_n = np.log(z.size)

        def h(eta: float) -> float:
            return float(eta * (float(logsumexp(z / eta)) - log_n + self.rho))

        eta_star = _bracket_and_minimize(h, self.eta_min)
        if eta_star == self.eta_min:
            # F-4 / §15 fallback: the boundary is optimal only when h is
            # non-decreasing everywhere on [eta_min, infinity) (see
            # _bracket_and_minimize's proof), which is exactly the
            # degenerate case this fallback targets. Use a fixed,
            # data-independent eta instead; validity is preserved by weak
            # duality for any eta > 0 (Math Spec §5).
            eta_star = self.fallback_eta
        mu_star = float(eta_star * (float(logsumexp(z / eta_star)) - log_n))
        return KLDualParams(eta=eta_star, mu=mu_star)

    def c(self, theta: Any) -> float:
        """Return `c(eta, mu) = mu + eta * rho`.

        Args:
            theta: A :class:`KLDualParams`, as returned by
                :meth:`estimate_dual`.

        Returns:
            `mu + eta * rho`.
        """
        params: KLDualParams = theta
        return float(params.mu + params.eta * self.rho)

    def t(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
        """Return `t(z; eta, mu) = eta * (exp((z - mu)/eta) - 1)`.

        Args:
            z: Scalar or array of loss values.
            theta: A :class:`KLDualParams`, as returned by
                :meth:`estimate_dual`.

        Returns:
            `eta * (exp((z - mu)/eta) - 1)`, same shape as `z`. May be
            non-finite for extreme inputs; :meth:`transform` (which wraps
            this) detects and rejects that case (F-3, Algorithm Spec §14).
        """
        params: KLDualParams = theta
        z_arr = np.asarray(z, dtype=np.float64)
        with np.errstate(over="ignore"):
            # Overflow -> inf is an expected, handled outcome here: transform()
            # (the sole caller in normal use) checks finiteness and raises
            # FamilyError, so this must not also emit a RuntimeWarning.
            return params.eta * (np.exp((z_arr - params.mu) / params.eta) - 1.0)
