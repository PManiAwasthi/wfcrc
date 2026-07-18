"""``SplitConformalLAC`` — classical split conformal prediction (Sadinle, Lei & Wasserman 2019).

Per `docs/EXPERIMENT_PROTOCOL.md`/`docs/MODEL_POLICY.md` ("Vanilla CRC /
split conformal (LAC): the marginal-risk/coverage reference"): the **Least
Ambiguous Set-Valued Classifier** paper's own calibration rule, a
finite-sample-exact marginal *coverage* guarantee `P(Y in C_lambda(X)) >=
1 - alpha`, distinct from `wfcrc.baselines.vanilla_crc.VanillaCRC`'s
*risk*-control guarantee (Angelopoulos et al. 2022) even though both are
listed together in the vault as "the marginal reference": LAC's threshold
is the `ceil((n+1)(1-alpha))`-th order statistic of calibration
nonconformity scores, not a CRC-style bias-corrected bound.

**Note on set construction vs. calibration.** `wfcrc.prediction_sets.
classification.ThresholdSets` is *already* named "the Least Ambiguous
set-valued Classifier (LAC)" in its own frozen module docstring (MS3) —
that class is the **set-construction** half of LAC (`C_lambda(x) = {k :
score_k >= 1-lambda}`), already used by WF-CRC itself. This module is the
other, missing half: LAC's own **calibration** procedure (how `lambda` is
chosen), which has no worst-case-over-family adjustment at all. The two
compose exactly as the original paper intends: `ThresholdSets` builds the
sets; `SplitConformalLAC` picks `lambda`.

**Exact formula, expressed over the already-built miscoverage loss table
(no duplicated quantile-of-raw-scores logic).** Since a miscoverage
`LossTable` already encodes `L[i, lambda] = 1{Y_i not in C_lambda(X_i)}`
for every grid point — monotone non-increasing in `lambda` by construction
(A2, the frozen `LossEvaluator.assert_monotone` precondition every
`LossTableBuilder` output already satisfies) — LAC's classical order-
statistic threshold is exactly:

    k = ceil((n+1) * (1-alpha))
    lambda_hat = min{ lambda in grid : (number of covered examples at lambda) >= k }
               = min{ lambda in grid : mean(L[:, lambda]) <= (n-k)/n }

which is the same `ThresholdSearch.search` binary search every frozen
branch of `WFCRCCalibrator` already uses, with `g(lambda) = mean(L[:,
lambda])` (the *plain*, untransformed miscoverage rate — no dual, no
bias-correction term) and target `(n-k)/n` in place of `alpha`. This is a
direct algebraic restatement of Sadinle-Lei-Wasserman's own order-
statistic rule, not an approximation of it.
"""

from __future__ import annotations

import math

import numpy as np

from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.threshold_search import ThresholdSearch
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import BaselineError

__all__ = ["SplitConformalLAC"]


class SplitConformalLAC(Calibrator):
    """Split conformal prediction (Sadinle, Lei & Wasserman 2019), the classical LAC baseline."""

    def __init__(self, *, threshold_search: ThresholdSearch | None = None) -> None:
        """Initialize the baseline.

        Args:
            threshold_search: An injected `ThresholdSearch`; defaults to a
                fresh instance (matching the frozen calibrator's own
                default-collaborator pattern).
        """
        self._threshold_search = (
            threshold_search if threshold_search is not None else ThresholdSearch()
        )

    @property
    def baseline_name(self) -> str:
        """Return ``"lac"``."""
        return "lac"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Run the classical split-conformal order-statistic calibration.

        Args:
            loss_table: The precomputed calibration miscoverage-loss table
                `L[i, lambda] = 1{Y_i not in C_lambda(X_i)}`. Passing a
                loss table built from a different loss is a caller error
                (LAC's guarantee is specifically a *coverage* statement)
                and is not validated here — this baseline, like the frozen
                `WFCRCCalibrator`, trusts its `LossTable` input per L1a
                dimension-independence.
            cfg: Calibration parameters (`pi`/`B` are unused — LAC has no
                A/B split and no loss-bound-dependent bias term).
            seed: Unused (LAC is fully deterministic given `loss_table`);
                accepted only to satisfy the common `Calibrator` interface.

        Returns:
            A `CalibrationResult` with `diagnostics = {"k": k, "n": n,
            "target_miscoverage_rate": (n-k)/n}`; `r_hat_b` holds the
            realized empirical miscoverage rate at `lambda_hat`.

        Raises:
            BaselineError: If `cfg.alpha` is outside `(0, 1)`, or `n` is
                too small for `k = ceil((n+1)(1-alpha))` to be a valid
                (`1 <= k <= n`) order statistic.
        """
        if not (0.0 < cfg.alpha < 1.0):
            raise BaselineError(
                f"LAC requires 0 < alpha < 1 (it is a coverage-level parameter here, "
                f"not a general risk bound), got alpha={cfg.alpha}"
            )
        n = loss_table.shape[0]
        k = math.ceil((n + 1) * (1.0 - cfg.alpha))
        if not (1 <= k <= n):
            raise BaselineError(
                f"n={n} is too small for LAC's order statistic k=ceil((n+1)(1-alpha))={k}; "
                f"need 1 <= k <= n"
            )
        target = (n - k) / n
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])

        def g(lam: float) -> float:
            return float(np.mean(loss_table.column(lam)))

        lambda_hat = self._threshold_search.search(g, lambda_grid, target, default=lambda_max)
        empty_flag = g(lambda_max) > target

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            r_hat_b=g(lambda_hat),
            diagnostics={"k": k, "n": n, "target_miscoverage_rate": target},
        )


BASELINES["lac"] = SplitConformalLAC
