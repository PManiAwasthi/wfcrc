"""Architecture-ablation baselines for E7 — pooled K-fold, total-`n` inflation, fixed-eta.

Per `docs/EXPERIMENT_PROTOCOL.md`'s E7 row ("architecture ablation
(single-split / K-fold / n_B / fixed-eta) — the load-bearing defense of
frozen Proof Obligations P3/P4") and the Experiment Blueprint's own
component-removal table (§23):

| Removed component | Expected effect |
|---|---|
| Cross-fitting (-> pooled K-fold) | under-covers (risk > alpha) -- confirms P3 |
| `n_B` inflation (-> total-n) | under-covers -- confirms P4 |
| Dual optimization (-> fixed-eta) | valid but conservative (larger sets) |

**Promotion, not reimplementation.** `PooledKFoldWFCRC` and
`TotalNInflationWFCRC` below are the exact same constructions already
implemented, validated, and frozen as test-only harnesses in
`tests/unit/calibration/test_negative_controls.py` (backfilling an MS3
exit-gate gap during MS4) — this module wraps that *same* logic (formula-
for-formula identical; the only change is exposing it via the common
`Calibrator` interface instead of module-private test functions) so E7 can
call it exactly like every other registered baseline. Per this project's
"do not modify frozen methodology" rule (MS8/MS9 instructions), the
original test file is left in place and unmodified — it continues to
exercise the empirical under-coverage claim standalone; this module is a
second, additive consumer of the identical procedure, not a replacement.

`FixedEtaWFCRC` is new (no prior implementation, test-only or otherwise,
existed anywhere in this repository — confirmed by a repository-wide
search during MS9's baseline audit). It implements the third row of the
table above: skip data-driven dual estimation entirely, substitute a
single, caller-supplied, data-independent `theta`, and — since there is
then nothing left to reserve a dual-estimation block *for* — use the
**entire** calibration set as the empirical-risk block with standard
`n`-based (not `n_B`-based) inflation. This is valid by weak duality for
*any* fixed dual parameter (`PROJECT_CONTEXT.md` §4, "weak duality holds
for any fixed dual parameter, not just the optimum — this is why a
fixed-eta fallback is valid, not a hack"; the same argument the frozen
`KLFamily`'s own F-4 fallback already relies on, `wfcrc/ambiguity/kl.py`)
— expected to be *valid but conservative*, not an under-covering negative
control like the other two.

**Monotonicity / search-algorithm note.** The original test harness
deliberately used a linear scan rather than `ThresholdSearch`'s binary
search for the two ablations, because a per-fold or a differently-
denominatored dual estimate is not guaranteed to keep `g` non-increasing
in `lambda` the way the frozen single-split procedure's own `g` is trusted
to be (`ThresholdSearch`'s own docstring: "assumes -- does not verify --
that g is non-increasing"). That reasoning is preserved unchanged here for
`PooledKFoldWFCRC`/`TotalNInflationWFCRC` (still a plain linear scan, not
`ThresholdSearch`) — switching search algorithms during promotion would
itself be a silent behavior change on constructions this project has
already validated. `FixedEtaWFCRC`, by contrast, uses a genuinely *fixed*
`theta` for every `lambda` (not per-lambda-estimated), so `transform(L[i,
lambda]; theta)` is non-increasing in `lambda` whenever the raw loss column
itself is (the same monotone-transform argument the frozen dual branch
already relies on for its own per-lambda-varying theta) — `ThresholdSearch`
is used there, consistent with every other single-fixed-theta construction
in this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wfcrc.ambiguity.base import DualAmbiguityFamily
from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.splitter import Splitter
from wfcrc.calibration.threshold_search import ThresholdSearch
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import BaselineError

__all__ = ["FixedEtaWFCRC", "PooledKFoldWFCRC", "TotalNInflationWFCRC"]


def _linear_scan_lambda_hat(
    g_values: NDArray[np.float64], lambda_grid: NDArray[np.float64], alpha: float
) -> float:
    """Smallest grid point with `g(lambda) <= alpha`, else `lambda_max` (no monotonicity assumed).

    Verbatim port of `tests/unit/calibration/test_negative_controls.py`'s
    own `_search_lambda_hat` helper (same algorithm, generalized to take
    `lambda_grid` as a parameter instead of a module constant).
    """
    feasible = lambda_grid[g_values <= alpha]
    if feasible.size == 0:
        return float(lambda_grid[-1])
    return float(np.min(feasible))


class TotalNInflationWFCRC(Calibrator):
    """The frozen single-split procedure, verbatim, with `n` substituted for `n_B` (P4 ablation).

    Attributes:
        family: The dual ambiguity family (`cvar`/`kl`).
    """

    def __init__(self, family: DualAmbiguityFamily, *, splitter: Splitter | None = None) -> None:
        """Initialize the baseline.

        Args:
            family: The dual ambiguity family to estimate on block A.
            splitter: An injected `Splitter`; defaults to a fresh instance.
        """
        self.family = family
        self._splitter = splitter if splitter is not None else Splitter()

    @property
    def baseline_name(self) -> str:
        """Return ``"total_n_inflation"``."""
        return "total_n_inflation"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Estimate the dual on block A exactly as WF-CRC does, but inflate by `n`, not `n_B`.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters.
            seed: Base seed for the A/B split (same `Splitter` WF-CRC uses).

        Returns:
            A `CalibrationResult`; `n_b` is `None` since `n` (not `n_B`)
            is the operative denominator here — recorded instead in
            `diagnostics["n"]`, alongside `n_a` for comparison.
        """
        n = loss_table.shape[0]
        a_idx, b_idx = self._splitter.split(n, cfg.pi, seed)
        lambda_grid = loss_table.lambda_grid

        theta_by_lambda = {
            float(lam): self.family.estimate_dual(loss_table.column(float(lam))[a_idx])
            for lam in lambda_grid
        }
        b_tilde = max(self.family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in lambda_grid)

        def r_hat(lam: float) -> float:
            theta = theta_by_lambda[float(lam)]
            col_b = loss_table.column(lam)[b_idx]
            return float(np.mean(self.family.transform(col_b, theta)))

        g_values = np.array(
            [(n / (n + 1)) * r_hat(float(lam)) + b_tilde / (n + 1) for lam in lambda_grid]
        )
        lambda_hat = _linear_scan_lambda_hat(g_values, lambda_grid, cfg.alpha)
        empty_flag = float(g_values[-1]) > cfg.alpha

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            n_a=len(a_idx),
            b_tilde=b_tilde,
            r_hat_b=r_hat(lambda_hat),
            diagnostics={"n": n, "n_b_unused": len(b_idx)},
        )


class PooledKFoldWFCRC(Calibrator):
    """Pooled K-fold cross-fit: per-fold dual on other folds, pooled `n`-inflation (P3 ablation).

    Attributes:
        family: The dual ambiguity family (`cvar`/`kl`).
        k_folds: Number of cross-fit folds, `k_folds >= 2`.
    """

    def __init__(self, family: DualAmbiguityFamily, *, k_folds: int = 5) -> None:
        """Initialize the baseline.

        Args:
            family: The dual ambiguity family to estimate per fold.
            k_folds: Number of cross-fit folds; must be `>= 2`.

        Raises:
            wfcrc.exceptions.BaselineError: If `k_folds < 2`.
        """
        if k_folds < 2:
            raise BaselineError(f"k_folds must be >= 2, got {k_folds}")
        self.family = family
        self.k_folds = k_folds

    @property
    def baseline_name(self) -> str:
        """Return ``"pooled_k_fold"``."""
        return "pooled_k_fold"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Cross-fit a dual per fold, pool every out-of-fold transformed loss, then threshold.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters.
            seed: Seed for the fold permutation (`numpy.random.default_rng(seed)`,
                matching the promoted test harness's own RNG usage exactly).

        Returns:
            A `CalibrationResult`; `diagnostics["thetas_seen"]` records how
            many distinct `(lambda, fold)` dual estimates contributed to
            `b_tilde`'s global max, for diagnostic parity with the original
            test harness's own two-pass structure.
        """
        n = loss_table.shape[0]
        lambda_grid = loss_table.lambda_grid
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n), self.k_folds)

        r_hats = np.empty(lambda_grid.size, dtype=np.float64)
        thetas_by_lambda: list[list[Any]] = []
        for j, lam in enumerate(lambda_grid):
            col = loss_table.column(float(lam))
            pooled_transformed = np.empty(n, dtype=np.float64)
            thetas: list[Any] = []
            for fold in folds:
                fold_mask = np.zeros(n, dtype=bool)
                fold_mask[fold] = True
                theta = self.family.estimate_dual(col[~fold_mask])
                pooled_transformed[fold] = self.family.transform(col[fold], theta)
                thetas.append(theta)
            thetas_by_lambda.append(thetas)
            r_hats[j] = np.mean(pooled_transformed)

        b_tilde = max(
            self.family.btil(theta, cfg.B) for thetas in thetas_by_lambda for theta in thetas
        )
        g_values = np.array(
            [(n / (n + 1)) * r_hats[j] + b_tilde / (n + 1) for j in range(lambda_grid.size)]
        )
        lambda_hat = _linear_scan_lambda_hat(g_values, lambda_grid, cfg.alpha)
        empty_flag = float(g_values[-1]) > cfg.alpha

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            b_tilde=b_tilde,
            diagnostics={
                "n": n,
                "k_folds": self.k_folds,
                "thetas_seen": sum(len(t) for t in thetas_by_lambda),
            },
        )


class FixedEtaWFCRC(Calibrator):
    """Fixed, data-independent dual parameter; no A/B split; standard `n`-inflation.

    Valid by weak duality for any fixed `theta` (Math Spec §5;
    `PROJECT_CONTEXT.md` §4) — expected to be *valid but conservative*
    (larger prediction sets than the data-adaptive single-split
    procedure), the third row of the Experiment Blueprint's own
    component-removal table (§23).

    Attributes:
        family: The dual ambiguity family (only used for its `.c`/`.t`/
            `.transform`/`.btil` methods — `estimate_dual` is never
            called by this baseline, since `theta` is fixed).
        theta: The fixed dual parameter (e.g. a `float` for CVaR, a
            `wfcrc.ambiguity.kl.KLDualParams` for KL) applied uniformly
            across the whole `lambda`-grid.
    """

    def __init__(
        self,
        family: DualAmbiguityFamily,
        theta: Any,
        *,
        threshold_search: ThresholdSearch | None = None,
    ) -> None:
        """Initialize the baseline.

        Args:
            family: The dual ambiguity family whose transform/bound
                functions are applied at the fixed `theta`.
            theta: The fixed, data-independent dual parameter.
            threshold_search: An injected `ThresholdSearch`; defaults to a
                fresh instance.
        """
        self.family = family
        self.theta = theta
        self._threshold_search = (
            threshold_search if threshold_search is not None else ThresholdSearch()
        )

    @property
    def baseline_name(self) -> str:
        """Return ``"fixed_eta"``."""
        return "fixed_eta"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Apply the fixed dual parameter to the whole calibration set, then threshold-search.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters (`pi` is unused — no A/B split).
            seed: Unused (this construction is fully deterministic given
                `loss_table`/`cfg`/`theta`); accepted only to satisfy the
                common `Calibrator` interface.

        Returns:
            A `CalibrationResult` with `n_a=None`, `n_b=None` (no split
            exists), `b_tilde = family.btil(theta, cfg.B)`, and `r_hat_b`
            the realized fixed-theta-transformed empirical risk at
            `lambda_hat`.
        """
        n = loss_table.shape[0]
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])
        b_tilde = self.family.btil(self.theta, cfg.B)

        def r_hat(lam: float) -> float:
            col = loss_table.column(lam)
            return float(np.mean(self.family.transform(col, self.theta)))

        def g(lam: float) -> float:
            return (n / (n + 1)) * r_hat(lam) + b_tilde / (n + 1)

        lambda_hat = self._threshold_search.search(g, lambda_grid, cfg.alpha, default=lambda_max)
        empty_flag = g(lambda_max) > cfg.alpha

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            b_tilde=b_tilde,
            r_hat_b=r_hat(lambda_hat),
            diagnostics={"n": n},
        )


BASELINES["total_n_inflation"] = TotalNInflationWFCRC
BASELINES["pooled_k_fold"] = PooledKFoldWFCRC
BASELINES["fixed_eta"] = FixedEtaWFCRC
