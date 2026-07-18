"""``RobustFDivergenceCP`` — the Cauchois-Duchi f-divergence-ball robust conformal baseline.

Per `docs/EXPERIMENT_PROTOCOL.md`/`docs/MODEL_POLICY.md` §1.2
("robust-CP (Cauchois-Duchi f-div; Levy-Prokhorov)"): the frozen Research
Vault names this baseline by author only, "Cauchois-Duchi 2024" (the
`Theorem Summit - Paper 1 Central Theorem.md` document's own words:
"Cauchois-Duchi 2024 worst-case-over-ball" is *folklore-equivalent* to
WF-CRC's own robust instantiation) — this repository's frozen `kl` family
already computes exactly the worst-case-over-KL-ball dual
`sup_{Q: KL(Q||P)<=rho} E_Q[L]` this baseline needs (Algorithm
Specification §7, `wfcrc.ambiguity.kl.KLFamily`), so no new dual-estimation
mathematics is introduced here.

**The one genuine construction difference from WF-CRC's own KL branch,
disclosed explicitly.** Cauchois & Duchi's own robust-validation
construction (unlike WF-CRC's frozen single-split procedure) does not
reserve a separate dual-estimation block: it estimates the worst-case
adjustment directly from the *same* pooled calibration sample its
robustness-adjusted quantile is then computed over, and applies the
standard (non-`n_B`) `n`-based conformal inflation. This module implements
exactly that pooled, single-block construction — reusing the frozen
`DualAmbiguityFamily.estimate_dual`/`.transform`/`.btil` public API
unmodified, just without `wfcrc.calibration.splitter.Splitter`'s A/B
partition. This is the same kind of "same-sample dual re-estimation"
already disclosed for `wfcrc.evaluation.metrics.realized_worst_case_risk`
(an ordinary plug-in-estimator optimism, not the P2/same-data-
threshold-selection failure mode Math Spec §12 warns against, since this
function *is* the threshold-selection step here, exactly as the original
Cauchois-Duchi construction itself is — the vault's own "folklore-
equivalent" framing is precisely this point: it is a legitimate published
method, just not WF-CRC's specific cross-fit-free single-split
architecture).

**Explicitly out of scope, not silently substituted (see
`docs/MODEL_POLICY.md`/final MS9 report):** the Levy-Prokhorov variant
named alongside Cauchois-Duchi in the same Blueprint line is a *different*
divergence (optimal-transport / Wasserstein-family), which the frozen
`Paper 1 - FRAMEWORK SPECIFICATION.md` itself already lists as "future
work... open gap" — not currently representable by any family this
repository implements. Implementing it would require extending the frozen
ambiguity-family architecture itself (a framework change), which is
outside this milestone's purely-additive, non-framework-modifying scope.
This module implements the f-divergence (KL-ball) variant only.
"""

from __future__ import annotations

import numpy as np

from wfcrc.ambiguity.base import DualAmbiguityFamily
from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.threshold_search import ThresholdSearch
from wfcrc.config.schema import CalibrationConfig

__all__ = ["RobustFDivergenceCP"]


class RobustFDivergenceCP(Calibrator):
    """Cauchois-Duchi (2024) f-divergence-ball robust conformal prediction.

    Attributes:
        family: The dual ambiguity family defining the divergence ball
            (`wfcrc.ambiguity.kl.KLFamily`, matching the paper's own
            f-divergence framing; `wfcrc.ambiguity.cvar.CVaRFamily` is also
            accepted, since both are `DualAmbiguityFamily` instances and
            nothing below is KL-specific, but KL is the intended match).
    """

    def __init__(
        self, family: DualAmbiguityFamily, *, threshold_search: ThresholdSearch | None = None
    ) -> None:
        """Initialize the baseline.

        Args:
            family: The divergence-ball family (dual-estimating).
            threshold_search: An injected `ThresholdSearch`; defaults to a
                fresh instance.
        """
        self.family = family
        self._threshold_search = (
            threshold_search if threshold_search is not None else ThresholdSearch()
        )

    @property
    def baseline_name(self) -> str:
        """Return ``"robust_fdiv"``."""
        return "robust_fdiv"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Calibrate a pooled (no-split), f-divergence-ball-adjusted threshold.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters (`pi` is unused — no A/B split).
            seed: Unused (this construction is fully deterministic given
                `loss_table`/`cfg`); accepted only to satisfy the common
                `Calibrator` interface.

        Returns:
            A `CalibrationResult` with `n_a=None`, `n_b=None` (no split
            exists in this construction — disclosed above), `b_tilde` the
            max transformed-loss bound over the grid, and `r_hat_b` the
            realized transformed empirical risk at `lambda_hat`.
        """
        n = loss_table.shape[0]
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])

        theta_by_lambda = {
            float(lam): self.family.estimate_dual(loss_table.column(float(lam)))
            for lam in lambda_grid
        }
        b_tilde = max(self.family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in lambda_grid)

        def r_hat(lam: float) -> float:
            theta = theta_by_lambda[float(lam)]
            col = loss_table.column(lam)
            return float(np.mean(self.family.transform(col, theta)))

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


BASELINES["robust_fdiv"] = RobustFDivergenceCP
