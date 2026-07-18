"""``GroupConditionalCRC`` — the finite-group Gibbs-Cherian-Candes conditional-coverage proxy.

Per `docs/EXPERIMENT_PROTOCOL.md`/`docs/MODEL_POLICY.md` §1.2
("conditional-coverage (Gibbs-Cherian-Candes)"): the Experiment Blueprint
names this baseline only by author-year (Gibbs, Cherian & Candès,
"Conformal Prediction With Conditional Guarantees", 2023) with no formula
transcribed anywhere in the frozen Research Vault. That paper's *general*
method (conditional validity over an arbitrary covariate-indexed function
class) is a substantially more complex construction (a minimax
decision-theoretic calibration, generally requiring its own dedicated
solver) than anything else in this module — implementing it in full is
explicitly **out of MS9's scope** (see this repository's `PROJECT_CONTEXT.md`
"stop and document, do not invent" rule; a full-generality implementation
is a candidate for a future, dedicated milestone, not a silent stand-in
here).

**What this module implements instead, disclosed explicitly.** E2's own
setting is not the general-covariate case: its groups are *discrete and
known in advance* (Cityscapes class/region groups, MSD organ groups — the
same `groups: tuple[tuple[int, ...], ...]` shape `wfcrc.ambiguity.
finite_group.FiniteGroupFamily` already consumes). For exactly that
finite-group case, the Gibbs et al. conditional-coverage guarantee
specializes to a well-established, unambiguous classical method: **Mondrian
(group-conditional / classwise) conformal prediction** (Vovk et al. 2005 ch.
4; Vovk 2012; the same per-group calibration idea Sadinle-Lei-Wasserman
2019 §2.3 use for classwise LAC) — calibrate a *separate* conformal
threshold independently within each group's own calibration subsample, so
each group individually achieves its own target level, rather than one
global threshold covering every group simultaneously.

**This is the qualitatively different deployment rule from WF-CRC's own
finite-group branch**, not a copy of it. `wfcrc.calibration.calibrator.
WFCRCCalibrator._calibrate_finite_group` computes the *same* per-group CRC
criterion this module does, but deploys `lambda_hat = max_G
lambda_hat_G` — **one** global threshold, conservative enough to satisfy
every group at once (the worst-case-over-family unification WF-CRC's own
theory is built on). `GroupConditionalCRC` deploys **each group's own**
`lambda_hat_G` separately — an "x-adaptive tight threshold, exact [risk
control] per group" as the frozen vault's own Theorem Summit document
characterizes the qualitative Gibbs-vs-robust distinction ("Gibbs yields an
x-adaptive tight threshold ...; robust yields one inflated global
threshold ⇒ different sets, so 'single construction' fails at the
algorithm level") — exactly the comparator WF-CRC's own conditional claim
(H2) needs to be measured against.

No frozen file is touched or imported privately: this module recomputes
the same public per-group CRC formula
(`wfcrc.calibration.threshold_search.ThresholdSearch`,
`wfcrc.calibration.calibrator`'s own documented `g_G(lambda) =
(n_G/(n_G+1))*R_hat_G(lambda) + B/(n_G+1)`) directly against
`wfcrc.ambiguity.finite_group.FiniteGroupFamily`'s public `groups()`
accessor, rather than reusing `WFCRCCalibrator`'s private
`_calibrate_finite_group` method (which hard-codes the `max` deployment
this baseline deliberately does not want).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.threshold_search import ThresholdSearch
from wfcrc.config.schema import CalibrationConfig

__all__ = ["GroupConditionalCRC"]


class GroupConditionalCRC(Calibrator):
    """Mondrian (group-conditional) conformal risk control — the finite-group Gibbs proxy.

    Attributes:
        family: The finite-group family naming this baseline's groups
            (row-index tuples into the calibration `LossTable`, identical
            shape to WF-CRC's own `finite_group` family).
    """

    def __init__(
        self, family: FiniteGroupFamily, *, threshold_search: ThresholdSearch | None = None
    ) -> None:
        """Initialize the baseline.

        Args:
            family: The groups to calibrate independently.
            threshold_search: An injected `ThresholdSearch`; defaults to a
                fresh instance.
        """
        self.family = family
        self._threshold_search = (
            threshold_search if threshold_search is not None else ThresholdSearch()
        )

    @property
    def baseline_name(self) -> str:
        """Return ``"group_conditional"``."""
        return "group_conditional"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Calibrate one CRC threshold independently per group.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters (`pi` is unused — no A/B split).
            seed: Unused (this baseline is fully deterministic given
                `loss_table`/`cfg`); accepted only to satisfy the common
                `Calibrator` interface.

        Returns:
            A `CalibrationResult` whose `lambda_hat` is
            `max_G lambda_hat_G` — a single scalar summary so this
            baseline satisfies the common `CalibrationResult` shape (e.g.
            for a single-figure `g`-curve or a single reported set-size
            number) — but whose `diagnostics["per_group"]` carries every
            group's own, individually-calibrated `lambda_hat_G`, which is
            this baseline's actual, deployed, per-group decision rule.
            Any evaluation that means to score this baseline the way Gibbs
            et al.'s guarantee is actually stated (per-group risk against
            that group's own threshold) must read `diagnostics["per_group"]`,
            not treat `lambda_hat` as a single global threshold the way it
            would for `wfcrc.baselines.wfcrc_adapter.WFCRCAdapter` — this
            is the disclosed adaptation this module's own docstring
            describes.
        """
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])
        groups = self.family.groups()

        per_group: list[dict[str, Any]] = []
        for group_idx, indices_seq in enumerate(groups):
            indices = np.asarray(indices_seq, dtype=np.intp)
            n_g = indices.size

            def g_group(lam: float, indices: np.ndarray = indices, n_g: int = n_g) -> float:
                col = loss_table.column(lam)[indices]
                r_hat = float(np.mean(col))
                return (n_g / (n_g + 1)) * r_hat + cfg.B / (n_g + 1)

            lambda_hat_g = self._threshold_search.search(
                g_group, lambda_grid, cfg.alpha, default=lambda_max
            )
            empty_g = g_group(lambda_max) > cfg.alpha
            per_group.append(
                {
                    "group": group_idx,
                    "n_g": int(n_g),
                    "lambda_hat": lambda_hat_g,
                    "empty": empty_g,
                }
            )

        lambda_hat = max(item["lambda_hat"] for item in per_group)
        empty_flag = any(item["empty"] for item in per_group)

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            diagnostics={"per_group": per_group},
        )


BASELINES["group_conditional"] = GroupConditionalCRC
