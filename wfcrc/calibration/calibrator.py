"""``WFCRCCalibrator`` — orchestrates the frozen single-split WF-CRC procedure.

Dispatches on `family.family_type` to one of three frozen branches
(Algorithm Specification §6-§7'):

- **Dual branch** (`cvar`, `kl`): split A/B (:class:`~wfcrc.calibration.splitter.Splitter`),
  estimate the dual on A, transform on B, apply the `n_B` inflation
  (§7 steps 2-6), binary-search threshold (§7 step 7).
- **Finite-group branch**: no split; standard CRC per group on the full
  calibration set, deploy `max_G lambda_hat_G` (§7').
- **Known-weight branch**: no split; weighted CRC over the full
  calibration set (§7').

This is the single integration point of MS2: it depends on
:mod:`wfcrc.ambiguity`, :mod:`wfcrc.calibration.splitter`,
:mod:`wfcrc.calibration.threshold_search`, and
:mod:`wfcrc.calibration.loss_table`, and consumes (but does not construct)
a :class:`~wfcrc.config.schema.CalibrationConfig`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.splitter import Splitter
from wfcrc.calibration.threshold_search import ThresholdSearch
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import FamilyError

__all__ = ["CalibrationResult", "WFCRCCalibrator"]


@dataclass(frozen=True)
class CalibrationResult:
    """The outcome of one WF-CRC calibration run.

    Fields mirror the Implementation Blueprint's `CalibrationResult`
    exactly for the dual branch (`n_A`, `n_B`, `B̃`, `R̂_B` all apply);
    for the finite-group/known-weight branches those quantities do not
    exist in the same form (no A/B split), so they are `None` there and
    branch-specific detail lives in `diagnostics` instead (Algorithm Spec
    §7').

    Attributes:
        lambda_hat: The selected threshold `λ̂`.
        empty_flag: `True` if no grid point satisfied the target risk
            (F-1, Algorithm Spec §14) and `lambda_hat` therefore fell back
            to the default (conventionally `λ_max`).
        n_a: Size of block `A` (dual branch only).
        n_b: Size of block `B` (dual branch only).
        b_tilde: `B̃`, the transformed-loss bound (dual branch only).
        r_hat_b: `R̂_B(λ̂)`, the (uninflated) worst-case empirical risk at
            `λ̂` (dual branch only).
        diagnostics: Branch-specific extra detail (e.g. per-group
            thresholds for the finite-group branch, or `n`/weight sum for
            the known-weight branch).
    """

    lambda_hat: float
    empty_flag: bool
    n_a: int | None = None
    n_b: int | None = None
    b_tilde: float | None = None
    r_hat_b: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class WFCRCCalibrator:
    """Orchestrates the frozen single-split WF-CRC calibration procedure.

    Attributes:
        splitter: The A/B splitter (dual branch only).
        threshold_search: The monotone binary search used by every branch.
    """

    def __init__(
        self,
        *,
        splitter: Splitter | None = None,
        threshold_search: ThresholdSearch | None = None,
    ) -> None:
        """Initialize the calibrator with (optionally injected) collaborators.

        Args:
            splitter: A/B splitter; defaults to a fresh :class:`Splitter`.
            threshold_search: Binary-search collaborator; defaults to a
                fresh :class:`~wfcrc.calibration.threshold_search.ThresholdSearch`.
        """
        self.splitter = splitter if splitter is not None else Splitter()
        self.threshold_search = (
            threshold_search if threshold_search is not None else ThresholdSearch()
        )

    def calibrate(
        self,
        loss_table: LossTable,
        family: AmbiguityFamily,
        cfg: CalibrationConfig,
        *,
        seed: int,
    ) -> CalibrationResult:
        """Run WF-CRC calibration, dispatching on `family.family_type`.

        Args:
            loss_table: The precomputed `L[i, lambda]` table (L1a
                dimension-independence: this and `family` are the only
                data this method touches).
            family: The ambiguity family (`cvar`/`kl`/`finite_group`/
                `known_weight`).
            cfg: Calibration parameters (`alpha`, `B`, `pi`, `lambda_grid`).
            seed: Base seed for the A/B split (dual branch only; ignored
                by the alternative branches, which are fully
                deterministic — Algorithm Spec §17).

        Returns:
            The :class:`CalibrationResult`.

        Raises:
            ValueError: If `loss_table.lambda_grid` does not match
                `cfg.lambda_grid`, or `loss_table` has too few rows for
                the dual branch's split.
            FamilyError: If `family.family_type` is not one of the four
                frozen supported types (F-6, Algorithm Spec §14).
        """
        cfg_grid = np.asarray(cfg.lambda_grid, dtype=np.float64)
        if not np.array_equal(loss_table.lambda_grid, cfg_grid):
            raise ValueError(
                "loss_table.lambda_grid does not match cfg.lambda_grid; "
                "the loss table must have been built against this config's grid"
            )

        if family.family_type in ("cvar", "kl"):
            if not isinstance(family, DualAmbiguityFamily):
                raise FamilyError(
                    f"family_type={family.family_type!r} requires a DualAmbiguityFamily"
                )
            return self._calibrate_dual(loss_table, family, cfg, seed)
        if family.family_type == "finite_group":
            groups_fn = getattr(family, "groups", None)
            if groups_fn is None:
                raise FamilyError("finite_group family must implement groups()")
            return self._calibrate_finite_group(loss_table, groups_fn(), cfg)
        if family.family_type == "known_weight":
            weights_fn = getattr(family, "weights", None)
            if weights_fn is None:
                raise FamilyError("known_weight family must implement weights()")
            return self._calibrate_known_weight(loss_table, weights_fn(), cfg)
        raise FamilyError(f"unsupported family type: {family.family_type!r}")

    def _calibrate_dual(
        self,
        loss_table: LossTable,
        family: DualAmbiguityFamily,
        cfg: CalibrationConfig,
        seed: int,
    ) -> CalibrationResult:
        """Dual branch: split A/B, estimate on A, transform + inflate on B.

        Implements Algorithm Specification §7 steps 2-7 and §8's
        `wf_crc_single_split` exactly.
        """
        n = loss_table.shape[0]
        a_idx, b_idx = self.splitter.split(n, cfg.pi, seed)
        n_b = len(b_idx)
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])

        theta_by_lambda = {
            float(lam): family.estimate_dual(loss_table.column(float(lam))[a_idx])
            for lam in lambda_grid
        }
        b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in lambda_grid)

        def r_hat(lam: float) -> float:
            theta = theta_by_lambda[float(lam)]
            col_b = loss_table.column(lam)[b_idx]
            l_tilde = family.transform(col_b, theta)
            return float(np.mean(l_tilde))

        def g(lam: float) -> float:
            return (n_b / (n_b + 1)) * r_hat(lam) + b_tilde / (n_b + 1)

        lambda_hat = self.threshold_search.search(g, lambda_grid, cfg.alpha, default=lambda_max)
        empty_flag = g(lambda_max) > cfg.alpha

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            n_a=len(a_idx),
            n_b=n_b,
            b_tilde=b_tilde,
            r_hat_b=r_hat(lambda_hat),
        )

    def _calibrate_finite_group(
        self,
        loss_table: LossTable,
        groups: tuple[tuple[int, ...], ...],
        cfg: CalibrationConfig,
    ) -> CalibrationResult:
        """Finite-group branch: standard CRC per group, deploy the max (§7').

        `lambda_hat_G = inf{ lambda : (n_G/(n_G+1))*mean_{i in G} L[i,lambda]
        + B/(n_G+1) <= alpha }`; deploy `lambda_hat = max_G lambda_hat_G`.
        """
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])

        per_group: list[dict[str, Any]] = []
        for group_idx, indices_seq in enumerate(groups):
            indices = np.asarray(indices_seq, dtype=np.intp)
            n_g = indices.size

            def g_group(lam: float, indices: np.ndarray = indices, n_g: int = n_g) -> float:
                col = loss_table.column(lam)[indices]
                r_hat = float(np.mean(col))
                return (n_g / (n_g + 1)) * r_hat + cfg.B / (n_g + 1)

            lambda_hat_g = self.threshold_search.search(
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

    def _calibrate_known_weight(
        self,
        loss_table: LossTable,
        weights: np.ndarray,
        cfg: CalibrationConfig,
    ) -> CalibrationResult:
        """Known-weight branch: weighted CRC over the full sample (§7').

        `g(lambda) = (n/(n+1)) * [sum_i w_i L[i,lambda] / sum_i w_i] + B/(n+1)`.
        """
        n = loss_table.shape[0]
        if weights.shape[0] != n:
            raise FamilyError(
                f"known-weight family has {weights.shape[0]} weights but the "
                f"loss table has {n} rows"
            )
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])
        weight_sum = float(np.sum(weights))

        def r_hat(lam: float) -> float:
            col = loss_table.column(lam)
            return float(np.sum(weights * col) / weight_sum)

        def g(lam: float) -> float:
            return (n / (n + 1)) * r_hat(lam) + cfg.B / (n + 1)

        lambda_hat = self.threshold_search.search(g, lambda_grid, cfg.alpha, default=lambda_max)
        empty_flag = g(lambda_max) > cfg.alpha

        return CalibrationResult(
            lambda_hat=lambda_hat,
            empty_flag=empty_flag,
            r_hat_b=r_hat(lambda_hat),
            diagnostics={"n": n, "weight_sum": weight_sum},
        )
