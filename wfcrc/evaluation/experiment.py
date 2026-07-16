"""``run_experiment`` — reduced, dataset-free experiment-execution entry point.

Composes the already-frozen
:func:`wfcrc.calibration.pipeline.run_calibration_pipeline` with
:mod:`wfcrc.evaluation.metrics` into a structured, JSON-serializable
:class:`ExperimentReport`, given an already-built calibration `LossTable`
and test `LossTable`.

This is *not* the Implementation Blueprint's full `runner.ExperimentRunner`
(MS5, M15): there is no dataset/model loading (`wfcrc.datasets`' concrete
loaders don't exist yet — see that package's docstring), no plotting
(`viz.Plotter`, M14), and no sweeps/checkpointing/resume. Those remain a
later milestone's scope, once real datasets/models and a plotting
requirement are actually in play; this module covers only the part of
"experiment execution" that is fully specified and buildable today —
calibrate, verify, measure, report — directly on pre-built loss tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.pipeline import VerifierLike, run_calibration_pipeline
from wfcrc.config.schema import CalibrationConfig
from wfcrc.evaluation.metrics import (
    duality_gap,
    effective_sizes,
    per_group_risk,
    realized_marginal_risk,
    realized_worst_case_risk,
)
from wfcrc.evaluation.verifier import VerificationReport
from wfcrc.utils.io import content_hash

__all__ = ["ExperimentReport", "run_experiment"]


@dataclass(frozen=True)
class ExperimentReport:
    """The outcome of one :func:`run_experiment` call.

    Attributes:
        calibration: The `CalibrationResult` from the calibration
            `LossTable`.
        verification: The merged `VerificationReport`, or `None` if no
            `verifier` was supplied to :func:`run_experiment`.
        metrics: Realized-risk / efficiency metrics computed on the test
            `LossTable` at `calibration.lambda_hat` (see
            :func:`run_experiment` for exactly which keys are populated).
        config_hash: A stable content hash of the calibration config used,
            for reproducibility provenance (Implementation Blueprint §17).
    """

    calibration: CalibrationResult
    verification: VerificationReport | None
    metrics: dict[str, Any]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Render this report as a plain, JSON-serializable dict.

        Returns:
            A nested `dict` with every field of `calibration` inlined
            (rather than nesting the dataclass itself), suitable for
            `json.dumps` or `wfcrc.utils.io.save_json`.
        """
        return {
            "lambda_hat": self.calibration.lambda_hat,
            "empty_flag": self.calibration.empty_flag,
            "n_a": self.calibration.n_a,
            "n_b": self.calibration.n_b,
            "b_tilde": self.calibration.b_tilde,
            "r_hat_b": self.calibration.r_hat_b,
            "diagnostics": dict(self.calibration.diagnostics),
            "verification_passed": (
                None if self.verification is None else self.verification.passed
            ),
            "metrics": self.metrics,
            "config_hash": self.config_hash,
        }


def run_experiment(
    cal_loss_table: LossTable,
    test_loss_table: LossTable,
    family: AmbiguityFamily,
    cfg: CalibrationConfig,
    *,
    seed: int,
    verifier: VerifierLike | None = None,
    groups: Sequence[Sequence[int]] | None = None,
) -> ExperimentReport:
    """Calibrate on `cal_loss_table`, then measure + report on `test_loss_table`.

    Args:
        cal_loss_table: The calibration `LossTable` (input to
            `run_calibration_pipeline`).
        test_loss_table: A held-out test `LossTable` (same `lambda_grid`)
            to measure realized metrics on.
        family: The ambiguity family to calibrate and measure under.
        cfg: Calibration parameters (`alpha`, `B`, `pi`, `lambda_grid`).
        seed: Base seed for the A/B split (dual branch only).
        verifier: An optional object satisfying `VerifierLike` (e.g.
            `wfcrc.evaluation.verifier.Verifier()`); forwarded to
            `run_calibration_pipeline`.
        groups: Optional per-group row-index sequences into
            `test_loss_table`, for `per_group_risk`.

    Returns:
        An `ExperimentReport` with `metrics` containing at least
        `"realized_marginal_risk"` and `"effective_sizes"`; also
        `"realized_worst_case_risk"` and `"duality_gap"` for dual families,
        and `"per_group_risk"` if `groups` is given.

    Raises:
        ValueError: Propagated from `run_calibration_pipeline` (e.g.
            mismatched `lambda_grid`).
        wfcrc.exceptions.FamilyError: Propagated from
            `run_calibration_pipeline`.
    """
    pipeline_result = run_calibration_pipeline(
        cal_loss_table, family, cfg, seed=seed, verifier=verifier
    )
    result = pipeline_result.calibration

    computed_metrics: dict[str, Any] = {
        "realized_marginal_risk": realized_marginal_risk(result, test_loss_table),
        "effective_sizes": effective_sizes(result),
    }
    if isinstance(family, DualAmbiguityFamily):
        worst_case = realized_worst_case_risk(result, test_loss_table, family)
        computed_metrics["realized_worst_case_risk"] = worst_case
        # WFCRCCalibrator's dual branch always populates r_hat_b (it is
        # None only for the finite-group/known-weight branches, which
        # cannot reach here since `family` is a DualAmbiguityFamily).
        assert result.r_hat_b is not None
        computed_metrics["duality_gap"] = duality_gap(result.r_hat_b, worst_case)
    if groups is not None:
        computed_metrics["per_group_risk"] = per_group_risk(result, test_loss_table, groups)

    config_hash = content_hash(
        {
            "alpha": cfg.alpha,
            "B": cfg.B,
            "pi": cfg.pi,
            "lambda_grid": list(cfg.lambda_grid),
        }
    )

    return ExperimentReport(
        calibration=result,
        verification=pipeline_result.verification,
        metrics=computed_metrics,
        config_hash=config_hash,
    )
