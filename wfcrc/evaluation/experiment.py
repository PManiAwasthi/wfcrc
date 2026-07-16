"""``run_experiment`` — reduced, dataset-free experiment-execution entry point.

Composes the already-frozen
:func:`wfcrc.calibration.pipeline.run_calibration_pipeline` with
:mod:`wfcrc.evaluation.metrics` into a structured, JSON-serializable
:class:`ExperimentReport`, given an already-built calibration `LossTable`
and test `LossTable`.

This function does not itself do dataset/model loading, plotting, or
sweeps/checkpointing/resume — it covers only "calibrate, verify, measure,
report" directly on pre-built loss tables. Those remaining responsibilities
are `wfcrc.runner.ExperimentRunner`'s (MS5, M15, complete): it composes
this function wholesale for calibrate+verify+metrics, then adds the
single-run `g`-curve figure (via `wfcrc.visualization`), checkpointing,
sweeps, and resume around it. Dataset/model *loading* remains out of scope
everywhere in this repository — `wfcrc.datasets`' concrete loaders don't
exist for any real, named dataset (see that package's docstring) — so both
this function and `ExperimentRunner` take already-built `LossTable`s
directly rather than resolving one from a dataset/model configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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


def _family_params(family: AmbiguityFamily) -> dict[str, Any]:
    """Extract a plain, JSON-safe snapshot of `family`'s instance state.

    Generic and non-invasive (no per-family special-casing, no change to
    any frozen `wfcrc.ambiguity` class): every concrete family (`CVaRFamily`,
    `KLFamily`, `FiniteGroupFamily`, `KnownWeightFamily`) is a plain object
    whose constructor parameters are its instance attributes, so
    `vars(family)` already captures them uniformly (e.g. `beta` for CVaR;
    `rho`/`eta_min`/`fallback_eta` for KL; `masks` for finite-group; the
    private `_weights` array for known-weight). `numpy` arrays are
    converted to lists so the result is directly JSON-serializable.

    Args:
        family: The ambiguity family to snapshot.

    Returns:
        A shallow `dict` copy of `vars(family)`, with any `numpy.ndarray`
        values rendered as lists.
    """
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in vars(family).items()
    }


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
        config_hash: A stable content hash of the calibration config,
            `seed`, and ambiguity-family type + parameters, for
            reproducibility provenance (Implementation Blueprint §17,
            which requires "seeds" in every run manifest alongside the
            config hash). Two runs that differ in seed, family type, or
            family parameters are guaranteed a different `config_hash`.
        seed: The base seed `run_experiment` was called with (the only
            stochastic quantity in the whole procedure, Algorithm Spec
            §17) — recorded explicitly so it is recoverable from the
            report alone, without which the report could not actually
            serve the reproducibility role its `config_hash` implies.
        family_type: `family.family_type` (`"cvar"`/`"kl"`/
            `"finite_group"`/`"known_weight"`) at the time of calibration.
        family_params: A plain snapshot of `family`'s parameters (see
            :func:`_family_params`), e.g. `{"beta": 0.2}` for a CVaR family.
    """

    calibration: CalibrationResult
    verification: VerificationReport | None
    metrics: dict[str, Any]
    config_hash: str
    seed: int
    family_type: str
    family_params: dict[str, Any]

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
            "seed": self.seed,
            "family_type": self.family_type,
            "family_params": self.family_params,
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
        and `"per_group_risk"` if `groups` is given. `config_hash` covers
        `cfg`, `seed`, and `family`'s type + parameters, so two calls that
        differ in any of those (not just `cfg`) get different hashes.

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

    family_params = _family_params(family)
    config_hash = content_hash(
        {
            "alpha": cfg.alpha,
            "B": cfg.B,
            "pi": cfg.pi,
            "lambda_grid": list(cfg.lambda_grid),
            "seed": seed,
            "family_type": family.family_type,
            "family_params": family_params,
        }
    )

    return ExperimentReport(
        calibration=result,
        verification=pipeline_result.verification,
        metrics=computed_metrics,
        config_hash=config_hash,
        seed=seed,
        family_type=family.family_type,
        family_params=family_params,
    )
