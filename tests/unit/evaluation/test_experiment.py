"""Unit tests for :mod:`wfcrc.evaluation.experiment`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.evaluation._helpers import calibration_config, monotone_loss_table
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.evaluation.experiment import ExperimentReport, run_experiment
from wfcrc.evaluation.verifier import Verifier
from wfcrc.exceptions import FamilyError


def test_run_experiment_dual_branch_populates_worst_case_and_duality_gap() -> None:
    cal_table = monotone_loss_table(n=60, seed=0)
    test_table = monotone_loss_table(n=40, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)

    report = run_experiment(cal_table, test_table, family, cfg, seed=0)

    assert isinstance(report, ExperimentReport)
    assert isinstance(report.calibration, CalibrationResult)
    assert report.verification is None
    assert "realized_marginal_risk" in report.metrics
    assert "realized_worst_case_risk" in report.metrics
    assert "duality_gap" in report.metrics
    assert "effective_sizes" in report.metrics
    assert isinstance(report.config_hash, str) and report.config_hash


def test_run_experiment_with_verifier_populates_verification() -> None:
    cal_table = monotone_loss_table(n=60, seed=0)
    test_table = monotone_loss_table(n=40, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)

    report = run_experiment(cal_table, test_table, family, cfg, seed=0, verifier=Verifier())

    assert report.verification is not None
    assert report.verification.passed is True


def test_run_experiment_with_groups_populates_per_group_risk() -> None:
    cal_table = monotone_loss_table(n=60, seed=0)
    test_table = monotone_loss_table(n=40, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)

    report = run_experiment(
        cal_table, test_table, family, cfg, seed=0, groups=[list(range(20)), list(range(20, 40))]
    )

    assert report.metrics["per_group_risk"] == {
        0: pytest.approx(np.mean(test_table.column(report.calibration.lambda_hat)[:20])),
        1: pytest.approx(np.mean(test_table.column(report.calibration.lambda_hat)[20:40])),
    }


def test_run_experiment_finite_group_branch_has_no_worst_case_or_duality_gap() -> None:
    cal_table = monotone_loss_table(n=20, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    family = FiniteGroupFamily(masks=[tuple(range(0, 10)), tuple(range(10, 20))])
    cfg = calibration_config(cal_table)

    report = run_experiment(cal_table, test_table, family, cfg, seed=0)

    assert "realized_marginal_risk" in report.metrics
    assert "realized_worst_case_risk" not in report.metrics
    assert "duality_gap" not in report.metrics


def test_run_experiment_propagates_lambda_grid_mismatch() -> None:
    cal_table = monotone_loss_table(n=20, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)
    bad_cfg = type(cfg)(
        alpha=cfg.alpha, B=cfg.B, pi=cfg.pi, lambda_grid=tuple(np.array(cfg.lambda_grid) + 5.0)
    )
    with pytest.raises(ValueError, match="lambda_grid"):
        run_experiment(cal_table, test_table, family, bad_cfg, seed=0)


def test_run_experiment_propagates_family_error() -> None:
    cal_table = monotone_loss_table(n=20, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    cfg = calibration_config(cal_table)

    class _UnsupportedFamily:
        @property
        def family_type(self) -> str:
            return "not_a_real_family"

    with pytest.raises(FamilyError):
        run_experiment(cal_table, test_table, _UnsupportedFamily(), cfg, seed=0)  # type: ignore[arg-type]


def test_experiment_report_to_dict_is_json_serializable_shape() -> None:
    cal_table = monotone_loss_table(n=30, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)

    report = run_experiment(cal_table, test_table, family, cfg, seed=0, verifier=Verifier())
    d = report.to_dict()

    assert d["lambda_hat"] == report.calibration.lambda_hat
    assert d["empty_flag"] == report.calibration.empty_flag
    assert d["n_a"] == report.calibration.n_a
    assert d["n_b"] == report.calibration.n_b
    assert d["verification_passed"] is True
    assert d["metrics"] == report.metrics
    assert d["config_hash"] == report.config_hash

    import json

    json.dumps(d)  # must not raise


def test_experiment_report_to_dict_verification_passed_is_none_without_verifier() -> None:
    cal_table = monotone_loss_table(n=30, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)

    report = run_experiment(cal_table, test_table, family, cfg, seed=0)
    assert report.to_dict()["verification_passed"] is None


def test_run_experiment_config_hash_is_deterministic_and_grid_sensitive() -> None:
    cal_table = monotone_loss_table(n=20, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    family = CVaRFamily(beta=0.2)
    cfg = calibration_config(cal_table)

    first = run_experiment(cal_table, test_table, family, cfg, seed=0)
    second = run_experiment(cal_table, test_table, family, cfg, seed=1)
    assert first.config_hash == second.config_hash  # seed isn't part of cfg

    other_table = monotone_loss_table(n=20, seed=0, lambda_max=0.5)
    other_cfg = calibration_config(other_table)
    third = run_experiment(other_table, other_table, family, other_cfg, seed=0)
    assert third.config_hash != first.config_hash


def test_run_experiment_uses_the_provided_loss_table_type() -> None:
    cal_table = monotone_loss_table(n=20, seed=0)
    test_table = monotone_loss_table(n=20, seed=1)
    assert isinstance(cal_table, LossTable)
    assert isinstance(test_table, LossTable)
