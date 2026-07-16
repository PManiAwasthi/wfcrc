"""Unit tests for :mod:`wfcrc.calibration.pipeline`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.unit.calibration._helpers import calibration_config, monotone_loss_table
from wfcrc.ambiguity import CVaRFamily
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.pipeline import PipelineResult, run_calibration_pipeline
from wfcrc.evaluation.verifier import CheckResult, Verifier
from wfcrc.exceptions import FamilyError


def test_pipeline_without_verifier_returns_none_verification() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)

    result = run_calibration_pipeline(table, family, cfg, seed=0)

    assert isinstance(result, PipelineResult)
    assert isinstance(result.calibration, CalibrationResult)
    assert result.verification is None


def test_pipeline_without_verifier_assert_ok_is_a_noop() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)

    result = run_calibration_pipeline(table, family, cfg, seed=0)
    result.assert_ok()  # must not raise


def test_pipeline_with_verifier_merges_precondition_and_calibration_reports() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)

    result = run_calibration_pipeline(table, family, cfg, seed=0, verifier=Verifier())

    assert result.verification is not None
    names = {item.name for item in result.verification.items}
    assert "p2_monotone_nonincreasing" in names  # from check_preconditions
    assert "reproducibility" in names  # from check_calibration
    assert result.verification.passed is True


def test_pipeline_matches_direct_calibrator_call() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)

    direct = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    piped = run_calibration_pipeline(table, family, cfg, seed=0)

    assert piped.calibration.lambda_hat == direct.lambda_hat
    assert piped.calibration.empty_flag == direct.empty_flag


def test_pipeline_accepts_an_injected_calibrator() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    injected = WFCRCCalibrator()

    result = run_calibration_pipeline(table, family, cfg, seed=0, calibrator=injected)

    assert result.calibration.lambda_hat in table.lambda_grid


def test_pipeline_propagates_lambda_grid_mismatch() -> None:
    table = monotone_loss_table()
    family = CVaRFamily(beta=0.2)
    wrong_cfg = calibration_config(table)
    wrong_cfg = type(wrong_cfg)(
        alpha=wrong_cfg.alpha,
        B=wrong_cfg.B,
        pi=wrong_cfg.pi,
        lambda_grid=tuple(np.array(wrong_cfg.lambda_grid) + 100.0),
    )

    with pytest.raises(ValueError, match="lambda_grid"):
        run_calibration_pipeline(table, family, wrong_cfg, seed=0)


def test_pipeline_propagates_family_error_from_calibrator() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)

    class _UnsupportedFamily:
        @property
        def family_type(self) -> str:
            return "not_a_real_family"

    with pytest.raises(FamilyError):
        run_calibration_pipeline(table, cfg=cfg, family=_UnsupportedFamily(), seed=0)  # type: ignore[arg-type]


def test_pipeline_result_assert_ok_raises_when_verification_failed() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    result = run_calibration_pipeline(table, family, cfg, seed=0, verifier=Verifier())
    assert result.verification is not None

    extra_items = (*result.verification.items, CheckResult("fake", False, "bad"))
    failed = dataclasses.replace(
        result,
        verification=dataclasses.replace(result.verification, items=extra_items),
    )
    with pytest.raises(Exception, match="fake"):
        failed.assert_ok()
