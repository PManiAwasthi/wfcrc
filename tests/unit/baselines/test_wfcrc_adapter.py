"""Tests for `wfcrc.baselines.wfcrc_adapter.WFCRCAdapter`."""

from __future__ import annotations

from tests.unit.baselines._helpers import cfg, loss_table, population_losses
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.baselines.wfcrc_adapter import WFCRCAdapter
from wfcrc.calibration.calibrator import WFCRCCalibrator


def test_adapter_matches_direct_wfcrc_calibrator_call() -> None:
    family = CVaRFamily(beta=0.2)
    table = loss_table(population_losses(seed=1, n=40))
    config = cfg()

    direct = WFCRCCalibrator().calibrate(table, family, config, seed=7)
    adapted = WFCRCAdapter(family).calibrate(table, config, seed=7)

    assert adapted.lambda_hat == direct.lambda_hat
    assert adapted.empty_flag == direct.empty_flag
    assert adapted.n_a == direct.n_a
    assert adapted.n_b == direct.n_b
    assert adapted.b_tilde == direct.b_tilde
    assert adapted.r_hat_b == direct.r_hat_b


def test_baseline_name_is_wfcrc() -> None:
    assert WFCRCAdapter(CVaRFamily(beta=0.2)).baseline_name == "wfcrc"


def test_adapter_accepts_injected_calibrator() -> None:
    family = CVaRFamily(beta=0.2)
    calibrator = WFCRCCalibrator()
    adapter = WFCRCAdapter(family, calibrator=calibrator)
    assert adapter._calibrator is calibrator
