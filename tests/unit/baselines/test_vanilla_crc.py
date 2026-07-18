"""Tests for `wfcrc.baselines.vanilla_crc.VanillaCRC`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.baselines._helpers import cfg, loss_table, population_losses
from wfcrc.ambiguity.known_weight import KnownWeightFamily
from wfcrc.baselines.vanilla_crc import VanillaCRC
from wfcrc.calibration.calibrator import WFCRCCalibrator


def test_vanilla_crc_matches_known_weight_branch_with_unit_weights() -> None:
    table = loss_table(population_losses(seed=3, n=30))
    config = cfg()
    n = table.shape[0]

    expected = WFCRCCalibrator().calibrate(
        table, KnownWeightFamily(weights=np.ones(n)), config, seed=0
    )
    actual = VanillaCRC().calibrate(table, config, seed=0)

    assert actual.lambda_hat == expected.lambda_hat
    assert actual.empty_flag == expected.empty_flag
    assert actual.r_hat_b == expected.r_hat_b
    assert actual.diagnostics == expected.diagnostics


def test_vanilla_crc_is_deterministic() -> None:
    table = loss_table(population_losses(seed=3, n=30))
    config = cfg()
    first = VanillaCRC().calibrate(table, config, seed=99)
    second = VanillaCRC().calibrate(table, config, seed=99)
    assert first.lambda_hat == second.lambda_hat


def test_baseline_name_is_vanilla_crc() -> None:
    assert VanillaCRC().baseline_name == "vanilla_crc"


def test_vanilla_crc_seed_is_ignored() -> None:
    table = loss_table(population_losses(seed=3, n=30))
    config = cfg()
    a = VanillaCRC().calibrate(table, config, seed=0)
    b = VanillaCRC().calibrate(table, config, seed=12345)
    assert a.lambda_hat == b.lambda_hat


@pytest.mark.parametrize("n", [2, 5, 50])
def test_vanilla_crc_runs_for_various_n(n: int) -> None:
    table = loss_table(population_losses(seed=7, n=n))
    result = VanillaCRC().calibrate(table, cfg(), seed=0)
    assert result.lambda_hat in table.lambda_grid
