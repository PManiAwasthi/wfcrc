"""Tests for `wfcrc.baselines.lac.SplitConformalLAC`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.baselines._helpers import cfg, loss_table, miscoverage_losses
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.calibration.loss_table import LossTable
from wfcrc.exceptions import BaselineError


def test_lac_threshold_achieves_the_exact_order_statistic_coverage() -> None:
    values = miscoverage_losses(seed=1, n=50)
    table = loss_table(values)
    alpha = 0.2
    result = SplitConformalLAC().calibrate(table, cfg(alpha=alpha), seed=0)

    n = table.shape[0]
    k = math.ceil((n + 1) * (1 - alpha))
    # empirical miscoverage rate at lambda_hat must be <= (n-k)/n
    assert result.r_hat_b is not None
    assert result.r_hat_b <= (n - k) / n + 1e-12
    assert result.diagnostics["k"] == k


def test_lac_is_deterministic_and_seed_independent() -> None:
    table = loss_table(miscoverage_losses(seed=2, n=40))
    a = SplitConformalLAC().calibrate(table, cfg(alpha=0.1), seed=0)
    b = SplitConformalLAC().calibrate(table, cfg(alpha=0.1), seed=999)
    assert a.lambda_hat == b.lambda_hat


def test_lac_rejects_alpha_out_of_range() -> None:
    table = loss_table(miscoverage_losses(seed=2, n=40))
    with pytest.raises(BaselineError):
        SplitConformalLAC().calibrate(table, cfg(alpha=0.0), seed=0)
    with pytest.raises(BaselineError):
        SplitConformalLAC().calibrate(table, cfg(alpha=1.0), seed=0)


def test_lac_rejects_n_too_small_for_order_statistic() -> None:
    # n=1, alpha=0.01 -> k = ceil(2*0.99) = 2 > n=1
    table = loss_table(miscoverage_losses(seed=2, n=1))
    with pytest.raises(BaselineError):
        SplitConformalLAC().calibrate(table, cfg(alpha=0.01), seed=0)


def test_lac_empty_flag_true_when_even_lambda_max_undercovers() -> None:
    # All-ones miscoverage column at every lambda -> nothing ever covers.
    n, t = 10, 5
    values = np.ones((n, t), dtype=np.float64)
    # LossTable requires a strictly increasing lambda_grid; values need not vary.
    grid = np.linspace(0.0, 0.9, t)
    lt = LossTable(values=values, lambda_grid=grid)
    result = SplitConformalLAC().calibrate(lt, cfg(alpha=0.1), seed=0)
    assert result.empty_flag is True
    assert result.lambda_hat == float(grid[-1])


def test_baseline_name_is_lac() -> None:
    assert SplitConformalLAC().baseline_name == "lac"
