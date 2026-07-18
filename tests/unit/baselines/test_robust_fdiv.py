"""Tests for `wfcrc.baselines.robust_fdiv.RobustFDivergenceCP`."""

from __future__ import annotations

import numpy as np

from tests.unit.baselines._helpers import cfg, loss_table, population_losses
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.kl import KLFamily
from wfcrc.baselines.robust_fdiv import RobustFDivergenceCP


def test_robust_fdiv_returns_a_valid_grid_point() -> None:
    table = loss_table(population_losses(seed=11, n=40))
    family = KLFamily(rho=0.1)
    result = RobustFDivergenceCP(family).calibrate(table, cfg(alpha=0.3), seed=0)
    assert result.lambda_hat in table.lambda_grid
    assert result.b_tilde is not None
    assert result.n_a is None
    assert result.n_b is None


def test_robust_fdiv_is_deterministic_and_seed_independent() -> None:
    table = loss_table(population_losses(seed=11, n=40))
    family = KLFamily(rho=0.1)
    a = RobustFDivergenceCP(family).calibrate(table, cfg(alpha=0.3), seed=0)
    b = RobustFDivergenceCP(family).calibrate(table, cfg(alpha=0.3), seed=777)
    assert a.lambda_hat == b.lambda_hat


def test_robust_fdiv_g_at_lambda_hat_is_at_most_alpha_or_empty() -> None:
    table = loss_table(population_losses(seed=11, n=40))
    family = KLFamily(rho=0.05)
    alpha = 0.25
    result = RobustFDivergenceCP(family).calibrate(table, cfg(alpha=alpha), seed=0)
    n = table.shape[0]
    theta = family.estimate_dual(table.column(result.lambda_hat))
    r_hat = float(np.mean(family.transform(table.column(result.lambda_hat), theta)))
    b_tilde = result.b_tilde
    assert b_tilde is not None
    g = (n / (n + 1)) * r_hat + b_tilde / (n + 1)
    if not result.empty_flag:
        assert g <= alpha + 1e-9


def test_robust_fdiv_accepts_cvar_family_too() -> None:
    table = loss_table(population_losses(seed=11, n=40))
    family = CVaRFamily(beta=0.2)
    result = RobustFDivergenceCP(family).calibrate(table, cfg(alpha=0.3), seed=0)
    assert result.lambda_hat in table.lambda_grid


def test_baseline_name_is_robust_fdiv() -> None:
    assert RobustFDivergenceCP(KLFamily(rho=0.1)).baseline_name == "robust_fdiv"
