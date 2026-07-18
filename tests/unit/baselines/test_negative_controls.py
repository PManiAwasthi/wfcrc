"""Tests for `wfcrc.baselines.negative_controls` (pooled K-fold, total-n, fixed-eta).

Cross-checks `PooledKFoldWFCRC`/`TotalNInflationWFCRC` against the original,
already-validated test-only harness in
`tests/unit/calibration/test_negative_controls.py` — this promotion must be
formula-for-formula identical, not merely "similar."
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.baselines._helpers import cfg, loss_table, population_losses
from tests.unit.calibration.test_negative_controls import (
    _pooled_k_fold_lambda_hat,
    _total_n_lambda_hat,
)
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.kl import KLDualParams, KLFamily
from wfcrc.baselines.negative_controls import (
    FixedEtaWFCRC,
    PooledKFoldWFCRC,
    TotalNInflationWFCRC,
)
from wfcrc.exceptions import BaselineError


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_total_n_inflation_matches_original_harness_exactly(seed: int) -> None:
    family = CVaRFamily(beta=0.2)
    values = population_losses(seed=1000 + seed, n=40)
    table = loss_table(values)
    config = cfg(alpha=0.25, pi=0.5)

    expected = _total_n_lambda_hat(values, family, seed=seed, alpha=0.25)
    actual = TotalNInflationWFCRC(family).calibrate(table, config, seed=seed).lambda_hat
    assert actual == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pooled_k_fold_matches_original_harness_exactly(seed: int) -> None:
    family = CVaRFamily(beta=0.2)
    values = population_losses(seed=1000 + seed, n=40)
    table = loss_table(values)
    config = cfg(alpha=0.25, pi=0.5)

    expected = _pooled_k_fold_lambda_hat(values, family, seed=seed, k_folds=5, alpha=0.25)
    actual = PooledKFoldWFCRC(family, k_folds=5).calibrate(table, config, seed=seed).lambda_hat
    assert actual == expected


def test_pooled_k_fold_rejects_too_few_folds() -> None:
    with pytest.raises(BaselineError):
        PooledKFoldWFCRC(CVaRFamily(beta=0.2), k_folds=1)


def test_total_n_inflation_empty_selection_falls_back_to_lambda_max() -> None:
    family = CVaRFamily(beta=0.2)
    table = loss_table(population_losses(seed=42, n=40))
    # alpha effectively unreachable at every grid point -> empty-selection fallback.
    result = TotalNInflationWFCRC(family).calibrate(table, cfg(alpha=1e-9, pi=0.5), seed=0)
    assert result.lambda_hat == float(table.lambda_grid[-1])
    assert result.empty_flag is True


def test_total_n_inflation_uses_n_not_n_b_denominator() -> None:
    family = CVaRFamily(beta=0.2)
    table = loss_table(population_losses(seed=42, n=40))
    result = TotalNInflationWFCRC(family).calibrate(table, cfg(alpha=0.25, pi=0.5), seed=0)
    assert result.n_b is None
    assert result.diagnostics["n"] == 40


def test_baseline_names() -> None:
    family = CVaRFamily(beta=0.2)
    assert TotalNInflationWFCRC(family).baseline_name == "total_n_inflation"
    assert PooledKFoldWFCRC(family).baseline_name == "pooled_k_fold"
    assert FixedEtaWFCRC(family, 0.1).baseline_name == "fixed_eta"


def test_fixed_eta_cvar_matches_hand_computed_formula() -> None:
    family = CVaRFamily(beta=0.2)
    values = population_losses(seed=9, n=30)
    table = loss_table(values)
    theta = 0.05
    config = cfg(alpha=0.3)

    result = FixedEtaWFCRC(family, theta).calibrate(table, config, seed=0)

    n = table.shape[0]
    b_tilde = family.btil(theta, config.B)
    assert result.b_tilde == b_tilde

    def g(lam: float) -> float:
        col = table.column(lam)
        r_hat = float(np.mean(family.transform(col, theta)))
        return (n / (n + 1)) * r_hat + b_tilde / (n + 1)

    grid = table.lambda_grid
    feasible = [float(lam) for lam in grid if g(float(lam)) <= config.alpha]
    expected_lambda_hat = min(feasible) if feasible else float(grid[-1])
    assert result.lambda_hat == expected_lambda_hat


def test_fixed_eta_kl_accepts_kl_dual_params_theta() -> None:
    family = KLFamily(rho=0.1)
    table = loss_table(population_losses(seed=9, n=30))
    theta = KLDualParams(eta=1.0, mu=0.0)
    result = FixedEtaWFCRC(family, theta).calibrate(table, cfg(alpha=0.3), seed=0)
    assert result.lambda_hat in table.lambda_grid


def test_fixed_eta_is_deterministic_and_seed_independent() -> None:
    family = CVaRFamily(beta=0.2)
    table = loss_table(population_losses(seed=9, n=30))
    a = FixedEtaWFCRC(family, 0.05).calibrate(table, cfg(alpha=0.3), seed=0)
    b = FixedEtaWFCRC(family, 0.05).calibrate(table, cfg(alpha=0.3), seed=555)
    assert a.lambda_hat == b.lambda_hat


def test_fixed_eta_is_no_more_liberal_than_data_adaptive_single_split_on_average() -> None:
    """Weak sanity check on the qualitative E7 expectation: fixed-eta is valid but conservative.

    Not a full replication of the Blueprint's own empirical E7 experiment
    (that is out of MS9's documentation-only-methodology / additive-code
    scope, per `docs/EXPERIMENT_PROTOCOL.md`) -- just a check that a
    reasonably-chosen fixed eta does not systematically produce *larger*
    lambda_hat (i.e. smaller, less conservative sets) than the data-
    adaptive single-split procedure across many resamples, which would
    contradict the qualitative "valid but conservative" expectation.
    """
    from wfcrc.calibration.calibrator import WFCRCCalibrator

    family = CVaRFamily(beta=0.2)
    config = cfg(alpha=0.25, pi=0.5)
    fixed_theta = 0.5  # a deliberately conservative, data-independent quantile guess

    adaptive_lambdas = []
    fixed_lambdas = []
    for r in range(30):
        values = population_losses(seed=2000 + r, n=40)
        table = loss_table(values)
        adaptive_lambdas.append(
            WFCRCCalibrator().calibrate(table, family, config, seed=r).lambda_hat
        )
        fixed_result = FixedEtaWFCRC(family, fixed_theta).calibrate(table, config, seed=r)
        fixed_lambdas.append(fixed_result.lambda_hat)

    # Conservative (larger B_tilde-inflated bound) => selects a *larger* lambda
    # threshold no more often, on average, than the data-adaptive procedure --
    # i.e. fixed-eta's mean deployed lambda_hat should not be smaller (looser).
    assert np.mean(fixed_lambdas) >= np.mean(adaptive_lambdas) - 1e-9
