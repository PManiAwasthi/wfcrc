"""Tests for `wfcrc.baselines.ensembles` (MC-dropout / deep-ensemble aggregation)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.baselines._helpers import cfg, loss_table, miscoverage_losses
from wfcrc.baselines.ensembles import (
    EnsembleAggregatedLAC,
    aggregate_deep_ensemble_scores,
    aggregate_mc_dropout_scores,
)
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.exceptions import BaselineError


def test_aggregate_mc_dropout_scores_mean_and_variance() -> None:
    rng = np.random.default_rng(0)
    k, n, c = 10, 5, 3
    stack = rng.normal(loc=0.0, scale=1.0, size=(k, n, c))
    mean, var = aggregate_mc_dropout_scores(stack)
    assert mean.shape == (n, c)
    assert var.shape == (n, c)
    assert np.allclose(mean, np.mean(stack, axis=0))
    assert np.allclose(var, np.var(stack, axis=0, ddof=1))


def test_aggregate_deep_ensemble_scores_same_arithmetic_as_mc_dropout() -> None:
    rng = np.random.default_rng(1)
    stack = rng.normal(size=(6, 4))
    mean_a, var_a = aggregate_mc_dropout_scores(stack)
    mean_b, var_b = aggregate_deep_ensemble_scores(stack)
    assert np.array_equal(mean_a, mean_b)
    assert np.array_equal(var_a, var_b)


def test_aggregate_rejects_too_few_samples() -> None:
    with pytest.raises(BaselineError):
        aggregate_mc_dropout_scores(np.ones((1, 5)))
    with pytest.raises(BaselineError):
        aggregate_deep_ensemble_scores(np.ones((1, 5)))


def test_aggregate_rejects_empty_stack() -> None:
    with pytest.raises(BaselineError):
        aggregate_mc_dropout_scores(np.ones((0, 5)))


def test_aggregate_rejects_empty_along_a_non_leading_axis() -> None:
    # axis 0 has >= 2 samples, but the per-example score shape is empty --
    # a distinct failure mode from "too few samples along axis 0" (arr.size
    # == 0 despite arr.shape[0] >= 2).
    with pytest.raises(BaselineError):
        aggregate_mc_dropout_scores(np.ones((3, 0)))


def test_ensemble_aggregated_lac_matches_plain_lac_over_the_same_loss_table() -> None:
    table = loss_table(miscoverage_losses(seed=6, n=40))
    config = cfg(alpha=0.15)
    plain = SplitConformalLAC().calibrate(table, config, seed=0)
    wrapped = EnsembleAggregatedLAC().calibrate(table, config, seed=0)
    assert plain.lambda_hat == wrapped.lambda_hat


def test_baseline_name_is_ensemble_aggregated_lac() -> None:
    assert EnsembleAggregatedLAC().baseline_name == "ensemble_aggregated_lac"
