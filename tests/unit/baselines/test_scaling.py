"""Tests for `wfcrc.baselines.scaling` (temperature scaling, selective scaling)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.baselines._helpers import cfg, loss_table, miscoverage_losses
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.baselines.scaling import (
    TemperatureScaledLAC,
    apply_selective_threshold,
    apply_temperature,
    fit_selective_threshold,
    fit_temperature,
)
from wfcrc.exceptions import BaselineError


def _synthetic_logits(
    seed: int, n: int, k: int, separation: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(n, k)` logits with a clear, separation-controlled true class per row."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    logits = rng.normal(scale=0.5, size=(n, k))
    logits[np.arange(n), labels] += separation
    return logits, labels


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = apply_temperature(logits, temperature)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels])))


def test_fit_temperature_achieves_nll_no_worse_than_the_identity_temperature() -> None:
    logits, labels = _synthetic_logits(seed=1, n=2000, k=4, separation=2.0)
    t_hat = fit_temperature(logits, labels)
    assert 0.0 < t_hat < 1e3
    assert _nll(logits, labels, t_hat) <= _nll(logits, labels, 1.0) + 1e-9


def test_fit_temperature_is_deterministic() -> None:
    logits, labels = _synthetic_logits(seed=1, n=200, k=4, separation=2.0)
    first = fit_temperature(logits, labels)
    second = fit_temperature(logits, labels)
    assert first == second


def test_apply_temperature_returns_a_valid_probability_simplex() -> None:
    logits, _ = _synthetic_logits(seed=1, n=50, k=5, separation=1.0)
    probs = apply_temperature(logits, temperature=2.0)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    assert np.all(probs >= 0.0)


def test_temperature_high_flattens_toward_uniform() -> None:
    logits, _ = _synthetic_logits(seed=1, n=1, k=4, separation=5.0)
    probs_hot = apply_temperature(logits, temperature=1.0)
    probs_flat = apply_temperature(logits, temperature=1000.0)
    # A very high temperature should push probabilities toward uniform (lower max).
    assert np.max(probs_flat) < np.max(probs_hot)


def test_apply_temperature_rejects_nonpositive_temperature() -> None:
    logits, _ = _synthetic_logits(seed=1, n=5, k=3, separation=1.0)
    with pytest.raises(BaselineError):
        apply_temperature(logits, temperature=0.0)
    with pytest.raises(BaselineError):
        apply_temperature(logits, temperature=-1.0)


def test_fit_temperature_rejects_bad_shapes() -> None:
    logits, labels = _synthetic_logits(seed=1, n=10, k=3, separation=1.0)
    with pytest.raises(BaselineError):
        fit_temperature(logits[:, 0], labels)  # 1-D logits
    with pytest.raises(BaselineError):
        fit_temperature(logits, labels[:-1])  # mismatched length
    with pytest.raises(BaselineError):
        fit_temperature(logits, labels + 100)  # out-of-range label
    with pytest.raises(BaselineError):
        fit_temperature(logits, labels, t_min=5.0, t_max=1.0)  # bad bounds


def test_fit_selective_threshold_selects_largest_coverage_meeting_target_risk() -> None:
    rng = np.random.default_rng(0)
    n = 200
    confidence = rng.uniform(0.0, 1.0, size=n)
    # Higher confidence -> lower loss, by construction, with some noise.
    losses = np.clip(1.0 - confidence + rng.normal(scale=0.05, size=n), 0.0, 1.0)
    losses = (losses > 0.5).astype(np.float64)  # binarize to a 0/1 error indicator

    tau = fit_selective_threshold(confidence, losses, target_risk=0.1)
    mask = apply_selective_threshold(confidence, tau)
    assert mask.sum() > 0
    assert float(np.mean(losses[mask])) <= 0.1 + 1e-9


def test_fit_selective_threshold_returns_inf_when_infeasible() -> None:
    n = 20
    confidence = np.linspace(0.0, 1.0, n)
    losses = np.ones(n)  # every example always errs; no achievable target < 1
    tau = fit_selective_threshold(confidence, losses, target_risk=0.05)
    assert tau == float("inf")
    mask = apply_selective_threshold(confidence, tau)
    assert not mask.any()


def test_fit_selective_threshold_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(BaselineError):
        fit_selective_threshold(np.array([1.0, 2.0]), np.array([1.0]), target_risk=0.1)
    with pytest.raises(BaselineError):
        fit_selective_threshold(np.array([]), np.array([]), target_risk=0.1)


def test_temperature_scaled_lac_matches_plain_lac_over_the_same_loss_table() -> None:
    table = loss_table(miscoverage_losses(seed=4, n=40))
    config = cfg(alpha=0.2)
    plain = SplitConformalLAC().calibrate(table, config, seed=0)
    wrapped = TemperatureScaledLAC().calibrate(table, config, seed=0)
    assert plain.lambda_hat == wrapped.lambda_hat


def test_baseline_name_is_temperature_scaled_lac() -> None:
    assert TemperatureScaledLAC().baseline_name == "temperature_scaled_lac"
