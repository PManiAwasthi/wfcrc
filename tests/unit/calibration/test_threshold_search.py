"""Unit tests for :class:`wfcrc.calibration.threshold_search.ThresholdSearch`."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.calibration.threshold_search import ThresholdSearch


def test_finds_smallest_lambda_satisfying_criterion() -> None:
    grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    def g(lam: float) -> float:
        return 1.0 - lam  # non-increasing

    lam_hat = ThresholdSearch().search(g, grid, alpha=0.7, default=grid[-1])
    # g(lam) <= 0.7  <=>  lam >= 0.3; smallest grid point satisfying this is 0.3.
    assert lam_hat == pytest.approx(0.3)


def test_matches_brute_force_scan() -> None:
    rng = np.random.default_rng(0)
    grid = np.sort(rng.uniform(0.0, 1.0, size=50))
    # A genuinely non-increasing g built from a random non-increasing sequence.
    decreasing_values = np.sort(rng.uniform(0.0, 1.0, size=50))[::-1]

    def g(lam: float) -> float:
        idx = int(np.searchsorted(grid, lam))
        idx = min(idx, len(grid) - 1)
        return float(decreasing_values[idx])

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        expected = None
        for lam, val in zip(grid, decreasing_values, strict=True):
            if val <= alpha:
                expected = lam
                break
        result = ThresholdSearch().search(g, grid, alpha, default=grid[-1])
        if expected is None:
            assert result == pytest.approx(grid[-1])
        else:
            assert result == pytest.approx(expected)


def test_empty_selection_returns_default() -> None:
    grid = np.array([0.0, 0.5, 1.0])

    def g(lam: float) -> float:
        return 10.0  # never satisfies any reasonable alpha

    result = ThresholdSearch().search(g, grid, alpha=1.0, default=99.0)
    assert result == pytest.approx(99.0)


def test_all_points_satisfy_returns_smallest() -> None:
    grid = np.array([0.0, 0.5, 1.0])

    def g(lam: float) -> float:
        return -1.0  # always satisfies

    result = ThresholdSearch().search(g, grid, alpha=0.0, default=grid[-1])
    assert result == pytest.approx(0.0)


def test_single_point_grid_satisfying() -> None:
    grid = np.array([0.5])
    result = ThresholdSearch().search(lambda lam: 0.0, grid, alpha=1.0, default=grid[-1])
    assert result == pytest.approx(0.5)


def test_single_point_grid_not_satisfying() -> None:
    grid = np.array([0.5])
    result = ThresholdSearch().search(lambda lam: 10.0, grid, alpha=1.0, default=-1.0)
    assert result == pytest.approx(-1.0)


def test_rejects_empty_grid() -> None:
    with pytest.raises(ValueError):
        ThresholdSearch().search(lambda lam: 0.0, np.array([]), alpha=1.0, default=0.0)


def test_rejects_non_increasing_grid() -> None:
    with pytest.raises(ValueError):
        ThresholdSearch().search(lambda lam: 0.0, np.array([0.0, 0.5, 0.2]), alpha=1.0, default=0.0)


def test_boundary_exact_equality_counts_as_satisfying() -> None:
    grid = np.array([0.0, 0.5, 1.0])

    def g(lam: float) -> float:
        return 1.0 - lam

    # At lam=0.5, g=0.5, exactly equal to alpha -> must count as satisfying (<=).
    result = ThresholdSearch().search(g, grid, alpha=0.5, default=grid[-1])
    assert result == pytest.approx(0.5)


def test_evaluates_g_lazily_not_at_every_grid_point() -> None:
    grid = np.linspace(0.0, 1.0, 1000)
    call_count = 0

    def g(lam: float) -> float:
        nonlocal call_count
        call_count += 1
        return 1.0 - lam

    ThresholdSearch().search(g, grid, alpha=0.5, default=grid[-1])
    # Binary search over 1000 points should take O(log2(1000)) ~ 10 evals,
    # plus the initial default-feasibility check; must not be O(n).
    assert call_count < 20
