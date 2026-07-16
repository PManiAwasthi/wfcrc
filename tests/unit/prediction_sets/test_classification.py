"""Unit tests for :class:`wfcrc.prediction_sets.classification.ThresholdSets` (LAC).

Test names mirror the frozen formula (`score_k >= 1 - lambda`) and the MS2
Implementation Spec's own named edge cases (empty/full-set boundaries,
nestedness across a grid) directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.prediction_sets.classification import ThresholdSets


def test_construct_matches_the_frozen_lac_formula() -> None:
    score = np.array([0.1, 0.5, 0.9])
    sets = ThresholdSets()
    result = sets.construct(score, 0.4)  # threshold = 0.6
    np.testing.assert_array_equal(result, np.array([False, False, True]))


def test_construct_at_lambda_zero_gives_the_empty_set_for_sub_certain_scores() -> None:
    score = np.array([0.1, 0.5, 0.9999])
    sets = ThresholdSets()
    result = sets.construct(score, 0.0)  # threshold = 1.0
    assert not np.any(result)


def test_construct_at_lambda_zero_includes_a_score_of_exactly_one() -> None:
    score = np.array([1.0, 0.5])
    sets = ThresholdSets()
    result = sets.construct(score, 0.0)
    np.testing.assert_array_equal(result, np.array([True, False]))


def test_construct_at_lambda_one_gives_the_full_set() -> None:
    score = np.array([0.0, 0.3, 1.0])
    sets = ThresholdSets()
    result = sets.construct(score, 1.0)  # threshold = 0.0
    assert np.all(result)


def test_construct_rejects_non_1d_score() -> None:
    sets = ThresholdSets()
    with pytest.raises(ValueError, match="1-D"):
        sets.construct(np.array([[0.1, 0.2], [0.3, 0.4]]), 0.5)


@pytest.mark.parametrize("bad_score", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
def test_construct_rejects_score_outside_unit_interval(bad_score: np.ndarray) -> None:
    sets = ThresholdSets()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sets.construct(bad_score, 0.5)


@pytest.mark.parametrize("bad_lam", [-0.01, 1.01])
def test_construct_rejects_lambda_outside_unit_interval(bad_lam: float) -> None:
    sets = ThresholdSets()
    with pytest.raises(ValueError, match="lam must be in"):
        sets.construct(np.array([0.5]), bad_lam)


def test_name_is_threshold() -> None:
    assert ThresholdSets().name() == "threshold"


def test_assert_nested_holds_across_the_full_unit_grid() -> None:
    rng = np.random.default_rng(0)
    score = rng.uniform(0.0, 1.0, size=20)
    grid = np.linspace(0.0, 1.0, 41)
    assert ThresholdSets().assert_nested(score, grid) is True


@pytest.mark.parametrize("seed", range(5))
def test_assert_nested_holds_for_random_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.0, 1.0, size=10)
    grid = np.sort(rng.uniform(0.0, 1.0, size=8))
    grid = np.unique(grid)
    if grid.size < 2:
        pytest.skip("degenerate grid after dedup")
    assert ThresholdSets().assert_nested(score, grid) is True
