"""Unit tests for :class:`wfcrc.losses.fpr.FPRLoss`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.losses._helpers import shrinking_sets
from wfcrc.losses.fpr import FPRLoss


def test_name_and_upper_bound() -> None:
    loss = FPRLoss()
    assert loss.name() == "fpr"
    assert loss.upper_bound() == 1.0


def test_no_false_positives_gives_zero_loss() -> None:
    loss = FPRLoss()
    label = np.array([True, True, False, False])
    predicted = np.array([True, True, False, False])
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_all_negatives_falsely_included_gives_full_loss() -> None:
    loss = FPRLoss()
    label = np.array([True, True, False, False])
    predicted = np.array([True, True, True, True])
    assert loss.evaluate(predicted, label) == pytest.approx(1.0)


def test_partial_false_positive_known_value() -> None:
    loss = FPRLoss()
    label = np.array([False, False, False, False])
    predicted = np.array([True, False, True, False])
    # 2 of 4 negatives falsely included -> FPR = 0.5
    assert loss.evaluate(predicted, label) == pytest.approx(0.5)


def test_all_positive_label_convention_is_zero() -> None:
    loss = FPRLoss()
    label = np.ones(5, dtype=np.bool_)
    predicted = np.ones(5, dtype=np.bool_)
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_missed_positives_do_not_affect_fpr() -> None:
    # FPR only cares about falsely included negatives, not missed positives.
    loss = FPRLoss()
    label = np.array([True, True, True, False])
    predicted = np.array([False, False, False, False])
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_result_always_within_bounds() -> None:
    loss = FPRLoss()
    rng = np.random.default_rng(1)
    for _ in range(20):
        label = rng.random(16) < 0.5
        predicted = rng.random(16) < 0.5
        value = loss.evaluate(predicted, label)
        assert 0.0 <= value <= loss.upper_bound()


def test_monotone_non_increasing_under_shrinking_set_family() -> None:
    # FPR requires the OPPOSITE pairing from FNR: it is non-increasing in
    # lambda only when the set family shrinks (erodes) as lambda grows.
    loss = FPRLoss()
    label = np.zeros((4, 4), dtype=np.bool_)
    label[1:3, 1:3] = True
    sets = shrinking_sets(n_steps=6, shape=(4, 4))
    losses_by_lambda = [loss.evaluate(s, label) for s in sets]
    assert loss.assert_monotone(losses_by_lambda) is True


def test_growing_set_family_violates_fpr_monotonicity() -> None:
    # Sanity check that the pairing direction actually matters: FPR under a
    # GROWING family is non-decreasing (the wrong direction for A2/P-2),
    # not non-increasing.
    from tests.unit.losses._helpers import growing_sets

    loss = FPRLoss()
    label = np.zeros((4, 4), dtype=np.bool_)  # all-negative label
    sets = growing_sets(n_steps=6, shape=(4, 4))
    losses_by_lambda = [loss.evaluate(s, label) for s in sets]
    assert losses_by_lambda == sorted(losses_by_lambda)  # strictly non-decreasing
    assert losses_by_lambda[0] < losses_by_lambda[-1]  # genuinely increases
    assert loss.assert_monotone(losses_by_lambda) is False


def test_deterministic_given_same_inputs() -> None:
    loss = FPRLoss()
    label = np.array([True, False, True, True])
    predicted = np.array([True, True, False, True])
    assert loss.evaluate(predicted, label) == loss.evaluate(predicted, label)
