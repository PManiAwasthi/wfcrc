"""Unit tests for :class:`wfcrc.losses.miscoverage.MiscoverageLoss`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.losses._helpers import growing_sets
from wfcrc.losses.miscoverage import MiscoverageLoss


def test_name_and_upper_bound() -> None:
    loss = MiscoverageLoss()
    assert loss.name() == "miscoverage"
    assert loss.upper_bound() == 1.0


def test_full_coverage_gives_zero() -> None:
    loss = MiscoverageLoss()
    label = np.array([True, True, False, False])
    predicted = np.array([True, True, True, True])
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_any_missed_positive_gives_one() -> None:
    loss = MiscoverageLoss()
    label = np.array([True, True, False, False])
    predicted = np.array([True, False, True, True])  # misses one positive
    assert loss.evaluate(predicted, label) == pytest.approx(1.0)


def test_empty_label_is_trivially_covered() -> None:
    loss = MiscoverageLoss()
    label = np.zeros(4, dtype=np.bool_)
    predicted = np.zeros(4, dtype=np.bool_)
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_is_binary_valued() -> None:
    loss = MiscoverageLoss()
    rng = np.random.default_rng(2)
    for _ in range(20):
        label = rng.random(16) < 0.5
        predicted = rng.random(16) < 0.5
        value = loss.evaluate(predicted, label)
        assert value in (0.0, 1.0)


def test_extra_predicted_elements_do_not_affect_coverage() -> None:
    loss = MiscoverageLoss()
    label = np.array([True, False, False, False])
    predicted = np.array([True, True, True, True])
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_monotone_non_increasing_under_growing_set_family() -> None:
    loss = MiscoverageLoss()
    label = np.zeros((4, 4), dtype=np.bool_)
    label[0, 0] = True
    label[3, 3] = True
    sets = growing_sets(n_steps=6, shape=(4, 4))
    losses_by_lambda = [loss.evaluate(s, label) for s in sets]
    assert loss.assert_monotone(losses_by_lambda) is True
    # And it must actually reach full coverage by the last (full) set.
    assert losses_by_lambda[-1] == pytest.approx(0.0)


def test_deterministic_given_same_inputs() -> None:
    loss = MiscoverageLoss()
    label = np.array([True, False, True, True])
    predicted = np.array([True, True, False, True])
    assert loss.evaluate(predicted, label) == loss.evaluate(predicted, label)


def test_equivalent_to_fnr_greater_than_zero() -> None:
    # By construction, miscoverage = 1{FN > 0}; cross-check against FNRLoss.
    from wfcrc.losses.fnr import FNRLoss

    fnr_loss = FNRLoss()
    miscov_loss = MiscoverageLoss()
    rng = np.random.default_rng(3)
    for _ in range(30):
        label = rng.random(16) < 0.5
        predicted = rng.random(16) < 0.5
        fnr = fnr_loss.evaluate(predicted, label)
        miscov = miscov_loss.evaluate(predicted, label)
        assert miscov == (1.0 if fnr > 0.0 else 0.0)
