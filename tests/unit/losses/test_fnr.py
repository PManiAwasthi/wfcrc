"""Unit tests for :class:`wfcrc.losses.fnr.FNRLoss`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.losses._helpers import growing_sets
from wfcrc.losses.fnr import FNRLoss


def test_name_and_upper_bound() -> None:
    loss = FNRLoss()
    assert loss.name() == "fnr"
    assert loss.upper_bound() == 1.0


def test_perfect_coverage_gives_zero_loss() -> None:
    loss = FNRLoss()
    label = np.array([True, True, False, False])
    predicted = np.array([True, True, True, True])
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_no_coverage_gives_full_loss() -> None:
    loss = FNRLoss()
    label = np.array([True, True, False, False])
    predicted = np.array([False, False, False, False])
    assert loss.evaluate(predicted, label) == pytest.approx(1.0)


def test_partial_coverage_known_value() -> None:
    loss = FNRLoss()
    label = np.array([True, True, True, True])
    predicted = np.array([True, False, True, False])
    # 2 of 4 positives missed -> FNR = 0.5
    assert loss.evaluate(predicted, label) == pytest.approx(0.5)


def test_empty_label_convention_is_zero() -> None:
    loss = FNRLoss()
    label = np.zeros(5, dtype=np.bool_)
    predicted = np.zeros(5, dtype=np.bool_)
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_extra_predicted_elements_do_not_affect_fnr() -> None:
    # FNR only cares about missed positives, not spurious inclusions.
    loss = FNRLoss()
    label = np.array([True, False, False, False])
    predicted = np.array([True, True, True, True])
    assert loss.evaluate(predicted, label) == pytest.approx(0.0)


def test_result_always_within_bounds() -> None:
    loss = FNRLoss()
    rng = np.random.default_rng(0)
    for _ in range(20):
        label = rng.random(16) < 0.5
        predicted = rng.random(16) < 0.5
        value = loss.evaluate(predicted, label)
        assert 0.0 <= value <= loss.upper_bound()


def test_monotone_non_increasing_under_growing_set_family() -> None:
    loss = FNRLoss()
    label = np.zeros((4, 4), dtype=np.bool_)
    label[1:3, 1:3] = True  # a 2x2 block of positives
    sets = growing_sets(n_steps=6, shape=(4, 4))
    losses_by_lambda = [loss.evaluate(s, label) for s in sets]
    assert loss.assert_monotone(losses_by_lambda) is True


def test_deterministic_given_same_inputs() -> None:
    loss = FNRLoss()
    label = np.array([True, False, True, True])
    predicted = np.array([True, True, False, True])
    assert loss.evaluate(predicted, label) == loss.evaluate(predicted, label)


def test_2d_segmentation_shape() -> None:
    loss = FNRLoss()
    label = np.zeros((5, 5), dtype=np.bool_)
    label[2, 2] = True
    predicted = np.zeros((5, 5), dtype=np.bool_)
    assert loss.evaluate(predicted, label) == pytest.approx(1.0)
