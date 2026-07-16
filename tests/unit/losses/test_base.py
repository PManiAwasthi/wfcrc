"""Unit tests for :class:`wfcrc.losses.base.LossEvaluator`."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.losses.fnr import FNRLoss


def test_assert_monotone_accepts_non_increasing_sequence() -> None:
    loss = FNRLoss()
    assert loss.assert_monotone([1.0, 0.8, 0.8, 0.3, 0.0]) is True


def test_assert_monotone_rejects_increasing_sequence() -> None:
    loss = FNRLoss()
    assert loss.assert_monotone([0.5, 0.6, 0.4]) is False


def test_assert_monotone_single_element_is_trivially_monotone() -> None:
    loss = FNRLoss()
    assert loss.assert_monotone([0.5]) is True


def test_assert_monotone_rejects_empty_sequence() -> None:
    loss = FNRLoss()
    with pytest.raises(ValueError):
        loss.assert_monotone([])


def test_assert_monotone_tolerates_floating_point_noise() -> None:
    loss = FNRLoss()
    # A tiny increase within tolerance must not be flagged as a violation.
    assert loss.assert_monotone([0.5, 0.5 + 1e-12, 0.4]) is True


def test_assert_monotone_flags_violation_beyond_tolerance() -> None:
    loss = FNRLoss()
    assert loss.assert_monotone([0.5, 0.5001, 0.4], tol=1e-9) is False


def test_validate_shapes_rejects_non_bool_predicted_set() -> None:
    loss = FNRLoss()
    predicted = np.ones((2, 2), dtype=np.float64)
    label = np.ones((2, 2), dtype=np.bool_)
    with pytest.raises(ValueError):
        loss.evaluate(predicted, label)


def test_validate_shapes_rejects_non_bool_label() -> None:
    loss = FNRLoss()
    predicted = np.ones((2, 2), dtype=np.bool_)
    label = np.ones((2, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        loss.evaluate(predicted, label)


def test_validate_shapes_rejects_shape_mismatch() -> None:
    loss = FNRLoss()
    predicted = np.ones((2, 2), dtype=np.bool_)
    label = np.ones((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError):
        loss.evaluate(predicted, label)
