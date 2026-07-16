"""Unit tests for :class:`wfcrc.prediction_sets.base.PredictionSetConstructor`.

Uses small local concrete subclasses (rather than
:class:`~wfcrc.prediction_sets.classification.ThresholdSets`) to exercise
:meth:`~wfcrc.prediction_sets.base.PredictionSetConstructor.assert_nested`
in isolation, including a deliberately non-nested constructor to hit the
`False` branch.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from wfcrc.prediction_sets.base import PredictionSetConstructor


class _GrowingSets(PredictionSetConstructor):
    """A trivially nested family: `C_lambda = {0, ..., floor(lambda)}` over a fixed universe."""

    def __init__(self, size: int = 5) -> None:
        self.size = size

    def construct(self, score: ArrayLike, lam: float) -> NDArray[np.bool_]:
        del score
        k = int(lam)
        out = np.zeros(self.size, dtype=bool)
        out[: min(k + 1, self.size)] = True
        return out

    def name(self) -> str:
        return "growing"


class _NonNestedSets(PredictionSetConstructor):
    """A constructor that violates P-1: each lambda selects an unrelated singleton."""

    def construct(self, score: ArrayLike, lam: float) -> NDArray[np.bool_]:
        del score
        size = 5
        out = np.zeros(size, dtype=bool)
        out[int(lam) % size] = True
        return out

    def name(self) -> str:
        return "non_nested"


def test_assert_nested_true_for_a_genuinely_nested_family() -> None:
    constructor = _GrowingSets(size=5)
    assert constructor.assert_nested(None, [0.0, 1.0, 2.0, 3.0]) is True


def test_assert_nested_false_for_a_non_nested_family() -> None:
    constructor = _NonNestedSets()
    assert constructor.assert_nested(None, [0.0, 1.0, 2.0]) is False


def test_assert_nested_single_grid_point_is_trivially_true() -> None:
    constructor = _GrowingSets()
    assert constructor.assert_nested(None, [2.0]) is True


def test_assert_nested_rejects_empty_grid() -> None:
    constructor = _GrowingSets()
    with pytest.raises(ValueError, match="non-empty"):
        constructor.assert_nested(None, [])


def test_assert_nested_rejects_non_increasing_grid() -> None:
    constructor = _GrowingSets()
    with pytest.raises(ValueError, match="strictly increasing"):
        constructor.assert_nested(None, [1.0, 0.5, 2.0])


def test_assert_nested_rejects_grid_with_ties() -> None:
    constructor = _GrowingSets()
    with pytest.raises(ValueError, match="strictly increasing"):
        constructor.assert_nested(None, [0.0, 0.0, 1.0])
