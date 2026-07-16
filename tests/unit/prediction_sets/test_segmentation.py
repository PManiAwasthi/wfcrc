"""Unit tests for :class:`wfcrc.prediction_sets.segmentation.MorphologicalSets`.

Covers the standard iterated-dilation construction (radius mapping, both
structuring-element shapes, dimension-agnosticism, no-wraparound boundary
behavior), the documented erosion-direction gap, and nestedness across a
`lambda`-grid including fractional values (exercising `floor(lambda)`).
"""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.exceptions import SetConstructionError
from wfcrc.prediction_sets.segmentation import MorphologicalSets


def _point_mask(shape: tuple[int, ...], index: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[index] = True
    return mask


def test_construct_radius_zero_is_the_identity() -> None:
    mask = _point_mask((7, 7), (3, 3))
    sets = MorphologicalSets()
    for lam in (0.0, 0.99):
        np.testing.assert_array_equal(sets.construct(mask, lam), mask)


def test_construct_cross_radius_one_adds_four_neighbors() -> None:
    mask = _point_mask((7, 7), (3, 3))
    sets = MorphologicalSets(element="cross")
    grown = sets.construct(mask, 1.0)
    assert grown.sum() == 5  # center + 4 axis-aligned neighbors
    assert grown[3, 3] and grown[2, 3] and grown[4, 3] and grown[3, 2] and grown[3, 4]
    assert not grown[2, 2]  # diagonal neighbor excluded by "cross"


def test_construct_square_radius_one_adds_all_eight_neighbors() -> None:
    mask = _point_mask((7, 7), (3, 3))
    sets = MorphologicalSets(element="square")
    grown = sets.construct(mask, 1.0)
    assert grown.sum() == 9  # full 3x3 neighborhood
    assert grown[2, 2] and grown[4, 4]  # diagonals included by "square"


def test_construct_radius_is_floor_of_lambda() -> None:
    mask = _point_mask((9, 9), (4, 4))
    sets = MorphologicalSets(element="cross")
    # lambda in [1, 2) all map to radius 1.
    grown_at_1 = sets.construct(mask, 1.0)
    grown_at_1_99 = sets.construct(mask, 1.99)
    np.testing.assert_array_equal(grown_at_1, grown_at_1_99)
    grown_at_2 = sets.construct(mask, 2.0)
    assert grown_at_2.sum() > grown_at_1.sum()


def test_construct_iterated_growth_is_monotone_in_radius() -> None:
    mask = _point_mask((11, 11), (5, 5))
    sets = MorphologicalSets(element="square")
    prev = sets.construct(mask, 0.0)
    for lam in (1.0, 2.0, 3.0):
        curr = sets.construct(mask, lam)
        assert np.all(prev <= curr)
        prev = curr


def test_construct_does_not_wrap_around_array_boundary() -> None:
    mask = _point_mask((5, 5), (0, 0))
    sets = MorphologicalSets(element="square")
    grown = sets.construct(mask, 1.0)
    assert not grown[4, 4]  # would be "wrapped" neighbor under np.roll semantics
    assert grown[0, 1] and grown[1, 0] and grown[1, 1]


def test_construct_is_dimension_agnostic_1d() -> None:
    mask = _point_mask((9,), (4,))
    sets = MorphologicalSets(element="cross")
    grown = sets.construct(mask, 1.0)
    assert grown.sum() == 3
    assert grown[3] and grown[4] and grown[5]


def test_construct_is_dimension_agnostic_3d() -> None:
    mask = _point_mask((5, 5, 5), (2, 2, 2))
    sets = MorphologicalSets(element="cross")
    grown = sets.construct(mask, 1.0)
    assert grown.sum() == 7  # center + 6 face neighbors in 3-D


def test_construct_rejects_non_bool_score() -> None:
    sets = MorphologicalSets()
    with pytest.raises(ValueError, match="dtype bool"):
        sets.construct(np.zeros((3, 3), dtype=np.float64), 1.0)


def test_construct_rejects_negative_lambda() -> None:
    sets = MorphologicalSets()
    mask = _point_mask((3, 3), (1, 1))
    with pytest.raises(ValueError, match="lam must be >= 0"):
        sets.construct(mask, -0.5)


def test_init_rejects_unsupported_element() -> None:
    with pytest.raises(ValueError, match="element must be"):
        MorphologicalSets(element="diamond")  # type: ignore[arg-type]


def test_init_rejects_erosion_direction_as_a_documented_gap() -> None:
    with pytest.raises(SetConstructionError, match="not implemented"):
        MorphologicalSets(direction="erosion")


def test_name_is_morphological() -> None:
    assert MorphologicalSets().name() == "morphological"


def test_assert_nested_holds_across_an_integer_grid() -> None:
    mask = _point_mask((15, 15), (7, 7))
    sets = MorphologicalSets(element="square")
    assert sets.assert_nested(mask, [0.0, 1.0, 2.0, 3.0]) is True


def test_assert_nested_holds_across_a_fractional_grid() -> None:
    mask = _point_mask((15, 15), (7, 7))
    sets = MorphologicalSets(element="cross")
    assert sets.assert_nested(mask, [0.0, 0.5, 1.5, 2.5]) is True
