"""Unit tests for :class:`wfcrc.ambiguity.finite_group.FiniteGroupFamily`."""

from __future__ import annotations

import pytest

from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.exceptions import FamilyError


def test_family_type() -> None:
    family = FiniteGroupFamily(masks=[[0, 1], [2, 3]])
    assert family.family_type == "finite_group"


def test_groups_returns_index_tuples() -> None:
    family = FiniteGroupFamily(masks=[[0, 1, 2], [3, 4]])
    assert family.groups() == ((0, 1, 2), (3, 4))


def test_rejects_empty_family() -> None:
    with pytest.raises(FamilyError):
        FiniteGroupFamily(masks=[])


def test_rejects_empty_group() -> None:
    with pytest.raises(FamilyError):
        FiniteGroupFamily(masks=[[0, 1], []])


def test_rejects_negative_index() -> None:
    with pytest.raises(FamilyError):
        FiniteGroupFamily(masks=[[0, -1, 2]])


def test_single_group() -> None:
    family = FiniteGroupFamily(masks=[[5, 6, 7]])
    assert family.groups() == ((5, 6, 7),)


def test_overlapping_groups_allowed() -> None:
    # No exclusivity requirement is stated anywhere in the frozen spec.
    family = FiniteGroupFamily(masks=[[0, 1, 2], [1, 2, 3]])
    assert family.groups() == ((0, 1, 2), (1, 2, 3))


def test_singleton_group() -> None:
    family = FiniteGroupFamily(masks=[[0]])
    assert family.groups() == ((0,),)


def test_masks_attribute_matches_groups() -> None:
    family = FiniteGroupFamily(masks=[[0, 1], [2]])
    assert family.masks == family.groups()


def test_deterministic_construction() -> None:
    masks = [[0, 1], [2, 3]]
    a = FiniteGroupFamily(masks=masks).groups()
    b = FiniteGroupFamily(masks=masks).groups()
    assert a == b
