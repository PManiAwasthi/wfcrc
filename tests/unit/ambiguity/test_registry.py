"""Unit tests for the :data:`wfcrc.ambiguity.FAMILIES` registry."""

from __future__ import annotations

from wfcrc.ambiguity import (
    FAMILIES,
    AmbiguityFamily,
    CVaRFamily,
    FiniteGroupFamily,
    KLFamily,
    KnownWeightFamily,
)


def test_registry_contains_exactly_the_four_frozen_families() -> None:
    assert set(FAMILIES) == {"cvar", "kl", "finite_group", "known_weight"}


def test_registry_maps_to_correct_classes() -> None:
    assert FAMILIES["cvar"] is CVaRFamily
    assert FAMILIES["kl"] is KLFamily
    assert FAMILIES["finite_group"] is FiniteGroupFamily
    assert FAMILIES["known_weight"] is KnownWeightFamily


def test_registry_values_are_ambiguity_family_subclasses() -> None:
    for cls in FAMILIES.values():
        assert issubclass(cls, AmbiguityFamily)


def test_registry_instances_report_matching_family_type() -> None:
    assert FAMILIES["cvar"](beta=0.1).family_type == "cvar"
    assert FAMILIES["kl"](rho=0.1).family_type == "kl"
    assert FAMILIES["finite_group"](masks=[[0]]).family_type == "finite_group"
    assert FAMILIES["known_weight"](weights=[1.0]).family_type == "known_weight"
