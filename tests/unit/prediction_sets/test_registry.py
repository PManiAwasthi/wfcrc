"""Unit tests for the :data:`wfcrc.prediction_sets.SETS` registry."""

from __future__ import annotations

from wfcrc.prediction_sets import SETS, MorphologicalSets, PredictionSetConstructor, ThresholdSets


def test_registry_contains_exactly_the_two_frozen_constructors() -> None:
    assert set(SETS) == {"threshold", "morphological"}


def test_registry_maps_to_correct_classes() -> None:
    assert SETS["threshold"] is ThresholdSets
    assert SETS["morphological"] is MorphologicalSets


def test_registry_values_are_prediction_set_constructor_subclasses() -> None:
    for cls in SETS.values():
        assert issubclass(cls, PredictionSetConstructor)


def test_registry_instances_report_matching_names() -> None:
    assert SETS["threshold"]().name() == "threshold"
    assert SETS["morphological"]().name() == "morphological"
