"""Unit tests for the :data:`wfcrc.losses.LOSSES` registry."""

from __future__ import annotations

from wfcrc.losses import LOSSES, FNRLoss, FPRLoss, LossEvaluator, MiscoverageLoss


def test_registry_contains_all_three_frozen_families() -> None:
    assert set(LOSSES) == {"fnr", "fpr", "miscoverage"}


def test_registry_maps_to_correct_classes() -> None:
    assert LOSSES["fnr"] is FNRLoss
    assert LOSSES["fpr"] is FPRLoss
    assert LOSSES["miscoverage"] is MiscoverageLoss


def test_registry_values_are_loss_evaluator_subclasses() -> None:
    for cls in LOSSES.values():
        assert issubclass(cls, LossEvaluator)


def test_registry_instantiates_and_reports_matching_name() -> None:
    for key, cls in LOSSES.items():
        instance = cls()
        assert instance.name() == key
