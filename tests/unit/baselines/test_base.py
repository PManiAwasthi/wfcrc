"""Tests for `wfcrc.baselines.base.Calibrator`/`BASELINES`."""

from __future__ import annotations

import pytest

import wfcrc.baselines as baselines_pkg
from wfcrc.baselines.base import BASELINES, Calibrator


def test_calibrator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Calibrator()  # type: ignore[abstract]


def test_baselines_registry_populated_by_importing_the_package() -> None:
    expected = {
        "wfcrc",
        "vanilla_crc",
        "lac",
        "group_conditional",
        "robust_fdiv",
        "pooled_k_fold",
        "total_n_inflation",
        "fixed_eta",
        "temperature_scaled_lac",
        "ensemble_aggregated_lac",
    }
    assert expected <= set(BASELINES.keys())


def test_every_registered_class_is_a_calibrator_subclass() -> None:
    for name, cls in BASELINES.items():
        assert issubclass(cls, Calibrator), f"{name} -> {cls} is not a Calibrator subclass"


def test_package_all_matches_public_names() -> None:
    # every name in __all__ must actually be importable from the package
    for name in baselines_pkg.__all__:
        assert hasattr(baselines_pkg, name)
