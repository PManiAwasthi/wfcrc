"""Unit tests for :mod:`wfcrc.datasets.registry` (MS6.1, populated from MS6.3A)."""

from __future__ import annotations

from wfcrc.datasets import DATASETS as DATASETS_FROM_PACKAGE
from wfcrc.datasets.base import DatasetLoader
from wfcrc.datasets.loaders.msd import MSDNiftiLoader
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.datasets.registry import DATASETS


def test_datasets_is_a_dict() -> None:
    assert isinstance(DATASETS, dict)


def test_datasets_contains_exactly_the_ms6_3a_entries() -> None:
    # MS6.3A implements and registers only the MSD/NIfTI family, starting
    # with Task04_Hippocampus. Cityscapes-format (+ACDC/-C), CIFAR, and
    # Kvasir are out of scope for this pass and must not appear yet.
    assert {"msd_hippocampus": MSDNiftiLoader} == DATASETS


def test_every_registered_entry_is_a_dataset_loader_subclass() -> None:
    for name, cls in DATASETS.items():
        assert issubclass(
            cls, DatasetLoader
        ), f"DATASETS[{name!r}] = {cls!r} is not a DatasetLoader subclass"


def test_registry_keys_are_consistent_with_dataset_metadata() -> None:
    # MS6 Architecture Specification §3.1: registry naming must stay
    # consistent with DATASET_METADATA's keys.
    for name in DATASETS:
        assert name in DATASET_METADATA


def test_datasets_reexported_from_package_init() -> None:
    assert DATASETS_FROM_PACKAGE is DATASETS
