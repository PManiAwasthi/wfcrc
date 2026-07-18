"""Unit tests for :mod:`wfcrc.datasets.registry` (MS6.1)."""

from __future__ import annotations

from wfcrc.datasets import DATASETS as DATASETS_FROM_PACKAGE
from wfcrc.datasets.base import DatasetLoader
from wfcrc.datasets.registry import DATASETS


def test_datasets_is_a_dict() -> None:
    assert isinstance(DATASETS, dict)


def test_datasets_starts_empty() -> None:
    # MS6.1 defines only the registry itself; MS6.3 populates concrete
    # loaders, one Phase-A dataset at a time, starting with MSD
    # Task04_Hippocampus.
    assert DATASETS == {}


def test_every_registered_entry_is_a_dataset_loader_subclass() -> None:
    # Vacuously true while DATASETS is empty; guards every future MS6.3
    # registration against accidentally registering a non-DatasetLoader.
    for name, cls in DATASETS.items():
        assert issubclass(
            cls, DatasetLoader
        ), f"DATASETS[{name!r}] = {cls!r} is not a DatasetLoader subclass"


def test_datasets_reexported_from_package_init() -> None:
    assert DATASETS_FROM_PACKAGE is DATASETS
