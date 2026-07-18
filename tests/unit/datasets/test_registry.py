"""Unit tests for :mod:`wfcrc.datasets.registry` (MS6.1, populated MS6.3A / DI-2)."""

from __future__ import annotations

from wfcrc.datasets import DATASETS as DATASETS_FROM_PACKAGE
from wfcrc.datasets.base import DatasetLoader
from wfcrc.datasets.loaders.acdc import ACDCLoader
from wfcrc.datasets.loaders.cifar import CifarLoader
from wfcrc.datasets.loaders.kvasir import KvasirLoader
from wfcrc.datasets.loaders.msd import MSDNiftiLoader
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.datasets.registry import DATASETS


def test_datasets_is_a_dict() -> None:
    assert isinstance(DATASETS, dict)


def test_datasets_contains_exactly_the_di2_entries() -> None:
    # DI-2 registers every locally-available Phase-A loader family. MSD tasks
    # share MSDNiftiLoader; CIFAR variants share CifarLoader. Cityscapes /
    # cityscapes_c remain unregistered (absent, out of DI-2 scope) even
    # though they have DATASET_METADATA entries.
    assert {
        "msd_hippocampus": MSDNiftiLoader,
        "msd_pancreas": MSDNiftiLoader,
        "acdc": ACDCLoader,
        "kvasir_seg": KvasirLoader,
        "cifar10": CifarLoader,
        "cifar10_1": CifarLoader,
    } == DATASETS


def test_cityscapes_keys_have_metadata_but_no_loader_yet() -> None:
    # Documents the deliberate DI-2 boundary: metadata exists, loader does not.
    for name in ("cityscapes", "cityscapes_c"):
        assert name in DATASET_METADATA
        assert name not in DATASETS


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
