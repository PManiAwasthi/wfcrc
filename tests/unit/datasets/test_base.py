"""Unit tests for :mod:`wfcrc.datasets.base`."""

from __future__ import annotations

import pytest

from tests.unit.datasets._helpers import FakeDataset
from wfcrc.datasets.base import Dataset, DatasetLoader, SplitManifest, assert_split_disjoint
from wfcrc.exceptions import SplitLeakageError


class _FixedLoader(DatasetLoader):
    """Minimal concrete `DatasetLoader` isolating the shared contract from any real dataset."""

    def load(self, split_name: str) -> Dataset:
        if split_name not in ("train", "calibration", "test"):
            raise ValueError(f"unknown split: {split_name}")
        return FakeDataset(3)


def test_fake_dataset_satisfies_the_dataset_contract() -> None:
    ds = FakeDataset(5)
    assert isinstance(ds, Dataset)
    assert len(ds) == 5
    assert list(ds.ids()) == [0, 1, 2, 3, 4]
    assert ds.labels(0).tolist() == [True, False, False]
    assert ds.meta() == {"version": "v1", "license": "test-license"}


def test_dataset_iter_yields_id_x_y_triples() -> None:
    ds = FakeDataset(3)
    items = list(ds)
    assert [item[0] for item in items] == [0, 1, 2]
    assert items[0][2].tolist() == [True, False, False]
    assert items[1][2].tolist() == [False, True, False]


def test_dataset_loader_loads_a_known_split() -> None:
    loader = _FixedLoader()
    ds = loader.load("calibration")
    assert len(ds) == 3


def test_dataset_loader_rejects_an_unknown_split() -> None:
    loader = _FixedLoader()
    with pytest.raises(ValueError, match="unknown split"):
        loader.load("nonsense")


def test_assert_split_disjoint_passes_for_genuinely_disjoint_ids() -> None:
    assert_split_disjoint([1, 2], [3, 4], [5, 6])  # must not raise


def test_assert_split_disjoint_raises_on_train_calibration_overlap() -> None:
    with pytest.raises(SplitLeakageError, match="train/calibration"):
        assert_split_disjoint([1, 2], [2, 3], [4])


def test_assert_split_disjoint_raises_on_train_test_overlap() -> None:
    with pytest.raises(SplitLeakageError, match="train/test"):
        assert_split_disjoint([1, 2], [3], [1])


def test_assert_split_disjoint_raises_on_calibration_test_overlap() -> None:
    with pytest.raises(SplitLeakageError, match="calibration/test"):
        assert_split_disjoint([1], [2, 3], [3])


def test_assert_split_disjoint_reports_every_overlapping_pair() -> None:
    with pytest.raises(SplitLeakageError, match=r"train/calibration.*train/test"):
        assert_split_disjoint([1], [1], [1])


def test_split_manifest_coerces_id_sequences_to_tuples() -> None:
    manifest = SplitManifest(train_ids=[1, 2], cal_ids=[3], test_ids=[4, 5])
    assert manifest.train_ids == (1, 2)
    assert manifest.cal_ids == (3,)
    assert manifest.test_ids == (4, 5)


def test_split_manifest_raises_on_overlap() -> None:
    with pytest.raises(SplitLeakageError):
        SplitManifest(train_ids=[1, 2], cal_ids=[2], test_ids=[3])
