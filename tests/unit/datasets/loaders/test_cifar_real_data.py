"""Opt-in, marker-gated real-data validation for CIFAR-10 / CIFAR-10.1 (DI-2).

Excluded from the default suite (``-m 'not real_data'``). Points at this
environment's actual archive locations; skips if absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfcrc.datasets.loaders.cifar import CifarLoader

pytestmark = pytest.mark.real_data

CIFAR10_ROOT = Path("datasets/CIFAR10")
CIFAR10_1_ROOT = Path("datasets/CIFAR10_1")


def test_cifar10_real_pool_counts_and_reads() -> None:
    if not (CIFAR10_ROOT / "test_batch").is_file():
        pytest.skip(f"CIFAR-10 not found at {CIFAR10_ROOT.resolve()}")
    loader = CifarLoader(CIFAR10_ROOT, split_manifest={"train": [], "calibration": [], "test": []})
    ids = list(loader._ids)
    assert len(ids) == 60000
    assert ids[0] == "train_00000" and ids[-1] == "test_09999"
    # Read a handful through a real split.
    sample = ["train_00000", "train_00001", "test_00000"]
    ds = CifarLoader(
        CIFAR10_ROOT, split_manifest={"train": sample, "calibration": [], "test": []}
    ).load("train")
    for cid in ["train_00000", "train_00001"]:
        image = ds.image(cid)
        label = ds.labels(cid)
        assert image.shape == (32, 32, 3) and image.dtype == np.uint8
        assert label.shape == (10,) and label.sum() == 1
        assert 0 <= ds.class_index(cid) < 10
    assert ds.verify_integrity().ok


def test_cifar10_1_v6_real_pool_and_test_only() -> None:
    if not (CIFAR10_1_ROOT / "datasets" / "cifar10.1_v6_data.npy").is_file():
        pytest.skip(f"CIFAR-10.1 not found at {CIFAR10_1_ROOT.resolve()}")
    loader = CifarLoader(
        CIFAR10_1_ROOT,
        split_manifest={"train": [], "calibration": [], "test": []},
        variant="cifar10_1",
    )
    ids = list(loader._ids)
    assert len(ids) == 2000  # v6, verified DI-2
    # 100%-test policy (DATASET_SPLIT_POLICY §3.6) expressed via the manifest.
    full = CifarLoader(
        CIFAR10_1_ROOT,
        split_manifest={"train": [], "calibration": [], "test": ids},
        variant="cifar10_1",
    )
    test = full.load("test")
    assert len(test) == 2000
    assert len(full.load("train")) == 0
    cid = test.ids()[0]
    assert test.image(cid).shape == (32, 32, 3)
    assert test.labels(cid).sum() == 1


def test_cifar10_1_v4_real_count() -> None:
    if not (CIFAR10_1_ROOT / "datasets" / "cifar10.1_v4_data.npy").is_file():
        pytest.skip("CIFAR-10.1 v4 not present")
    loader = CifarLoader(
        CIFAR10_1_ROOT,
        split_manifest={"train": [], "calibration": [], "test": []},
        variant="cifar10_1",
        cifar10_1_version="v4",
    )
    assert len(list(loader._ids)) == 2021
