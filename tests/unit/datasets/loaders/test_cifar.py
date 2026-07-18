"""Unit tests for :mod:`wfcrc.datasets.loaders.cifar` (DI-2).

Synthetic fixtures: tiny CIFAR-10 pickle batches and CIFAR-10.1 ``.npy``
arrays written to a temp dir — never real data. See
``test_cifar_real_data.py`` for the opt-in real-data validation. Mirrors the
DI-1 test checklist (`docs/DATASET_INTEGRATION_GUIDE.md` §8).
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from wfcrc.datasets.loaders.cifar import CifarDataset, CifarLoader
from wfcrc.datasets.preprocessing import resize_and_normalize
from wfcrc.exceptions import SerializationError, SplitLeakageError

ROWS_PER_TRAIN_BATCH = 2  # 5 batches -> 10 train rows
TEST_ROWS = 3


def _batch_bytes(n: int, *, label_start: int = 0, r: int | None = None) -> bytes:
    data = np.zeros((n, 3072), dtype=np.uint8)
    if r is not None:
        data[:, 0:1024] = r
        data[:, 1024:2048] = r + 10
        data[:, 2048:3072] = r + 20
    else:
        data[:] = np.arange(n * 3072, dtype=np.uint8).reshape(n, 3072)
    labels = [(label_start + i) % 10 for i in range(n)]
    return pickle.dumps(
        {"data": data, "labels": labels, "filenames": [f"f{i}.png" for i in range(n)]}
    )


def _build_cifar10(root: Path, *, r_for_first: int | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for b in range(1, 6):
        payload = _batch_bytes(
            ROWS_PER_TRAIN_BATCH, label_start=b, r=r_for_first if b == 1 else None
        )
        (root / f"data_batch_{b}").write_bytes(payload)
    (root / "test_batch").write_bytes(_batch_bytes(TEST_ROWS, label_start=7))
    return root


def _build_cifar10_1(root: Path, *, versions: Sequence[str] = ("v6",), n: int = 5) -> Path:
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    for v in versions:
        data = np.arange(n * 32 * 32 * 3, dtype=np.uint8).reshape(n, 32, 32, 3)
        labels = np.array([i % 10 for i in range(n)], dtype=np.int64)
        np.save(root / "datasets" / f"cifar10.1_{v}_data.npy", data)
        np.save(root / "datasets" / f"cifar10.1_{v}_labels.npy", labels)
    return root


def _manifest(train: list[str], cal: list[str], test: list[str]) -> dict[str, list[str]]:
    return {"train": train, "calibration": cal, "test": test}


# --- CIFAR-10 ----------------------------------------------------------------


def test_cifar10_pool_ids_and_counts(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    loader = CifarLoader(tmp_path, split_manifest=_manifest([], [], []))
    ids = list(loader._ids)
    assert ids[:10] == [f"train_{i:05d}" for i in range(10)]
    assert ids[10:] == [f"test_{i:05d}" for i in range(TEST_ROWS)]


def test_cifar10_load_iterate_and_labels(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    m = _manifest(["train_00000", "train_00001"], ["test_00000"], ["test_00001"])
    loader = CifarLoader(tmp_path, split_manifest=m)
    train = loader.load("train")
    assert len(train) == 2
    triples = list(train)
    assert [i for i, _, _ in triples] == ["train_00000", "train_00001"]
    for _id, image, label in triples:
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert label.shape == (10,)
        assert label.dtype == np.bool_
        assert label.sum() == 1


def test_cifar10_channel_layout_reshape(tmp_path: Path) -> None:
    # data row [R=5 | G=15 | B=25] must reconstruct to HWC with those channels.
    _build_cifar10(tmp_path, r_for_first=5)
    loader = CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], [], []))
    image = loader.load("train").image("train_00000")
    assert image[..., 0].min() == 5 and image[..., 0].max() == 5
    assert image[..., 1].min() == 15 and image[..., 1].max() == 15
    assert image[..., 2].min() == 25 and image[..., 2].max() == 25


def test_cifar10_class_index_name_and_onehot_consistency(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    ds = CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], [], [])).load("train")
    cid = "train_00000"
    idx = ds.class_index(cid)
    assert 0 <= idx < 10
    assert ds.class_name(cid) in {
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    }
    assert int(np.argmax(ds.labels(cid))) == idx


def test_cifar10_meta(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    meta = (
        CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], [], []))
        .load("train")
        .meta()
    )
    assert meta["name"] == "cifar10"
    assert meta["num_classes"] == 10
    assert len(meta["class_names"]) == 10


# --- CIFAR-10.1 --------------------------------------------------------------


def test_cifar10_1_v6_pool_and_test_only_split(tmp_path: Path) -> None:
    _build_cifar10_1(tmp_path, n=5)
    loader = CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1")
    ids = list(loader._ids)
    assert ids == [f"v6_{i:04d}" for i in range(5)]
    # policy §3.6: 100% test
    full = CifarLoader(tmp_path, split_manifest=_manifest([], [], ids), variant="cifar10_1")
    assert len(full.load("test")) == 5
    assert len(full.load("train")) == 0


def test_cifar10_1_v4_version_selectable(tmp_path: Path) -> None:
    _build_cifar10_1(tmp_path, versions=("v4", "v6"), n=4)
    loader = CifarLoader(
        tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1", cifar10_1_version="v4"
    )
    assert list(loader._ids) == [f"v4_{i:04d}" for i in range(4)]


def test_cifar10_1_meta(tmp_path: Path) -> None:
    _build_cifar10_1(tmp_path)
    ds = CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1").load(
        "train"
    )
    assert ds.meta()["name"] == "cifar10_1"


# --- shared behaviour --------------------------------------------------------


def test_split_manifest_json_path(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(_manifest(["train_00000"], [], [])), encoding="utf-8")
    loader = CifarLoader(tmp_path, split_manifest=path)
    assert list(loader.load("train").ids()) == ["train_00000"]


def test_empty_split(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    loader = CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], [], []))
    assert len(loader.load("test")) == 0
    assert list(loader.load("test")) == []


def test_reproducible(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    m = _manifest(["train_00000", "train_00001"], [], [])
    a = CifarLoader(tmp_path, split_manifest=m).load("train").ids()
    b = CifarLoader(tmp_path, split_manifest=m).load("train").ids()
    assert list(a) == list(b)


def test_unknown_id_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    ds = CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], [], [])).load("train")
    with pytest.raises(ValueError, match="unknown id"):
        ds.image("train_99999")


# --- construction / config validation ---------------------------------------


def test_unsupported_variant_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    with pytest.raises(ValueError, match="unsupported variant"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar100")


def test_unsupported_version_raises(tmp_path: Path) -> None:
    _build_cifar10_1(tmp_path)
    with pytest.raises(ValueError, match="unsupported cifar10_1_version"):
        CifarLoader(
            tmp_path,
            split_manifest=_manifest([], [], []),
            variant="cifar10_1",
            cifar10_1_version="v9",
        )


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="root directory not found"):
        CifarLoader(tmp_path / "nope", split_manifest=_manifest([], [], []))


def test_missing_batch_file_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    (tmp_path / "data_batch_3").unlink()
    with pytest.raises(SerializationError, match="batch file not found"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_malformed_batch_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    (tmp_path / "data_batch_1").write_bytes(pickle.dumps({"data": np.zeros((2, 3072), np.uint8)}))
    with pytest.raises(SerializationError, match="malformed CIFAR-10 batch"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_unpicklable_batch_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    (tmp_path / "data_batch_1").write_bytes(b"not a pickle")
    with pytest.raises(SerializationError, match="could not read/unpickle"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_wrong_data_width_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    (tmp_path / "data_batch_1").write_bytes(
        pickle.dumps({"data": np.zeros((2, 100), np.uint8), "labels": [0, 1]})
    )
    with pytest.raises(SerializationError, match=r"must be \(n, 3072\)"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_data_label_length_mismatch_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    (tmp_path / "data_batch_1").write_bytes(
        pickle.dumps({"data": np.zeros((2, 3072), np.uint8), "labels": [0]})
    )
    with pytest.raises(SerializationError, match="data/label length mismatch"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_missing_npy_raises(tmp_path: Path) -> None:
    (tmp_path / "datasets").mkdir()
    with pytest.raises(SerializationError, match=r"CIFAR-10\.1 file not found"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1")


def test_cifar10_1_label_length_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "datasets").mkdir()
    np.save(tmp_path / "datasets" / "cifar10.1_v6_data.npy", np.zeros((5, 32, 32, 3), np.uint8))
    np.save(tmp_path / "datasets" / "cifar10.1_v6_labels.npy", np.zeros((3,), np.int64))
    with pytest.raises(SerializationError, match="data/label length mismatch"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1")


def test_unknown_split_name_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    loader = CifarLoader(tmp_path, split_manifest=_manifest([], [], []))
    with pytest.raises(ValueError, match="split_name must be one of"):
        loader.load("valid")


def test_manifest_id_outside_pool_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    with pytest.raises(ValueError, match="not present in the discovered"):
        CifarLoader(tmp_path, split_manifest=_manifest(["train_88888"], [], []))


def test_overlapping_manifest_raises(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    with pytest.raises(SplitLeakageError):
        CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], ["train_00000"], []))


# --- CifarDataset direct construction guards ---------------------------------


def test_duplicate_id_within_split_raises() -> None:
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    labels = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate id"):
        CifarDataset(["x", "x"], images, labels, metadata_key="cifar10")


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        CifarDataset(
            ["a", "b"],
            np.zeros((1, 32, 32, 3), np.uint8),
            np.zeros((1,), np.int64),
            metadata_key="cifar10",
        )


def test_verify_integrity_clean(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    ds = CifarLoader(
        tmp_path, split_manifest=_manifest(["train_00000", "train_00001"], [], [])
    ).load("train")
    assert ds.verify_integrity().ok


def test_verify_integrity_bad_class_index() -> None:
    ds = CifarDataset(
        ["a"], np.zeros((1, 32, 32, 3), np.uint8), np.array([99], np.int64), metadata_key="cifar10"
    )
    report = ds.verify_integrity()
    assert not report.ok
    assert any("class index" in i.problem for i in report.issues)


def test_verify_integrity_bad_image_shape() -> None:
    ds = CifarDataset(
        ["a"], np.zeros((1, 30, 30, 3), np.uint8), np.array([0], np.int64), metadata_key="cifar10"
    )
    report = ds.verify_integrity()
    assert not report.ok
    assert any("image must be" in i.problem for i in report.issues)


def test_image_feeds_frozen_resize_and_normalize(tmp_path: Path) -> None:
    _build_cifar10(tmp_path)
    ds = CifarLoader(tmp_path, split_manifest=_manifest(["train_00000"], [], [])).load("train")
    out = resize_and_normalize(
        ds.image("train_00000"), target_size=(16, 16), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
    )
    assert out.shape == (16, 16, 3)


def test_cifar10_1_wrong_image_shape_raises(tmp_path: Path) -> None:
    (tmp_path / "datasets").mkdir()
    np.save(tmp_path / "datasets" / "cifar10.1_v6_data.npy", np.zeros((3, 16, 16, 3), np.uint8))
    np.save(tmp_path / "datasets" / "cifar10.1_v6_labels.npy", np.zeros((3,), np.int64))
    with pytest.raises(SerializationError, match=r"expected CIFAR images of shape"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1")


def test_cifar10_1_corrupt_npy_raises(tmp_path: Path) -> None:
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "cifar10.1_v6_data.npy").write_bytes(b"not a real npy file")
    (tmp_path / "datasets" / "cifar10.1_v6_labels.npy").write_bytes(b"also not npy")
    with pytest.raises(SerializationError, match=r"could not load CIFAR-10\.1"):
        CifarLoader(tmp_path, split_manifest=_manifest([], [], []), variant="cifar10_1")
