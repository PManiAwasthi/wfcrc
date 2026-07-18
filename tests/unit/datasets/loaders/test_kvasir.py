"""Unit tests for :mod:`wfcrc.datasets.loaders.kvasir` (DI-2).

Synthetic fixture: ``<root>/images/<stem>.jpg`` + ``<root>/masks/<stem>.jpg``
pairs written with Pillow — never real data. See ``test_kvasir_real_data.py``
for the opt-in real-data validation. Mirrors the DI-1 test checklist
(`docs/DATASET_INTEGRATION_GUIDE.md` §8).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wfcrc.datasets.loaders.kvasir import KvasirDataset, KvasirLoader
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.datasets.preprocessing import resize_and_normalize
from wfcrc.exceptions import SerializationError, SplitLeakageError

STEMS = ("aaa111", "bbb222", "ccc333", "ddd444")


def _write_image(path: Path, h: int = 16, w: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(abs(hash(path.name)) % (2**32)).integers(
        0, 256, size=(h, w, 3), dtype=np.uint8
    )
    Image.fromarray(arr, mode="RGB").save(path, quality=100)


def _write_mask(path: Path, h: int = 16, w: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # left half background (0), right half polyp (255) — robust to JPEG on
    # 8-aligned boundaries.
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:, w // 2 :] = 255
    Image.fromarray(arr, mode="L").save(path, quality=100)


def _build_kvasir(
    root: Path,
    stems: Sequence[str] = STEMS,
    *,
    skip_mask_for: str | None = None,
    corrupt_image_for: str | None = None,
    mask_size: tuple[int, int] | None = None,
) -> Path:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img_path = images / f"{stem}.jpg"
        if stem == corrupt_image_for:
            img_path.write_bytes(b"not a real jpg")
        else:
            _write_image(img_path)
        if stem != skip_mask_for:
            if mask_size is not None and stem == stems[0]:
                _write_mask(masks / f"{stem}.jpg", h=mask_size[0], w=mask_size[1])
            else:
                _write_mask(masks / f"{stem}.jpg")
    return root


def _manifest(train: list[str], cal: list[str], test: list[str]) -> dict[str, list[str]]:
    return {"train": train, "calibration": cal, "test": test}


# --- normal loading ----------------------------------------------------------


def test_load_and_iterate(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    loader = KvasirLoader(
        tmp_path, split_manifest=_manifest(list(STEMS[:2]), list(STEMS[2:3]), list(STEMS[3:]))
    )
    train = loader.load("train")
    assert len(train) == 2
    assert list(train.ids()) == list(STEMS[:2])
    triples = list(train)
    assert len(triples) == 2
    for _id, image, label in triples:
        assert image.ndim == 3 and image.shape[2] == 3
        assert label.dtype == np.bool_
        assert label.shape == image.shape[:2]


def test_discovery_pairs_all(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    loader = KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
    assert set(loader.load("train").ids()) == set(STEMS)


def test_split_manifest_from_json_path(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(_manifest(list(STEMS), [], [])), encoding="utf-8")
    loader = KvasirLoader(tmp_path, split_manifest=path)
    assert len(loader.load("train")) == len(STEMS)


def test_empty_split(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    loader = KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
    assert len(loader.load("test")) == 0
    assert list(loader.load("test")) == []


def test_deterministic_and_reproducible(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    m = _manifest(list(STEMS), [], [])
    a = KvasirLoader(tmp_path, split_manifest=m).load("train")
    b = KvasirLoader(tmp_path, split_manifest=m).load("train")
    assert list(a.ids()) == list(b.ids())
    assert [i for i, _, _ in a] == [i for i, _, _ in a]


# --- per-id accessors --------------------------------------------------------


def test_image_mask_and_labels(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    loader = KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
    ds = loader.load("train")
    cid = ds.ids()[0]
    assert ds.image(cid).shape == (16, 16, 3)
    raw = ds.raw_mask(cid)
    assert raw.dtype == np.uint8 and raw.shape == (16, 16)
    label = ds.labels(cid)
    assert label.dtype == np.bool_
    # binarization: some polyp, some background
    assert label.any() and (~label).any()
    np.testing.assert_array_equal(label, raw > 127)


def test_resolution_and_unknown_id(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    ds = KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], [])).load("train")
    assert ds.resolution(ds.ids()[0]) == (16, 16)
    with pytest.raises(ValueError, match="unknown id"):
        ds.labels("nope")


def test_meta_records_unresolved_split_unit(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    meta = (
        KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], [])).load("train").meta()
    )
    assert meta["name"] == "kvasir_seg"
    assert meta["version"] == DATASET_METADATA["kvasir_seg"].version
    assert meta["label_kind"] == "binary_polyp_mask"
    assert meta["mask_threshold"] == 127
    assert "UNRESOLVED" in meta["split_unit_status"]


# --- construction / config validation ---------------------------------------


def test_missing_images_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "masks").mkdir()
    with pytest.raises(ValueError, match="images directory not found"):
        KvasirLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_missing_masks_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="masks directory not found"):
        KvasirLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_no_images_raises(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    with pytest.raises(SerializationError, match="no Kvasir-SEG images"):
        KvasirLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_missing_mask_raises_at_discovery(tmp_path: Path) -> None:
    _build_kvasir(tmp_path, skip_mask_for="ccc333")
    with pytest.raises(SerializationError, match="no paired mask"):
        KvasirLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_unknown_split_name_raises(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    loader = KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
    with pytest.raises(ValueError, match="split_name must be one of"):
        loader.load("holdout")


def test_manifest_id_outside_pool_raises(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    with pytest.raises(ValueError, match="not present in the discovered"):
        KvasirLoader(tmp_path, split_manifest=_manifest(["ghost"], [], []))


def test_overlapping_manifest_raises(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    with pytest.raises(SplitLeakageError):
        KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS[:1]), [], list(STEMS[:1])))


def test_duplicate_id_within_split_raises() -> None:
    from wfcrc.datasets.loaders.kvasir import _KvasirCase

    case = _KvasirCase(id_="x", image_path=Path("i"), mask_path=Path("m"))
    with pytest.raises(ValueError, match="duplicate id"):
        KvasirDataset([case, case])


# --- integrity ---------------------------------------------------------------


def test_verify_integrity_clean(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    report = (
        KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
        .load("train")
        .verify_integrity()
    )
    assert report.ok


def test_verify_integrity_shape_mismatch(tmp_path: Path) -> None:
    _build_kvasir(tmp_path, mask_size=(8, 8))  # first stem's mask is 8x8, image 16x16
    report = (
        KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
        .load("train")
        .verify_integrity()
    )
    assert not report.ok
    assert any("shape mismatch" in i.problem for i in report.issues)


def test_verify_integrity_unreadable_image(tmp_path: Path) -> None:
    _build_kvasir(tmp_path, corrupt_image_for="aaa111")
    report = (
        KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
        .load("train")
        .verify_integrity()
    )
    assert not report.ok
    assert any("image unreadable" in i.problem for i in report.issues)


def test_verify_integrity_unreadable_mask(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    (tmp_path / "masks" / "aaa111.jpg").write_bytes(b"corrupt mask")
    report = (
        KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], []))
        .load("train")
        .verify_integrity()
    )
    assert not report.ok
    assert any("mask unreadable" in i.problem for i in report.issues)


# --- preprocessing compatibility ---------------------------------------------


def test_image_feeds_frozen_resize_and_normalize(tmp_path: Path) -> None:
    _build_kvasir(tmp_path)
    ds = KvasirLoader(tmp_path, split_manifest=_manifest(list(STEMS), [], [])).load("train")
    out = resize_and_normalize(
        ds.image(ds.ids()[0]), target_size=(8, 8), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
    )
    assert out.shape == (8, 8, 3)
