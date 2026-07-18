"""Unit tests for :mod:`wfcrc.datasets.loaders.acdc` (DI-2).

All tests run against a tiny **synthetic** fixture reproducing the real ACDC
directory layout (``rgb_anon/<condition>/<split>/<sequence>/*_rgb_anon.png``
+ ``gt/<condition>/<split>/<sequence>/*_gt_labelTrainIds.png``) — never real
downloaded data. See ``test_acdc_real_data.py`` for the opt-in, marker-gated
real-data validation. Mirrors the DI-1 test checklist
(`docs/DATASET_INTEGRATION_GUIDE.md` §8).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wfcrc.datasets.loaders.acdc import ACDCDataset, ACDCLoader
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.datasets.preprocessing import resize_and_normalize
from wfcrc.exceptions import SerializationError, SplitLeakageError

# (condition, split, sequence, stem) for a small, deterministic fixture.
FRAMES: tuple[tuple[str, str, str, str], ...] = (
    ("fog", "train", "GP01", "GP01_frame_000001"),
    ("fog", "train", "GP01", "GP01_frame_000002"),
    ("night", "val", "GP02", "GP02_frame_000010"),
    ("rain", "train", "GP03", "GP03_frame_000020"),
)
LABEL_VALUES = (0, 1, 13, 255)  # a couple of trainIds + the ignore index


def _write_rgb(path: Path, h: int = 3, w: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_label(path: Path, h: int = 3, w: int = 4, values: Sequence[int] = LABEL_VALUES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((h, w), dtype=np.uint8)
    flat = arr.reshape(-1)
    for i, v in enumerate(values):
        flat[i % flat.size] = v
    Image.fromarray(arr, mode="L").save(path)


def _build_acdc(
    root: Path,
    frames: Sequence[tuple[str, str, str, str]] = FRAMES,
    *,
    skip_label_for: str | None = None,
    corrupt_image_for: str | None = None,
    label_h: int = 3,
    label_w: int = 4,
    label_values: Sequence[int] = LABEL_VALUES,
    add_ref_dir: bool = False,
    add_unlabelled_test: bool = False,
) -> Path:
    """Build a synthetic ACDC archive under ``root``; return ``root``."""
    rgb_root = root / "rgb_anon_trainvaltest" / "rgb_anon"
    gt_root = root / "gt_trainval" / "gt"
    for condition, split, seq, stem in frames:
        img_path = rgb_root / condition / split / seq / f"{stem}_rgb_anon.png"
        if stem == corrupt_image_for:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(b"not a real png")
        else:
            _write_rgb(img_path)
        if stem != skip_label_for:
            _write_label(
                gt_root / condition / split / seq / f"{stem}_gt_labelTrainIds.png",
                h=label_h,
                w=label_w,
                values=label_values,
            )
    if add_ref_dir:
        # A *_ref reference frame (no gt) must be ignored by discovery.
        _write_rgb(rgb_root / "fog" / "train_ref" / "GP01" / "GP01_frame_000001_rgb_anon.png")
    if add_unlabelled_test:
        # An official test frame (no gt) must be ignored by discovery.
        _write_rgb(rgb_root / "fog" / "test" / "GP09" / "GP09_frame_000099_rgb_anon.png")
    return root


def _manifest(train: list[str], cal: list[str], test: list[str]) -> dict[str, list[str]]:
    return {"train": train, "calibration": cal, "test": test}


ALL_IDS = [stem for *_, stem in FRAMES]


# --- normal loading ----------------------------------------------------------


def test_load_all_splits_and_iterate(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    manifest = _manifest(ALL_IDS[:2], ALL_IDS[2:3], ALL_IDS[3:])
    loader = ACDCLoader(tmp_path, split_manifest=manifest, foreground_class=13)
    train = loader.load("train")
    assert len(train) == 2
    assert list(train.ids()) == ALL_IDS[:2]
    triples = list(train)
    assert len(triples) == 2
    for (id_, image, label), expected_id in zip(triples, ALL_IDS[:2], strict=True):
        assert id_ == expected_id
        assert image.shape == (3, 4, 3)
        assert label.dtype == np.bool_
        assert label.shape == (3, 4)


def test_discovery_finds_all_labelled_frames(tmp_path: Path) -> None:
    _build_acdc(tmp_path, add_ref_dir=True, add_unlabelled_test=True)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    # ref frames and unlabelled test frames are ignored; only the 4 labelled.
    assert set(loader.load("train").ids()) == set(ALL_IDS)


def test_split_manifest_from_json_path(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(ALL_IDS, [], [])), encoding="utf-8")
    loader = ACDCLoader(tmp_path, split_manifest=manifest_path)
    assert len(loader.load("train")) == len(ALL_IDS)


def test_empty_split_has_zero_length(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    test = loader.load("test")
    assert len(test) == 0
    assert list(test) == []


def test_iteration_is_deterministic(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []), foreground_class=1)
    ds = loader.load("train")
    assert [i for i, _, _ in ds] == [i for i, _, _ in ds]


def test_reproducible_across_constructions(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    m = _manifest(ALL_IDS, [], [])
    a = ACDCLoader(tmp_path, split_manifest=m).load("train").ids()
    b = ACDCLoader(tmp_path, split_manifest=m).load("train").ids()
    assert list(a) == list(b)


# --- per-id accessors --------------------------------------------------------


def test_image_raw_labels_and_labels(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []), foreground_class=13)
    ds = loader.load("train")
    cid = ds.ids()[0]
    assert ds.image(cid).shape == (3, 4, 3)
    raw = ds.raw_labels(cid)
    assert raw.dtype == np.uint8
    assert set(np.unique(raw).tolist()).issubset(set(range(19)) | {255})
    label = ds.labels(cid)
    assert label.dtype == np.bool_
    np.testing.assert_array_equal(label, raw == 13)


def test_resolution_and_condition(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    ds = loader.load("train")
    assert ds.resolution(ds.ids()[0]) == (3, 4)
    assert ds.condition("GP01_frame_000001") == "fog"
    assert ds.condition("GP03_frame_000020") == "rain"


def test_labels_without_foreground_class_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    ds = loader.load("train")
    with pytest.raises(ValueError, match="needs an explicit foreground_class"):
        ds.labels(ds.ids()[0])


def test_unknown_id_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    ds = loader.load("train")
    with pytest.raises(ValueError, match="unknown id"):
        ds.image("does_not_exist")


# --- meta --------------------------------------------------------------------


def test_meta_content(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []), foreground_class=7)
    meta = loader.load("train").meta()
    assert meta["name"] == "acdc"
    assert meta["version"] == DATASET_METADATA["acdc"].version
    assert meta["label_map"] == "cityscapes_19"
    assert meta["num_classes"] == 19
    assert meta["ignore_index"] == 255
    assert meta["foreground_class"] == 7


# --- construction / config validation ---------------------------------------


def test_unsupported_label_map_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    with pytest.raises(ValueError, match="unsupported label_map"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []), label_map="ade20k")


def test_invalid_foreground_class_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    with pytest.raises(ValueError, match="not a valid trainId"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []), foreground_class=99)


def test_missing_rgb_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "gt_trainval" / "gt").mkdir(parents=True)
    with pytest.raises(ValueError, match="rgb_anon directory not found"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_missing_gt_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "rgb_anon_trainvaltest" / "rgb_anon").mkdir(parents=True)
    with pytest.raises(ValueError, match="gt directory not found"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_no_labelled_frames_raises(tmp_path: Path) -> None:
    (tmp_path / "rgb_anon_trainvaltest" / "rgb_anon").mkdir(parents=True)
    (tmp_path / "gt_trainval" / "gt").mkdir(parents=True)
    with pytest.raises(SerializationError, match="no labelled ACDC frames"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_missing_label_file_raises_at_discovery(tmp_path: Path) -> None:
    _build_acdc(tmp_path, skip_label_for="GP03_frame_000020")
    with pytest.raises(SerializationError, match="no paired label"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []))


def test_unknown_split_name_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    with pytest.raises(ValueError, match="split_name must be one of"):
        loader.load("validation")


def test_manifest_id_outside_pool_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    with pytest.raises(ValueError, match="not present in the discovered"):
        ACDCLoader(tmp_path, split_manifest=_manifest(["ghost_frame"], [], []))


def test_overlapping_manifest_raises(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    with pytest.raises(SplitLeakageError):
        ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS[:1], ALL_IDS[:1], []))


def test_duplicate_id_within_split_raises() -> None:
    # Exercised directly on ACDCDataset (a well-formed archive cannot produce
    # a within-split duplicate, but the guard must exist).
    from wfcrc.datasets.loaders.acdc import _ACDCFrame

    frame = _ACDCFrame(
        id_="x", image_path=Path("i"), label_path=Path("l"), condition="fog", official_split="train"
    )
    with pytest.raises(ValueError, match="duplicate id"):
        ACDCDataset([frame, frame], label_map="cityscapes_19", foreground_class=None)


# --- integrity ---------------------------------------------------------------


def test_verify_integrity_clean(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    report = loader.load("train").verify_integrity()
    assert report.ok
    assert report.issues == ()


def test_verify_integrity_shape_mismatch(tmp_path: Path) -> None:
    # Build normally, then overwrite one label with a different-sized image.
    _build_acdc(tmp_path)
    bad_label = (
        tmp_path
        / "gt_trainval"
        / "gt"
        / "fog"
        / "train"
        / "GP01"
        / "GP01_frame_000001_gt_labelTrainIds.png"
    )
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8), mode="L").save(bad_label)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    report = loader.load("train").verify_integrity()
    assert not report.ok
    assert any("shape mismatch" in issue.problem for issue in report.issues)


def test_verify_integrity_out_of_vocab_label(tmp_path: Path) -> None:
    _build_acdc(tmp_path, label_values=(0, 1, 100))  # 100 is not a valid trainId
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    report = loader.load("train").verify_integrity()
    assert not report.ok
    assert any("outside the declared" in issue.problem for issue in report.issues)


def test_verify_integrity_unreadable_image(tmp_path: Path) -> None:
    _build_acdc(tmp_path, corrupt_image_for="GP01_frame_000001")
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    report = loader.load("train").verify_integrity()
    assert not report.ok
    assert any("image unreadable" in issue.problem for issue in report.issues)


def test_verify_integrity_unreadable_label(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    bad_label = (
        tmp_path
        / "gt_trainval"
        / "gt"
        / "fog"
        / "train"
        / "GP01"
        / "GP01_frame_000001_gt_labelTrainIds.png"
    )
    bad_label.write_bytes(b"corrupt label bytes")
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    report = loader.load("train").verify_integrity()
    assert not report.ok
    assert any("label unreadable" in issue.problem for issue in report.issues)


# --- preprocessing compatibility ---------------------------------------------


def test_image_feeds_frozen_resize_and_normalize(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    ds = loader.load("train")
    image = ds.image(ds.ids()[0])
    out = resize_and_normalize(image, target_size=(2, 2), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    assert out.shape == (2, 2, 3)


def test_ignores_non_condition_directory(tmp_path: Path) -> None:
    _build_acdc(tmp_path)
    # A stray directory under rgb_anon that is not a known condition is skipped.
    (tmp_path / "rgb_anon_trainvaltest" / "rgb_anon" / "misc").mkdir()
    loader = ACDCLoader(tmp_path, split_manifest=_manifest(ALL_IDS, [], []))
    assert set(loader.load("train").ids()) == set(ALL_IDS)


def test_duplicate_stem_across_conditions_raises(tmp_path: Path) -> None:
    # Two frames sharing a stem across different condition dirs (real ACDC
    # stems are globally unique, but the discovery guard must exist).
    frames = (
        ("fog", "train", "S1", "DUP_frame_000001"),
        ("night", "train", "S2", "DUP_frame_000001"),
    )
    _build_acdc(tmp_path, frames=frames)
    with pytest.raises(ValueError, match="duplicate ACDC frame id"):
        ACDCLoader(tmp_path, split_manifest=_manifest([], [], []))
