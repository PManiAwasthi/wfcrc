"""Unit tests for :mod:`wfcrc.datasets.loaders.msd` (MS6.3A).

All tests run against a tiny **synthetic** fixture that reproduces the real
MSD Task04_Hippocampus directory structure exactly (`dataset.json` +
`imagesTr`/`labelsTr`/`imagesTs`, identical image/label filename stems) —
never real downloaded data. See `test_msd_real_data.py` for the opt-in,
marker-gated real-data integration test (MS6 Architecture Specification §6,
Q3 frozen policy).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from wfcrc.datasets.loaders.msd import MSDDataset, MSDNiftiLoader
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.datasets.preprocessing import resample_volume
from wfcrc.exceptions import SerializationError, SplitLeakageError

TASK = "Task04_Hippocampus"
SHAPE = (4, 5, 6)
SPACING = (1.0, 1.5, 2.0)


def _write_volume(path: Path, array: np.ndarray, spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    image.header.set_zooms(spacing)
    nib.save(image, str(path))


def _build_task(
    root_dir: Path,
    case_ids: Sequence[str],
    *,
    task: str = TASK,
    shape: tuple[int, int, int] = SHAPE,
    spacing: tuple[float, float, float] = SPACING,
    test_case_ids: Sequence[str] = (),
    skip_image_for: str | None = None,
    skip_label_for: str | None = None,
    corrupt_label_for: str | None = None,
    label_mismatch_for: str | None = None,
) -> Path:
    """Build a synthetic MSD task directory: one valid case per id in `case_ids`.

    Every case's label has one Anterior (`1`) voxel and one Posterior (`2`)
    voxel, so both the foreground mask and the raw multi-class volume are
    non-trivial. `skip_image_for`/`skip_label_for` omit writing that case's
    image/label file (but still reference it in `dataset.json`, simulating
    a missing-file dataset). `corrupt_label_for` writes garbage bytes
    instead of a real NIfTI file for that case's label. `label_mismatch_for`
    makes that case's `dataset.json` "label" entry point at a differently
    named file than its "image" entry.
    """
    task_dir = root_dir / task
    training_entries = []
    for i, case_id in enumerate(case_ids):
        image = np.zeros(shape, dtype=np.float32)
        image[0, 0, 0] = 100.0 + i
        label = np.zeros(shape, dtype=np.float32)
        label[0, 0, 0] = 1.0
        label[-1, -1, -1] = 2.0

        if case_id != skip_image_for:
            _write_volume(task_dir / "imagesTr" / f"{case_id}.nii.gz", image, spacing)
        if case_id == corrupt_label_for:
            label_path = task_dir / "labelsTr" / f"{case_id}.nii.gz"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_bytes(b"not a real nifti file")
        elif case_id != skip_label_for:
            _write_volume(task_dir / "labelsTr" / f"{case_id}.nii.gz", label, spacing)

        label_name = f"{case_id}_wrongname.nii.gz" if case_id == label_mismatch_for else case_id
        training_entries.append(
            {
                "image": f"./imagesTr/{case_id}.nii.gz",
                "label": f"./labelsTr/{label_name}.nii.gz",
            }
        )

    test_entries = []
    for case_id in test_case_ids:
        _write_volume(task_dir / "imagesTs" / f"{case_id}.nii.gz", np.zeros(shape), spacing)
        test_entries.append(f"./imagesTs/{case_id}.nii.gz")

    dataset_json = {
        "name": "Hippocampus",
        "description": "synthetic MS6.3A test fixture",
        "reference": "synthetic",
        "licence": "synthetic",
        "release": "0.0",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI"},
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "numTraining": len(case_ids),
        "numTest": len(test_case_ids),
        "training": training_entries,
        "test": test_entries,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    return task_dir


def _manifest(
    train: Sequence[str], calibration: Sequence[str], test: Sequence[str]
) -> dict[str, list[str]]:
    return {"train": list(train), "calibration": list(calibration), "test": list(test)}


CASE_IDS = ["hippocampus_001", "hippocampus_002", "hippocampus_003", "hippocampus_004"]


@pytest.fixture
def basic_root(tmp_path: Path) -> Path:
    _build_task(tmp_path, CASE_IDS, test_case_ids=["hippocampus_999"])
    return tmp_path


@pytest.fixture
def basic_loader(basic_root: Path) -> MSDNiftiLoader:
    manifest = _manifest(
        train=["hippocampus_001"],
        calibration=["hippocampus_002", "hippocampus_003"],
        test=["hippocampus_004"],
    )
    return MSDNiftiLoader(basic_root, TASK, split_manifest=manifest)


# --- discovery, pairing, stable ids -----------------------------------------


def test_load_train_split_len_and_ids(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    assert len(dataset) == 1
    assert dataset.ids() == ("hippocampus_001",)


def test_load_calibration_split_len_and_ids(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("calibration")
    assert len(dataset) == 2
    assert set(dataset.ids()) == {"hippocampus_002", "hippocampus_003"}


def test_load_test_split_len_and_ids(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("test")
    assert len(dataset) == 1
    assert dataset.ids() == ("hippocampus_004",)


def test_ids_are_stable_across_repeated_loads(basic_loader: MSDNiftiLoader) -> None:
    assert basic_loader.load("calibration").ids() == basic_loader.load("calibration").ids()


def test_official_unlabelled_test_cases_are_never_loadable(basic_root: Path) -> None:
    # dataset.json's own "test" list (hippocampus_999) is never in the
    # discoverable id pool; a split_manifest cannot reference it.
    with pytest.raises(ValueError, match="hippocampus_999"):
        MSDNiftiLoader(
            basic_root,
            TASK,
            split_manifest=_manifest(["hippocampus_999"], [], []),
        )


# --- iteration / len / labels / meta / spacing ------------------------------


def test_iteration_yields_len_triples(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("calibration")
    triples = list(dataset)
    assert len(triples) == len(dataset)
    yielded_ids = {t[0] for t in triples}
    assert yielded_ids == set(dataset.ids())


def test_iteration_image_and_label_shapes(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    _id, image, label = next(iter(dataset))
    assert image.shape == SHAPE
    assert label.shape == SHAPE
    assert label.dtype == np.bool_


def test_labels_returns_boolean_foreground_mask(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    label = dataset.labels("hippocampus_001")
    assert label.dtype == np.bool_
    assert label[0, 0, 0]  # Anterior voxel -> foreground
    assert label[-1, -1, -1]  # Posterior voxel -> foreground
    assert not label[1, 1, 1]  # background voxel


def test_raw_labels_preserves_multiclass_values(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    raw = dataset.raw_labels("hippocampus_001")
    assert raw.dtype == np.int64
    assert raw[0, 0, 0] == 1
    assert raw[-1, -1, -1] == 2
    assert raw[1, 1, 1] == 0


def test_labels_unknown_id_raises_value_error(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    with pytest.raises(ValueError, match="unknown id"):
        dataset.labels("not-a-real-id")


def test_meta_matches_dataset_metadata_and_records_task(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    meta = dataset.meta()
    assert meta["name"] == DATASET_METADATA["msd_hippocampus"].name
    assert meta["license"] == DATASET_METADATA["msd_hippocampus"].license
    assert meta["task"] == TASK
    assert meta["task_labels"] == {"0": "background", "1": "Anterior", "2": "Posterior"}


def test_spacing_matches_nifti_header(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    spacing = dataset.spacing("hippocampus_001")
    assert spacing == pytest.approx(SPACING)


# --- preprocessing compatibility --------------------------------------------


def test_loader_output_is_compatible_with_frozen_resample_volume(
    basic_loader: MSDNiftiLoader,
) -> None:
    dataset = basic_loader.load("train")
    _, image, _ = next(iter(dataset))
    spacing = dataset.spacing("hippocampus_001")
    target_spacing = (1.0, 1.0, 1.0)
    resampled = resample_volume(image, spacing, target_spacing)
    assert resampled.dtype == np.float64
    assert resampled.ndim == 3


# --- missing / malformed data ------------------------------------------------


def test_missing_image_file_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, CASE_IDS, skip_image_for="hippocampus_002")
    with pytest.raises(SerializationError, match="does not exist"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(CASE_IDS, [], []))


def test_missing_label_file_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, CASE_IDS, skip_label_for="hippocampus_003")
    with pytest.raises(SerializationError, match="does not exist"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(CASE_IDS, [], []))


def test_malformed_nifti_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, CASE_IDS, corrupt_label_for="hippocampus_004")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(CASE_IDS, [], []))
    dataset = loader.load("train")
    with pytest.raises(SerializationError, match="could not read NIfTI"):
        dataset.labels("hippocampus_004")


def test_uncompressed_nii_extension_is_supported(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    image = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), affine=np.eye(4))
    image.header.set_zooms(SPACING)
    (task_dir / "imagesTr").mkdir(parents=True)
    (task_dir / "labelsTr").mkdir(parents=True)
    nib.save(image, str(task_dir / "imagesTr" / "case_a.nii"))
    nib.save(image, str(task_dir / "labelsTr" / "case_a.nii"))
    dataset_json = {
        "training": [{"image": "./imagesTr/case_a.nii", "label": "./labelsTr/case_a.nii"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["case_a"], [], []))
    assert loader.load("train").ids() == ("case_a",)


def test_non_nifti_filename_in_dataset_json_raises_serialization_error(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    task_dir.mkdir(parents=True)
    dataset_json = {
        "training": [{"image": "./imagesTr/case_a.mha", "label": "./labelsTr/case_a.mha"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    with pytest.raises(SerializationError, match=r"expected a \.nii or \.nii\.gz filename"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


def test_file_removed_after_discovery_raises_serialization_error_on_lazy_access(
    tmp_path: Path,
) -> None:
    # Discovery only checks existence once, at construction time; access is
    # lazy, so a file removed afterwards must still fail explicitly rather
    # than silently returning stale/garbage data.
    _build_task(tmp_path, ["hippocampus_001"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))
    dataset = loader.load("train")
    (tmp_path / TASK / "imagesTr" / "hippocampus_001.nii.gz").unlink()
    with pytest.raises(SerializationError, match="does not exist"):
        next(iter(dataset))


def test_non_3d_volume_raises_serialization_error(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    (task_dir / "imagesTr").mkdir(parents=True)
    (task_dir / "labelsTr").mkdir(parents=True)
    image_2d = nib.Nifti1Image(np.zeros((4, 4), dtype=np.float32), affine=np.eye(4))
    label_3d = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), affine=np.eye(4))
    nib.save(image_2d, str(task_dir / "imagesTr" / "case_a.nii.gz"))
    nib.save(label_3d, str(task_dir / "labelsTr" / "case_a.nii.gz"))
    dataset_json = {
        "training": [{"image": "./imagesTr/case_a.nii.gz", "label": "./labelsTr/case_a.nii.gz"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["case_a"], [], []))
    dataset = loader.load("train")
    with pytest.raises(SerializationError, match="expected a 3-D NIfTI volume"):
        next(iter(dataset))


def test_dataset_json_top_level_not_an_object_raises_serialization_error(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    task_dir.mkdir(parents=True)
    (task_dir / "dataset.json").write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(SerializationError, match="must contain a JSON object"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


def test_malformed_training_entry_raises_serialization_error(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    task_dir.mkdir(parents=True)
    dataset_json = {"training": [{"image": "./imagesTr/case_a.nii.gz"}], "test": []}  # no "label"
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    with pytest.raises(SerializationError, match=r"malformed dataset\.json 'training' entry"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


def test_image_label_filename_mismatch_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, CASE_IDS, label_mismatch_for="hippocampus_001")
    with pytest.raises(SerializationError, match="mismatch"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(CASE_IDS, [], []))


def test_duplicate_id_in_dataset_json_raises_value_error(tmp_path: Path) -> None:
    task_dir = _build_task(tmp_path, ["hippocampus_001"])
    dataset_json_path = task_dir / "dataset.json"
    payload = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    payload["training"].append(dict(payload["training"][0]))  # duplicate entry, same id
    dataset_json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))


def test_missing_dataset_json_raises_serialization_error(tmp_path: Path) -> None:
    (tmp_path / TASK).mkdir(parents=True)
    with pytest.raises(SerializationError, match=r"dataset\.json"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


def test_malformed_dataset_json_raises_serialization_error(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    task_dir.mkdir(parents=True)
    (task_dir / "dataset.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SerializationError, match="could not read/parse"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


def test_dataset_json_without_training_list_raises_serialization_error(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    task_dir.mkdir(parents=True)
    (task_dir / "dataset.json").write_text(json.dumps({"training": []}), encoding="utf-8")
    with pytest.raises(SerializationError, match="training"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


# --- unknown task / nonexistent root ----------------------------------------


def test_unknown_task_raises_value_error(tmp_path: Path) -> None:
    _build_task(tmp_path, CASE_IDS)
    with pytest.raises(ValueError, match="unsupported MSD task"):
        MSDNiftiLoader(tmp_path, "Task07_Pancreas", split_manifest=_manifest([], [], []))


def test_task_directory_not_found_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest([], [], []))


def test_nonexistent_root_raises_value_error(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist_at_all"
    with pytest.raises(ValueError, match="not found"):
        MSDNiftiLoader(nonexistent, TASK, split_manifest=_manifest([], [], []))


# --- split validation ---------------------------------------------------------


def test_invalid_split_name_raises_value_error(basic_loader: MSDNiftiLoader) -> None:
    with pytest.raises(ValueError, match="split_name must be one of"):
        basic_loader.load("validation")


def test_split_overlap_raises_split_leakage_error(basic_root: Path) -> None:
    manifest = _manifest(
        train=["hippocampus_001", "hippocampus_002"],
        calibration=["hippocampus_002", "hippocampus_003"],  # overlaps train
        test=["hippocampus_004"],
    )
    with pytest.raises(SplitLeakageError):
        MSDNiftiLoader(basic_root, TASK, split_manifest=manifest)


def test_split_manifest_missing_key_raises_value_error(basic_root: Path) -> None:
    incomplete = {"train": ["hippocampus_001"], "calibration": ["hippocampus_002"]}
    with pytest.raises(ValueError, match="missing required split"):
        MSDNiftiLoader(basic_root, TASK, split_manifest=incomplete)


def test_split_manifest_unrecognized_key_raises_value_error(basic_root: Path) -> None:
    manifest = _manifest(["hippocampus_001"], ["hippocampus_002"], ["hippocampus_003"])
    manifest["validation"] = ["hippocampus_004"]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unrecognized split name"):
        MSDNiftiLoader(basic_root, TASK, split_manifest=manifest)


def test_split_manifest_not_a_mapping_raises_value_error(basic_root: Path) -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        MSDNiftiLoader(basic_root, TASK, split_manifest=["not", "a", "mapping"])  # type: ignore[arg-type]


def test_split_manifest_unknown_id_raises_value_error(basic_root: Path) -> None:
    manifest = _manifest(["hippocampus_001", "not-a-real-case"], [], [])
    with pytest.raises(ValueError, match="not present in the discovered"):
        MSDNiftiLoader(basic_root, TASK, split_manifest=manifest)


def test_split_manifest_file_not_found_raises_value_error(basic_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split manifest file not found"):
        MSDNiftiLoader(basic_root, TASK, split_manifest=tmp_path / "missing_manifest.json")


def test_split_manifest_malformed_file_raises_serialization_error(
    basic_root: Path, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SerializationError, match="could not read/parse"):
        MSDNiftiLoader(basic_root, TASK, split_manifest=manifest_path)


def test_split_manifest_as_json_file_path_is_accepted(basic_root: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(["hippocampus_001"], ["hippocampus_002"], ["hippocampus_003"])),
        encoding="utf-8",
    )
    loader = MSDNiftiLoader(basic_root, TASK, split_manifest=manifest_path)
    assert loader.load("train").ids() == ("hippocampus_001",)


# --- MSDDataset direct construction (empty split) ---------------------------


def test_empty_split_has_zero_length(basic_root: Path) -> None:
    manifest = _manifest([], CASE_IDS[:2], CASE_IDS[2:])
    loader = MSDNiftiLoader(basic_root, TASK, split_manifest=manifest)
    empty = loader.load("train")
    assert len(empty) == 0
    assert list(empty) == []
    assert isinstance(empty, MSDDataset)


# --- DI-1: direct image access -----------------------------------------------


def test_image_returns_native_resolution_array(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    image = dataset.image("hippocampus_001")
    assert image.shape == SHAPE
    assert image.dtype == np.float64
    assert image[0, 0, 0] == pytest.approx(100.0)  # case index 0 -> value 100.0 + 0


def test_image_matches_iteration_output(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    _id, iter_image, _label = next(iter(dataset))
    direct_image = dataset.image("hippocampus_001")
    assert np.array_equal(iter_image, direct_image)


def test_image_unknown_id_raises_value_error(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    with pytest.raises(ValueError, match="unknown id"):
        dataset.image("not-a-real-id")


def test_image_missing_file_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, ["hippocampus_001"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))
    dataset = loader.load("train")
    (tmp_path / TASK / "imagesTr" / "hippocampus_001.nii.gz").unlink()
    with pytest.raises(SerializationError, match="does not exist"):
        dataset.image("hippocampus_001")


# --- DI-1: orientation --------------------------------------------------------


def test_orientation_returns_axis_codes_for_identity_affine(basic_loader: MSDNiftiLoader) -> None:
    # The synthetic fixture writes every volume with affine=np.eye(4), whose
    # canonical nibabel orientation is ("R", "A", "S").
    dataset = basic_loader.load("train")
    assert dataset.orientation("hippocampus_001") == ("R", "A", "S")


def test_orientation_unknown_id_raises_value_error(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("train")
    with pytest.raises(ValueError, match="unknown id"):
        dataset.orientation("not-a-real-id")


def test_orientation_missing_file_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, ["hippocampus_001"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))
    dataset = loader.load("train")
    (tmp_path / TASK / "imagesTr" / "hippocampus_001.nii.gz").unlink()
    with pytest.raises(SerializationError, match="does not exist"):
        dataset.orientation("hippocampus_001")


def test_orientation_malformed_nifti_raises_serialization_error(tmp_path: Path) -> None:
    _build_task(tmp_path, ["hippocampus_001"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))
    dataset = loader.load("train")
    (tmp_path / TASK / "imagesTr" / "hippocampus_001.nii.gz").write_bytes(b"not a real nifti file")
    with pytest.raises(SerializationError, match="could not read NIfTI"):
        dataset.orientation("hippocampus_001")


# --- DI-1: verify_integrity ---------------------------------------------------


def test_verify_integrity_reports_ok_for_a_clean_split(basic_loader: MSDNiftiLoader) -> None:
    dataset = basic_loader.load("calibration")
    report = dataset.verify_integrity()
    assert report.ok is True
    report.assert_ok()  # must not raise


def test_verify_integrity_detects_shape_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    _write_volume(task_dir / "imagesTr" / "case_a.nii.gz", np.zeros((4, 5, 6)), SPACING)
    _write_volume(task_dir / "labelsTr" / "case_a.nii.gz", np.zeros((4, 5, 7)), SPACING)
    dataset_json = {
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "training": [{"image": "./imagesTr/case_a.nii.gz", "label": "./labelsTr/case_a.nii.gz"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["case_a"], [], []))
    report = loader.load("train").verify_integrity()
    assert report.ok is False
    assert any("shape mismatch" in issue.problem for issue in report.issues)


def test_verify_integrity_detects_non_finite_image(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    bad_image = np.zeros(SHAPE, dtype=np.float32)
    bad_image[0, 0, 0] = np.nan
    _write_volume(task_dir / "imagesTr" / "case_a.nii.gz", bad_image, SPACING)
    _write_volume(task_dir / "labelsTr" / "case_a.nii.gz", np.zeros(SHAPE), SPACING)
    dataset_json = {
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "training": [{"image": "./imagesTr/case_a.nii.gz", "label": "./labelsTr/case_a.nii.gz"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["case_a"], [], []))
    report = loader.load("train").verify_integrity()
    assert report.ok is False
    assert any("NaN/Inf" in issue.problem for issue in report.issues)


def test_verify_integrity_detects_unexpected_label_value(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    _write_volume(task_dir / "imagesTr" / "case_a.nii.gz", np.zeros(SHAPE), SPACING)
    bad_label = np.zeros(SHAPE, dtype=np.float32)
    bad_label[0, 0, 0] = 7.0  # not in the declared {0, 1, 2} label map
    _write_volume(task_dir / "labelsTr" / "case_a.nii.gz", bad_label, SPACING)
    dataset_json = {
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "training": [{"image": "./imagesTr/case_a.nii.gz", "label": "./labelsTr/case_a.nii.gz"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["case_a"], [], []))
    report = loader.load("train").verify_integrity()
    assert report.ok is False
    assert any("outside the declared label map" in issue.problem for issue in report.issues)


def test_verify_integrity_reports_unreadable_files_without_raising(tmp_path: Path) -> None:
    _build_task(tmp_path, ["hippocampus_001"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))
    dataset = loader.load("train")
    (tmp_path / TASK / "labelsTr" / "hippocampus_001.nii.gz").unlink()
    report = dataset.verify_integrity()
    assert report.ok is False
    assert any("label unreadable" in issue.problem for issue in report.issues)


def test_verify_integrity_reports_unreadable_image_without_raising(tmp_path: Path) -> None:
    _build_task(tmp_path, ["hippocampus_001"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["hippocampus_001"], [], []))
    dataset = loader.load("train")
    (tmp_path / TASK / "imagesTr" / "hippocampus_001.nii.gz").unlink()
    report = dataset.verify_integrity()
    assert report.ok is False
    assert any("image unreadable" in issue.problem for issue in report.issues)


def test_verify_integrity_detects_non_finite_label(tmp_path: Path) -> None:
    task_dir = tmp_path / TASK
    _write_volume(task_dir / "imagesTr" / "case_a.nii.gz", np.zeros(SHAPE), SPACING)
    bad_label = np.zeros(SHAPE, dtype=np.float32)
    bad_label[0, 0, 0] = np.nan
    _write_volume(task_dir / "labelsTr" / "case_a.nii.gz", bad_label, SPACING)
    dataset_json = {
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "training": [{"image": "./imagesTr/case_a.nii.gz", "label": "./labelsTr/case_a.nii.gz"}],
        "test": [],
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=_manifest(["case_a"], [], []))
    report = loader.load("train").verify_integrity()
    assert report.ok is False
    assert any("label contains NaN/Inf" in issue.problem for issue in report.issues)


def test_verify_integrity_on_empty_split_is_ok(basic_root: Path) -> None:
    manifest = _manifest([], CASE_IDS[:2], CASE_IDS[2:])
    loader = MSDNiftiLoader(basic_root, TASK, split_manifest=manifest)
    report = loader.load("train").verify_integrity()
    assert report.ok is True


# --- DI-1: duplicate id within a single split --------------------------------


def test_duplicate_id_within_a_single_split_raises_value_error(tmp_path: Path) -> None:
    _build_task(tmp_path, ["hippocampus_001", "hippocampus_002"])
    # A caller-supplied split_manifest listing the same id twice within one
    # split (distinct from the already-covered "duplicate within
    # dataset.json's own training list" case, `_discover_cases`). Neither
    # the frozen A1 gate (`assert_split_disjoint`, which only checks
    # *between*-split overlap) nor `MSDNiftiLoader.__init__` itself catches
    # this -- it only surfaces once `.load()` actually constructs the
    # `MSDDataset` for that split.
    manifest = _manifest(["hippocampus_001", "hippocampus_001"], [], ["hippocampus_002"])
    loader = MSDNiftiLoader(tmp_path, TASK, split_manifest=manifest)
    with pytest.raises(ValueError, match="duplicate id"):
        loader.load("train")
