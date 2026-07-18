"""Opt-in, marker-gated real-data integration test for `MSDNiftiLoader`
against actual, locally-acquired MSD Task04_Hippocampus data (MS6
Architecture Specification §6 / Q3 frozen policy, §8.3).

Excluded from the default suite by `pyproject.toml`'s
`addopts = "... -m 'not real_data'"`; run explicitly with
`pytest -m real_data tests/unit/datasets/loaders/test_msd_real_data.py`
once the dataset has been manually acquired per the acquisition guide in
`wfcrc/datasets/loaders/msd.py`'s module docstring (§7).

Per the MS6.3A task brief's Task 8: if the expected local path is absent,
this test **skips**, not fails — that is not an implementation failure,
just a not-yet-acquired dataset. No download happens automatically here or
anywhere else in this repository.

This test does **not** propose or exercise a specific WFCRC train/
calibration/test research split (see `msd.py` module docstring §3 — no
such split is frozen). Where a `split_manifest` is needed to exercise the
loader, either the entire discovered pool is used as a single "train"
split, or an arbitrary small technical partition is used purely to prove
the split-manifest *mechanism* — never presented as a scientific choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wfcrc.datasets.loaders.msd import MSDNiftiLoader

pytestmark = pytest.mark.real_data

#: Per the Dataset Selection Audit §5 / `DATASET_METADATA["msd_hippocampus"]`'s
#: `repo_cache_dir`, and `msd.py`'s own acquisition guide (§7).
EXPECTED_ROOT = Path("data/msd")
EXPECTED_TASK_DIR = EXPECTED_ROOT / "Task04_Hippocampus"
TASK = "Task04_Hippocampus"


def _skip_if_absent() -> None:
    if not EXPECTED_TASK_DIR.is_dir():
        pytest.skip(
            f"MSD Task04_Hippocampus not found at {EXPECTED_TASK_DIR.resolve()} — "
            "REAL-DATA INTEGRATION PENDING DATASET ACQUISITION. See "
            "wfcrc/datasets/loaders/msd.py's module docstring §7 for the acquisition guide."
        )


def _discovered_case_ids() -> list[str]:
    dataset_json = json.loads((EXPECTED_TASK_DIR / "dataset.json").read_text(encoding="utf-8"))
    ids = []
    for entry in dataset_json["training"]:
        name = Path(entry["image"]).name
        stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name.rsplit(".", 1)[0]
        ids.append(stem)
    return ids


def test_real_hippocampus_discovery_pairing_and_case_count() -> None:
    _skip_if_absent()
    case_ids = _discovered_case_ids()
    assert len(case_ids) > 0

    manifest = {"train": case_ids, "calibration": [], "test": []}
    loader = MSDNiftiLoader(EXPECTED_ROOT, TASK, split_manifest=manifest)
    dataset = loader.load("train")

    assert len(dataset) == len(case_ids)
    assert set(dataset.ids()) == set(case_ids)


def test_real_hippocampus_nifti_readability_dimensions_spacing_labels() -> None:
    _skip_if_absent()
    case_ids = _discovered_case_ids()
    manifest = {"train": case_ids, "calibration": [], "test": []}
    loader = MSDNiftiLoader(EXPECTED_ROOT, TASK, split_manifest=manifest)
    dataset = loader.load("train")

    first_id = dataset.ids()[0]
    _, image, label = next(iter(dataset))
    assert image.ndim == 3
    assert label.shape == image.shape
    assert label.dtype == np.bool_

    raw = dataset.raw_labels(first_id)
    assert set(np.unique(raw).tolist()).issubset({0, 1, 2})

    spacing = dataset.spacing(first_id)
    assert len(spacing) == 3
    assert all(s > 0 for s in spacing)


def test_real_hippocampus_metadata() -> None:
    _skip_if_absent()
    case_ids = _discovered_case_ids()
    manifest = {"train": case_ids[:1], "calibration": [], "test": []}
    loader = MSDNiftiLoader(EXPECTED_ROOT, TASK, split_manifest=manifest)
    meta = loader.load("train").meta()
    assert meta["name"] == "msd_hippocampus"
    assert meta["task"] == TASK
    assert set(meta["task_labels"].values()) >= {"background"}


def test_real_hippocampus_split_manifest_mechanism() -> None:
    # Exercises the split-manifest mechanism with an arbitrary technical
    # partition — not a proposed WFCRC research split (see module docstring).
    _skip_if_absent()
    case_ids = _discovered_case_ids()
    assert len(case_ids) >= 3
    manifest = {
        "train": case_ids[:-2],
        "calibration": case_ids[-2:-1],
        "test": case_ids[-1:],
    }
    loader = MSDNiftiLoader(EXPECTED_ROOT, TASK, split_manifest=manifest)
    train, cal, test = (loader.load(name) for name in ("train", "calibration", "test"))
    assert len(train) + len(cal) + len(test) == len(case_ids)
    assert set(train.ids()).isdisjoint(cal.ids())
    assert set(train.ids()).isdisjoint(test.ids())
    assert set(cal.ids()).isdisjoint(test.ids())
