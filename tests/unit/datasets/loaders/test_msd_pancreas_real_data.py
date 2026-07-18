"""Opt-in, marker-gated real-data validation for MSD Task07_Pancreas (DI-2).

Excluded from the default suite by ``pyproject.toml``'s
``-m 'not real_data'``. Points at this environment's actual archive location
(``datasets/Task07_Pancreas/``); skips cleanly if absent. Pancreas volumes
are large CT scans, so only a few cases are read end-to-end — discovery and
counts (fast) are checked over the full pool.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wfcrc.datasets.loaders.msd import MSDNiftiLoader

pytestmark = pytest.mark.real_data

ROOT = Path("datasets")
TASK = "Task07_Pancreas"
TASK_DIR = ROOT / TASK


def _skip_if_absent() -> None:
    if not (TASK_DIR / "dataset.json").is_file():
        pytest.skip(f"MSD Task07_Pancreas not found at {TASK_DIR.resolve()}")


def _ids() -> list[str]:
    dj = json.loads((TASK_DIR / "dataset.json").read_text(encoding="utf-8"))
    return [Path(e["image"]).name[: -len(".nii.gz")] for e in dj["training"]]


def test_pancreas_discovery_and_count() -> None:
    _skip_if_absent()
    ids = _ids()
    assert len(ids) == 281  # real archive, verified DI-2
    loader = MSDNiftiLoader(
        ROOT, TASK, split_manifest={"train": ids, "calibration": [], "test": []}
    )
    ds = loader.load("train")
    assert len(ds) == 281
    assert len(set(ds.ids())) == 281  # no duplicates


def test_pancreas_reads_real_cases_and_labels() -> None:
    _skip_if_absent()
    ids = _ids()[:2]
    loader = MSDNiftiLoader(
        ROOT, TASK, split_manifest={"train": ids, "calibration": [], "test": []}
    )
    ds = loader.load("train")
    for cid in ids:
        image = ds.image(cid)
        label = ds.labels(cid)
        raw = ds.raw_labels(cid)
        assert image.ndim == 3
        assert label.shape == image.shape and label.dtype == np.bool_
        assert set(np.unique(raw).tolist()).issubset({0, 1, 2})
        assert ds.orientation(cid) == ("R", "A", "S")
        assert len(ds.spacing(cid)) == 3


def test_pancreas_meta_and_integrity() -> None:
    _skip_if_absent()
    ids = _ids()[:2]
    loader = MSDNiftiLoader(
        ROOT, TASK, split_manifest={"train": ids, "calibration": [], "test": []}
    )
    ds = loader.load("train")
    assert ds.meta()["name"] == "msd_pancreas"
    assert set(ds.meta()["task_labels"].values()) == {"background", "pancreas", "cancer"}
    assert ds.verify_integrity().ok
