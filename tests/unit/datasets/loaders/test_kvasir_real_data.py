"""Opt-in, marker-gated real-data validation for Kvasir-SEG (DI-2).

Excluded from the default suite (``-m 'not real_data'``). Points at this
environment's actual archive location; skips if absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfcrc.datasets.loaders.kvasir import KvasirLoader

pytestmark = pytest.mark.real_data

ROOT = Path("datasets/Kvasir_SEG/kvasir-seg/Kvasir-SEG")


def _skip_if_absent() -> None:
    if not (ROOT / "images").is_dir():
        pytest.skip(f"Kvasir-SEG not found at {ROOT.resolve()}")


def _all_ids() -> list[str]:
    loader = KvasirLoader(ROOT, split_manifest={"train": [], "calibration": [], "test": []})
    return sorted(loader._cases)


def test_kvasir_discovery_count() -> None:
    _skip_if_absent()
    ids = _all_ids()
    assert len(ids) == 1000  # verified DI-2
    assert len(set(ids)) == 1000


def test_kvasir_reads_real_images_and_masks() -> None:
    _skip_if_absent()
    sample = _all_ids()[:4]
    ds = KvasirLoader(ROOT, split_manifest={"train": sample, "calibration": [], "test": []}).load(
        "train"
    )
    for cid in sample:
        image = ds.image(cid)
        label = ds.labels(cid)
        assert image.ndim == 3 and image.shape[2] == 3
        assert label.dtype == np.bool_
        assert label.shape == image.shape[:2]
        # every Kvasir-SEG image contains a polyp -> non-empty foreground
        assert label.any()


def test_kvasir_integrity_and_meta_on_sample() -> None:
    _skip_if_absent()
    sample = _all_ids()[:4]
    ds = KvasirLoader(ROOT, split_manifest={"train": sample, "calibration": [], "test": []}).load(
        "train"
    )
    assert ds.verify_integrity().ok
    meta = ds.meta()
    assert meta["name"] == "kvasir_seg"
    assert "UNRESOLVED" in meta["split_unit_status"]
