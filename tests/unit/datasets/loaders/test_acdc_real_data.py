"""Opt-in, marker-gated real-data validation for ACDC (DI-2).

Excluded from the default suite (``-m 'not real_data'``). Points at this
environment's actual archive location (``datasets/ACDC/``); skips if absent.
Discovery/counts are checked over the full 2006-frame pool; pixel reads and
``verify_integrity`` are checked over a small sample (1920x1080 PNGs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfcrc.datasets.loaders.acdc import ACDCLoader

pytestmark = pytest.mark.real_data

ROOT = Path("datasets/ACDC")


def _skip_if_absent() -> None:
    if not (ROOT / "gt_trainval" / "gt").is_dir():
        pytest.skip(f"ACDC not found at {ROOT.resolve()}")


def _all_ids() -> list[str]:
    loader = ACDCLoader(ROOT, split_manifest={"train": [], "calibration": [], "test": []})
    return sorted(loader._frames)


def test_acdc_discovery_count_and_conditions() -> None:
    _skip_if_absent()
    ids = _all_ids()
    assert len(ids) == 2006  # 1600 train + 406 val, verified DI-2
    assert len(set(ids)) == 2006  # globally-unique stems
    loader = ACDCLoader(ROOT, split_manifest={"train": ids, "calibration": [], "test": []})
    ds = loader.load("train")
    conditions = {ds.condition(i) for i in ids}
    assert conditions == {"fog", "night", "rain", "snow"}


def test_acdc_reads_real_frames_and_labels() -> None:
    _skip_if_absent()
    ids = _all_ids()
    sample = ids[:2] + ids[-2:]
    loader = ACDCLoader(
        ROOT, split_manifest={"train": sample, "calibration": [], "test": []}, foreground_class=13
    )
    ds = loader.load("train")
    for cid in sample:
        image = ds.image(cid)
        raw = ds.raw_labels(cid)
        label = ds.labels(cid)
        assert image.ndim == 3 and image.shape[2] == 3
        assert raw.shape == image.shape[:2]
        assert set(np.unique(raw).tolist()).issubset(set(range(19)) | {255})
        assert label.dtype == np.bool_
        assert ds.resolution(cid) == image.shape[:2]


def test_acdc_integrity_and_meta_on_sample() -> None:
    _skip_if_absent()
    sample = _all_ids()[:4]
    loader = ACDCLoader(
        ROOT, split_manifest={"train": sample, "calibration": [], "test": []}, foreground_class=13
    )
    ds = loader.load("train")
    assert ds.verify_integrity().ok
    assert ds.meta()["name"] == "acdc"
    assert ds.meta()["num_classes"] == 19
