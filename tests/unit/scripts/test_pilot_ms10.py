"""MS10 — opt-in real-data regression wrapper for `scripts/pilot_ms10.py`.

Excluded from the default suite (`pyproject.toml`'s `-m 'not real_data'`);
run explicitly with `pytest -m real_data`. Skips cleanly if the dataset is
not present locally, per the opt-in real-data philosophy established in
MS6.3A/MS7. This is the checked-in, re-runnable regression form of the
pilot MS10 executed once to produce `docs/PILOT_REPORT.md` — running it
again must reproduce the same qualitative outcome (every baseline
succeeds, deterministic, every failure-injection check graceful).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from scripts import pilot_ms10

from wfcrc.baselines import BASELINES
from wfcrc.calibration.loss_table import LossTable


def _skip_if_absent() -> None:
    if not pilot_ms10.dataset_present():
        pytest.skip(
            f"MSD Task04_Hippocampus not found at {pilot_ms10.TASK_DIR.resolve()} -- "
            "MS10 pilot regression skipped, per the opt-in real-data philosophy."
        )


@pytest.mark.real_data
def test_pilot_runs_every_registered_baseline_without_error(tmp_path) -> None:
    _skip_if_absent()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    inputs = pilot_ms10.build_pipeline_inputs(work_dir)
    baselines = pilot_ms10.build_baselines(inputs["cal_loss_table"])

    n_test = inputs["test_loss_table"].shape[0]
    test_groups = (tuple(range(0, n_test // 2)), tuple(range(n_test // 2, n_test)))

    for name, calibrator in baselines.items():
        outcome = pilot_ms10.run_one_baseline(
            name,
            calibrator,
            inputs["cal_loss_table"],
            inputs["test_loss_table"],
            inputs["cfg"],
            test_groups,
        )
        assert outcome["error"] is None, f"{name} raised: {outcome['error']}"
        assert outcome["deterministic"] is True, f"{name} was not deterministic"
        assert outcome["lambda_hat"] in inputs["cal_loss_table"].lambda_grid

    shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.real_data
def test_failure_injection_checks_all_fail_gracefully(tmp_path) -> None:
    _skip_if_absent()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    inputs = pilot_ms10.build_pipeline_inputs(work_dir)
    results = pilot_ms10.failure_injection_checks(inputs, work_dir)

    assert set(results) == {
        "missing_checkpoint",
        "corrupted_cache",
        "empty_calibration_split",
        "invalid_split_manifest",
        "missing_case_id",
    }
    for name, outcome in results.items():
        assert outcome["graceful"] is True, f"{name} did not fail gracefully: {outcome['detail']}"

    shutil.rmtree(work_dir, ignore_errors=True)


def test_pilot_baseline_set_matches_the_registry() -> None:
    """`build_baselines` must exercise every registered baseline -- no silent omission."""
    dummy = LossTable(
        values=np.linspace(1.0, 0.0, 8 * 6).reshape(8, 6),
        lambda_grid=np.linspace(0.0, 5.0, 6),
    )
    baselines = pilot_ms10.build_baselines(dummy)
    assert set(baselines) == set(BASELINES)
