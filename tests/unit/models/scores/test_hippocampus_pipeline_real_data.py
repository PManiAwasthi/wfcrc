"""MS7 — the first complete end-to-end WFCRC pipeline, opt-in real-data test.

Exercises the full vertical slice against real, locally-acquired MSD
Task04_Hippocampus data (MS6.3A real-data validation precedent):

    real dataset -> real MSDNiftiLoader -> real HippocampusScoreProvider
    -> real boolean score tensors -> LossTableBuilder (frozen)
    -> WFCRC calibration (frozen) -> prediction-set construction (frozen,
    happens *inside* LossTableBuilder.build(), before calibration runs —
    see `test_contract_boundaries_are_all_exercised`'s docstring for why
    that ordering matters) -> evaluation (frozen `run_experiment`).

Excluded from the default suite (`pyproject.toml`'s `-m 'not real_data'`);
run explicitly with `pytest -m real_data`. Skips cleanly (not a failure)
if the dataset is not present locally, per the opt-in real-data philosophy
established in MS6.3A. No training, no optimizer, no GPU — inference only,
against a deterministically-initialized, never-trained checkpoint (see
`wfcrc/models/scores/hippocampus_segmenter.py`'s module docstring for why).

This is a **framework-validation smoke test** (MS7 Task 6): its purpose is
proving the pipeline executes end to end and every interface boundary
holds, not demonstrating segmentation accuracy. It uses a tiny,
non-scientific subset of cases — see `docs/DATASET_SPLIT_POLICY.md` §3.1's
own "reduced-scale exception" for the MS7 smoke pipeline, which explicitly
distinguishes this from the frozen 60/20/20 research-scale split policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.config.schema import CalibrationConfig
from wfcrc.datasets.loaders.msd import MSDNiftiLoader
from wfcrc.datasets.loss_table_builder import LossTableBuilder
from wfcrc.evaluation.experiment import run_experiment
from wfcrc.evaluation.verifier import Verifier
from wfcrc.losses.fnr import FNRLoss
from wfcrc.models.scores.hippocampus_segmenter import (
    HippocampusScoreProvider,
    create_untrained_checkpoint,
)
from wfcrc.prediction_sets.segmentation import MorphologicalSets

pytestmark = pytest.mark.real_data

ROOT_DIR = Path("datasets/Task04_Hippocampus")
TASK = "Task04_Hippocampus"
TASK_DIR = ROOT_DIR / TASK

#: A tiny, fixed (deterministic, non-random) subset of real case ids —
#: an engineering smoke-test convenience, not a research split (see module
#: docstring and DATASET_SPLIT_POLICY.md §3.1's "reduced-scale exception").
_N_CALIBRATION = 8
_N_TEST = 4

#: A dilation-radius grid for `MorphologicalSets(direction="dilation")`.
_LAMBDA_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def _skip_if_absent() -> None:
    if not TASK_DIR.is_dir():
        pytest.skip(
            f"MSD Task04_Hippocampus not found at {TASK_DIR.resolve()} — "
            "real-data pipeline test skipped, per the opt-in philosophy "
            "established in MS6.3A."
        )


def _smoke_manifest() -> dict[str, list[str]]:
    dataset_json = json.loads((TASK_DIR / "dataset.json").read_text(encoding="utf-8"))
    all_ids = sorted(
        Path(entry["image"]).name[: -len(".nii.gz")] for entry in dataset_json["training"]
    )
    calibration_ids = all_ids[:_N_CALIBRATION]
    test_ids = all_ids[_N_CALIBRATION : _N_CALIBRATION + _N_TEST]
    return {"train": [], "calibration": calibration_ids, "test": test_ids}


@pytest.fixture
def pipeline_inputs(tmp_path: Path):
    _skip_if_absent()
    manifest = _smoke_manifest()
    loader = MSDNiftiLoader(ROOT_DIR, TASK, split_manifest=manifest)
    cal_dataset = loader.load("calibration")
    test_dataset = loader.load("test")

    checkpoint_path = tmp_path / "smoke_checkpoint.pt"
    create_untrained_checkpoint(checkpoint_path, seed=0)

    score_provider = HippocampusScoreProvider(
        checkpoint_path,
        [cal_dataset, test_dataset],
        cache_dir=tmp_path / "score_cache",
    )
    return cal_dataset, test_dataset, score_provider


def test_dataset_to_score_provider_boundary(pipeline_inputs) -> None:
    """Every real id from both splits is servable by the shared ScoreProvider."""
    cal_dataset, test_dataset, score_provider = pipeline_inputs
    for dataset in (cal_dataset, test_dataset):
        for id_ in dataset.ids():
            score = score_provider.scores_for(id_)
            assert score.dtype == np.bool_
            assert score.shape == dataset.labels(id_).shape


def test_score_provider_to_loss_table_builder_boundary(pipeline_inputs) -> None:
    """Real scores/labels satisfy MorphologicalSets/FNRLoss's own shape+dtype checks."""
    cal_dataset, _test_dataset, score_provider = pipeline_inputs
    constructor = MorphologicalSets()
    loss = FNRLoss()
    for id_ in cal_dataset.ids():
        score = score_provider.scores_for(id_)
        label = cal_dataset.labels(id_)
        predicted_set = constructor.construct(score, 2.0)
        value = loss.evaluate(predicted_set, label)
        assert 0.0 <= value <= loss.upper_bound()


def test_full_pipeline_executes_end_to_end(pipeline_inputs) -> None:
    """Dataset -> ScoreProvider -> LossTableBuilder -> calibration -> evaluation, real, no mocks."""
    cal_dataset, test_dataset, score_provider = pipeline_inputs

    constructor = MorphologicalSets()
    loss = FNRLoss()
    builder = LossTableBuilder()

    # --- LossTableBuilder boundary: prediction sets are constructed HERE,
    # once per (id, lambda), *before* calibration ever runs -- calibration
    # only ever sees the resulting scalar loss table, never a score or a
    # predicted set directly. This is the frozen architecture's own
    # ordering (wfcrc/datasets/loss_table_builder.py), not an MS7 choice.
    cal_loss_table = builder.build(cal_dataset, score_provider, constructor, loss, _LAMBDA_GRID)
    test_loss_table = builder.build(test_dataset, score_provider, constructor, loss, _LAMBDA_GRID)

    assert cal_loss_table.shape == (len(cal_dataset), len(_LAMBDA_GRID))
    assert test_loss_table.shape == (len(test_dataset), len(_LAMBDA_GRID))
    assert np.all(np.isfinite(cal_loss_table.values))
    assert np.all(np.isfinite(test_loss_table.values))

    # --- LossTableBuilder -> Calibration boundary: WFCRCCalibrator itself
    # asserts loss_table.lambda_grid == cfg.lambda_grid; both loss tables
    # were built from the same _LAMBDA_GRID, so this holds by construction.
    family = CVaRFamily(beta=0.2)
    cfg = CalibrationConfig(alpha=0.5, B=loss.upper_bound(), pi=0.5, lambda_grid=_LAMBDA_GRID)
    verifier = Verifier()

    # --- Calibration -> Evaluation boundary: run_experiment composes
    # calibration (on cal_loss_table) with metrics measured on
    # test_loss_table, which is held out from calibration entirely.
    report = run_experiment(cal_loss_table, test_loss_table, family, cfg, seed=0, verifier=verifier)

    # Smoke-experiment outputs (Task 6): calibration output.
    assert report.calibration.lambda_hat in _LAMBDA_GRID
    assert report.calibration.n_a is not None and report.calibration.n_b is not None
    assert report.calibration.n_a + report.calibration.n_b == len(cal_dataset)

    # Verification.
    assert report.verification is not None
    if not report.verification.passed:
        pytest.fail(f"verification failed: {report.verification}")

    # Evaluation metrics.
    assert "realized_marginal_risk" in report.metrics
    assert "effective_sizes" in report.metrics
    assert "realized_worst_case_risk" in report.metrics  # CVaR is a DualAmbiguityFamily
    assert "duality_gap" in report.metrics
    assert np.isfinite(report.metrics["realized_marginal_risk"])
    assert np.isfinite(report.metrics["realized_worst_case_risk"])

    print("\n--- MS7 smoke experiment result ---")
    print(f"lambda_hat = {report.calibration.lambda_hat}")
    print(f"n_a={report.calibration.n_a}, n_b={report.calibration.n_b}")
    print(f"realized_marginal_risk = {report.metrics['realized_marginal_risk']}")
    print(f"realized_worst_case_risk = {report.metrics['realized_worst_case_risk']}")
    print(f"verification passed = {report.verification.passed}")
    print(f"config_hash = {report.config_hash}")


def test_pipeline_is_deterministic_across_runs(pipeline_inputs) -> None:
    """Same checkpoint/seed/config -> byte-for-byte identical report.to_dict()."""
    cal_dataset, test_dataset, score_provider = pipeline_inputs
    constructor = MorphologicalSets()
    loss = FNRLoss()
    builder = LossTableBuilder()
    family = CVaRFamily(beta=0.2)
    cfg = CalibrationConfig(alpha=0.5, B=loss.upper_bound(), pi=0.5, lambda_grid=_LAMBDA_GRID)

    def run() -> dict:
        cal_lt = builder.build(cal_dataset, score_provider, constructor, loss, _LAMBDA_GRID)
        test_lt = builder.build(test_dataset, score_provider, constructor, loss, _LAMBDA_GRID)
        return run_experiment(cal_lt, test_lt, family, cfg, seed=0).to_dict()

    first = run()
    second = run()
    assert first == second
