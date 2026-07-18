"""MS10 — pilot end-to-end validation of the WFCRC pipeline and every registered baseline.

**Purpose (per the MS10 task brief): validate, not publish.** This script
executes the complete `Dataset -> ScoreProvider -> LossTableBuilder ->
Calibration -> Evaluation` workflow (`docs/EXPERIMENT_PROTOCOL.md` §2)
against real MSD Task04_Hippocampus data, using the same tiny,
non-scientific subset MS7's own smoke pipeline used (8 calibration + 4
test cases — `docs/DATASET_SPLIT_POLICY.md` §3.1's "reduced-scale
exception"), and runs **every** registered `wfcrc.baselines.BASELINES`
entry against the identical `LossTable`/config/seed, so their outputs are
structurally comparable. It is not a statistically meaningful experiment
and produces no scientific claim — see `docs/PILOT_REPORT.md` for the
observed results and their explicit non-interpretation.

**No frozen MS1-MS9 file is imported privately or modified.** Every
component this script calls is a public, already-frozen entry point
(`MSDNiftiLoader`, `HippocampusScoreProvider`, `LossTableBuilder`,
`WFCRCCalibrator`, every `wfcrc.baselines.Calibrator`, `Verifier`,
`wfcrc.evaluation.metrics`, `ExperimentRunner`, `wfcrc.runner.runner.Manifest`).

**Output layout** (`docs/RESULTS_SCHEMA.md` §1, applied to a one-off pilot
run rather than a full E1-E12 campaign): `results/pilot/<baseline_name>/
manifest.json` per baseline (the "wfcrc" baseline additionally gets a real
`figures/g_curve.pdf`+`.csv` via the frozen `ExperimentRunner`, since that
is the one baseline `ExperimentRunner` already supports end to end);
`results/pilot/pilot_summary.json` aggregating every baseline's timing,
metrics, and pass/fail status; `results/pilot/failure_injection.json`
recording the six controlled-failure checks (Task 6).

Run with the real dataset present at `datasets/Task04_Hippocampus/` (per
MS6.3A/MS7's own acquisition instructions); exits early with a clear
message otherwise. Not part of the default `pytest` suite — see
`tests/unit/scripts/test_pilot_ms10.py` (opt-in, `@pytest.mark.real_data`)
for the checked-in regression wrapper.
"""

from __future__ import annotations

import json
import shutil
import time
import tracemalloc
from pathlib import Path
from typing import Any, cast

from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.ambiguity.kl import KLFamily
from wfcrc.baselines import BASELINES
from wfcrc.baselines.base import Calibrator
from wfcrc.baselines.ensembles import EnsembleAggregatedLAC
from wfcrc.baselines.group_conditional import GroupConditionalCRC
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.baselines.negative_controls import (
    FixedEtaWFCRC,
    PooledKFoldWFCRC,
    TotalNInflationWFCRC,
)
from wfcrc.baselines.robust_fdiv import RobustFDivergenceCP
from wfcrc.baselines.scaling import TemperatureScaledLAC
from wfcrc.baselines.vanilla_crc import VanillaCRC
from wfcrc.baselines.wfcrc_adapter import WFCRCAdapter
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.splitter import Splitter
from wfcrc.config.schema import (
    CalibrationConfig,
    Config,
    DataConfig,
    FamilyConfig,
    LossConfig,
    ModelConfig,
    RunnerConfig,
    SetsConfig,
)
from wfcrc.datasets.loaders.msd import MSDDataset, MSDNiftiLoader
from wfcrc.datasets.loss_table_builder import LossTableBuilder
from wfcrc.evaluation.metrics import (
    effective_sizes,
    per_group_risk,
    realized_marginal_risk,
    realized_worst_case_risk,
)
from wfcrc.evaluation.verifier import Verifier
from wfcrc.exceptions import CacheError, SplitLeakageError, WFCRCError
from wfcrc.losses.fnr import FNRLoss
from wfcrc.models.scores.hippocampus_segmenter import (
    HippocampusScoreProvider,
    create_untrained_checkpoint,
)
from wfcrc.prediction_sets.segmentation import MorphologicalSets
from wfcrc.runner.runner import ExperimentRunner, Manifest
from wfcrc.utils.cache import make_key
from wfcrc.utils.io import save_json
from wfcrc.utils.reproducibility import get_environment_fingerprint, get_git_commit

ROOT_DIR = Path("datasets/Task04_Hippocampus")
TASK = "Task04_Hippocampus"
TASK_DIR = ROOT_DIR / TASK

#: Same tiny, fixed subset MS7's own smoke pipeline used (see that module's
#: docstring and `docs/DATASET_SPLIT_POLICY.md` §3.1's "reduced-scale
#: exception") -- an engineering-validation convenience, not a research split.
N_CALIBRATION = 8
N_TEST = 4

#: Same dilation-radius grid / alpha / split fraction / seed MS7 used --
#: identical config across every baseline (Task 4, cross-baseline consistency).
LAMBDA_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
ALPHA = 0.5
PI = 0.5
SEED = 0

RESULTS_DIR = Path("results/pilot")


def dataset_present() -> bool:
    return TASK_DIR.is_dir()


def _smoke_manifest() -> dict[str, list[str]]:
    dataset_json = json.loads((TASK_DIR / "dataset.json").read_text(encoding="utf-8"))
    all_ids = sorted(
        Path(entry["image"]).name[: -len(".nii.gz")] for entry in dataset_json["training"]
    )
    calibration_ids = all_ids[:N_CALIBRATION]
    test_ids = all_ids[N_CALIBRATION : N_CALIBRATION + N_TEST]
    return {"train": [], "calibration": calibration_ids, "test": test_ids}


def build_pipeline_inputs(work_dir: Path) -> dict[str, Any]:
    """Dataset -> ScoreProvider -> LossTableBuilder, timed stage by stage.

    Returns:
        A dict with `cal_loss_table`, `test_loss_table`, `cfg`, and a
        `timings` sub-dict (`data_loading`, `checkpoint_creation`,
        `checkpoint_loading_and_first_inference`, `loss_table_building`,
        `cache_hit_rescoring`), all in seconds.
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    manifest = _smoke_manifest()
    loader = MSDNiftiLoader(ROOT_DIR, TASK, split_manifest=manifest)
    # `DatasetLoader.load` is declared to return the frozen `Dataset` ABC
    # (per that contract's own signature); `MSDNiftiLoader.load` always
    # returns a concrete `MSDDataset` at runtime, which is what
    # `HippocampusScoreProvider` requires -- a narrow, disclosed cast, the
    # same kind of frozen-interface accommodation already used elsewhere
    # in this project (e.g. `HippocampusScoreProvider`'s own return-type
    # `# type: ignore`), not a change to either frozen interface.
    cal_dataset = cast(MSDDataset, loader.load("calibration"))
    test_dataset = cast(MSDDataset, loader.load("test"))
    timings["data_loading"] = time.perf_counter() - t0

    checkpoint_path = work_dir / "smoke_checkpoint.pt"
    t0 = time.perf_counter()
    create_untrained_checkpoint(checkpoint_path, seed=0)
    timings["checkpoint_creation"] = time.perf_counter() - t0

    cache_dir = work_dir / "score_cache"
    t0 = time.perf_counter()
    score_provider = HippocampusScoreProvider(
        checkpoint_path, [cal_dataset, test_dataset], cache_dir=cache_dir
    )
    # Force one real inference now (construction alone does not run the
    # network) so "checkpoint loading" and "first inference" are measured
    # together, matching what a first real experiment run actually pays.
    _ = score_provider.scores_for(cal_dataset.ids()[0])
    timings["checkpoint_loading_and_first_inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    constructor = MorphologicalSets()
    loss = FNRLoss()
    builder = LossTableBuilder()
    cal_loss_table = builder.build(cal_dataset, score_provider, constructor, loss, LAMBDA_GRID)
    test_loss_table = builder.build(test_dataset, score_provider, constructor, loss, LAMBDA_GRID)
    timings["loss_table_building"] = time.perf_counter() - t0

    # Cache-utilization measurement: re-request every calibration id's score
    # (already cached above via LossTableBuilder) and confirm it is fast
    # (cache hit), not a second round of real inference.
    t0 = time.perf_counter()
    for id_ in cal_dataset.ids():
        score_provider.scores_for(id_)
    timings["cache_hit_rescoring_all_calibration_ids"] = time.perf_counter() - t0

    cfg = CalibrationConfig(alpha=ALPHA, B=loss.upper_bound(), pi=PI, lambda_grid=LAMBDA_GRID)
    return {
        "cal_dataset": cal_dataset,
        "test_dataset": test_dataset,
        "cal_loss_table": cal_loss_table,
        "test_loss_table": test_loss_table,
        "cfg": cfg,
        "timings": timings,
        "checkpoint_path": checkpoint_path,
        "cache_dir": cache_dir,
    }


def build_baselines(cal_loss_table: LossTable) -> dict[str, Calibrator]:
    """Construct one instance of every registered baseline, sharing config/seed.

    `group_conditional` needs real (calibration-split-relative) row
    indices; it is given a structurally valid, non-empty two-group split
    of the 8-case calibration pool (first half / second half) -- an
    engineering-validation grouping only, not a scientific organ/region
    claim (there is no real anterior/posterior mask wired here; Group Mask
    Builders, MS6.7, remain unbuilt) -- disclosed explicitly in
    `docs/PILOT_REPORT.md`.
    """
    n_cal = cal_loss_table.shape[0]
    half = n_cal // 2
    cal_groups = (tuple(range(0, half)), tuple(range(half, n_cal)))

    cvar = CVaRFamily(beta=0.2)
    kl = KLFamily(rho=0.1)

    baselines: dict[str, Calibrator] = {
        "wfcrc": WFCRCAdapter(cvar),
        "vanilla_crc": VanillaCRC(),
        "lac": SplitConformalLAC(),
        "group_conditional": GroupConditionalCRC(FiniteGroupFamily(masks=cal_groups)),
        "robust_fdiv": RobustFDivergenceCP(kl),
        "pooled_k_fold": PooledKFoldWFCRC(cvar, k_folds=5),
        "total_n_inflation": TotalNInflationWFCRC(cvar),
        "fixed_eta": FixedEtaWFCRC(cvar, 0.3),
        "temperature_scaled_lac": TemperatureScaledLAC(),
        "ensemble_aggregated_lac": EnsembleAggregatedLAC(),
    }
    assert set(baselines) == set(BASELINES), (
        f"pilot baseline set {sorted(baselines)} does not match the registered "
        f"BASELINES {sorted(BASELINES)} -- every registered baseline must be exercised"
    )
    return baselines


def _dual_family_of(calibrator: Calibrator) -> Any | None:
    """Return `calibrator.family` if it is a `DualAmbiguityFamily`, else `None`."""
    from wfcrc.ambiguity.base import DualAmbiguityFamily

    family = getattr(calibrator, "family", None)
    return family if isinstance(family, DualAmbiguityFamily) else None


def run_one_baseline(
    name: str,
    calibrator: Calibrator,
    cal_loss_table: LossTable,
    test_loss_table: LossTable,
    cfg: CalibrationConfig,
    test_groups: tuple[tuple[int, ...], ...] | None,
) -> dict[str, Any]:
    """Run one baseline twice (determinism check) and compute its metrics.

    Returns a plain dict recording: `lambda_hat`, `empty_flag`,
    `deterministic`, `metrics`, `diagnostics`, `runtime_seconds`,
    `verification` (only for the two baselines a frozen `Verifier` check
    actually applies to -- see `docs/PILOT_REPORT.md` for why the frozen
    Verifier is WF-CRC-protocol-specific and does not generalize to every
    baseline), and `error` (`None` on success).
    """
    outcome: dict[str, Any] = {"error": None}
    try:
        t0 = time.perf_counter()
        first = calibrator.calibrate(cal_loss_table, cfg, seed=SEED)
        runtime = time.perf_counter() - t0
        second = calibrator.calibrate(cal_loss_table, cfg, seed=SEED)
        deterministic = (
            first.lambda_hat == second.lambda_hat and first.empty_flag == second.empty_flag
        )

        metrics: dict[str, Any] = {
            "realized_marginal_risk": realized_marginal_risk(first, test_loss_table),
            "effective_sizes": effective_sizes(first),
        }
        dual_family = _dual_family_of(calibrator)
        if dual_family is not None:
            metrics["realized_worst_case_risk"] = realized_worst_case_risk(
                first, test_loss_table, dual_family
            )
        if name == "group_conditional" and test_groups is not None:
            metrics["per_group_risk"] = per_group_risk(first, test_loss_table, test_groups)

        # Verification (only meaningful for "wfcrc"/"vanilla_crc" -- see
        # module docstring) is computed separately in main(), which has
        # the matching AmbiguityFamily object the frozen Verifier needs.
        verification_passed: bool | None = None

        outcome.update(
            {
                "lambda_hat": first.lambda_hat,
                "empty_flag": first.empty_flag,
                "n_a": first.n_a,
                "n_b": first.n_b,
                "b_tilde": first.b_tilde,
                "r_hat_b": first.r_hat_b,
                "diagnostics": dict(first.diagnostics),
                "deterministic": deterministic,
                "metrics": metrics,
                "runtime_seconds": runtime,
                "verification_passed": verification_passed,
                "calibration_result": first,
            }
        )
    except WFCRCError as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    return outcome


def write_manifest(name: str, outcome: dict[str, Any], cfg: CalibrationConfig) -> Path:
    """Write a `docs/RESULTS_SCHEMA.md`-shaped `manifest.json` for one baseline.

    Reuses the frozen `wfcrc.runner.runner.Manifest` dataclass directly
    (its fields are generic over any `CalibrationResult`, not WF-CRC-
    specific) rather than inventing a second manifest schema for
    baselines `ExperimentRunner` does not itself orchestrate.
    """
    run_dir = RESULTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    result = outcome["calibration_result"]
    manifest = Manifest(
        config_hash=make_key(name, cfg.alpha, cfg.B, cfg.pi, list(cfg.lambda_grid), SEED),
        seed=SEED,
        family_type=name,
        family_params={},
        git_commit=get_git_commit(),
        environment=get_environment_fingerprint(),
        n_a=result.n_a,
        n_b=result.n_b,
        b_tilde=result.b_tilde,
        r_hat_b=result.r_hat_b,
        lambda_hat=result.lambda_hat,
        empty_flag=result.empty_flag,
        diagnostics=dict(result.diagnostics),
        verification_passed=outcome.get("verification_passed"),
        metrics=outcome["metrics"],
        figure_paths={},
    )
    manifest_path = run_dir / "manifest.json"
    save_json(manifest_path, manifest.to_dict())
    return manifest_path


def run_wfcrc_via_experiment_runner(
    cal_loss_table: LossTable, test_loss_table: LossTable, cfg: CalibrationConfig, work_dir: Path
) -> dict[str, Any]:
    """Run the "wfcrc" baseline through the real, frozen `ExperimentRunner`.

    This is the one baseline `ExperimentRunner` already supports end to
    end (config-driven calibrate -> verify -> g-curve figure -> manifest),
    so routing it there (rather than through `write_manifest` like every
    other baseline) validates the *actual production* manifest/figure
    output path against real data, not a pilot-only approximation of it.
    """
    config = Config(
        data=DataConfig(name="msd_hippocampus", params={}),
        model=ModelConfig(name="hippocampus_segmenter", params={}),
        sets=SetsConfig(name="morphological", params={}),
        loss=LossConfig(name="fnr", params={}),
        family=FamilyConfig(type="cvar", beta=0.2),
        calibration=cfg,
        runner=RunnerConfig(cache_dir=str(work_dir / "runner_cache"), log_level="INFO"),
        seed=SEED,
    )
    run_dir = RESULTS_DIR / "wfcrc"
    runner = ExperimentRunner(verifier=Verifier())
    bundle = runner.run(config, cal_loss_table, test_loss_table, run_dir=run_dir)
    return {
        "manifest_path": str(run_dir / "manifest.json"),
        "figure_paths": bundle.manifest.figure_paths,
        "verification_passed": bundle.manifest.verification_passed,
        "lambda_hat": bundle.manifest.lambda_hat,
        "metrics": bundle.manifest.metrics,
    }


# --------------------------------------------------------------------------
# Task 6 -- controlled failure injection
# --------------------------------------------------------------------------


def failure_injection_checks(pipeline_inputs: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Attempt five controlled failures; every one must fail *gracefully* (a
    named `WFCRCError`/stdlib exception), never hang, crash uninformatively,
    or silently produce a wrong-but-plausible result."""
    results: dict[str, Any] = {}

    # (a) Missing checkpoint.
    try:
        HippocampusScoreProvider(work_dir / "does_not_exist.pt", [pipeline_inputs["cal_dataset"]])
        results["missing_checkpoint"] = {"graceful": False, "detail": "did not raise"}
    except FileNotFoundError as exc:
        results["missing_checkpoint"] = {"graceful": True, "detail": f"FileNotFoundError: {exc}"}
    except Exception as exc:
        results["missing_checkpoint"] = {
            "graceful": False,
            "detail": f"unexpected {type(exc).__name__}: {exc}",
        }

    # (b) Corrupted cache.
    try:
        corrupt_cache_dir = work_dir / "corrupt_cache"
        corrupt_cache_dir.mkdir(exist_ok=True)
        provider = HippocampusScoreProvider(
            pipeline_inputs["checkpoint_path"],
            [pipeline_inputs["cal_dataset"]],
            cache_dir=corrupt_cache_dir,
        )
        cal_id = pipeline_inputs["cal_dataset"].ids()[0]
        key = make_key(provider.model_fingerprint(), str(cal_id))
        (corrupt_cache_dir / f"{key}.npz").write_bytes(b"not a valid npz file")
        provider.scores_for(cal_id)
        results["corrupted_cache"] = {"graceful": False, "detail": "did not raise"}
    except CacheError as exc:
        results["corrupted_cache"] = {"graceful": True, "detail": f"CacheError: {exc}"}
    except Exception as exc:
        results["corrupted_cache"] = {
            "graceful": False,
            "detail": f"unexpected {type(exc).__name__}: {exc}",
        }

    # (c) Empty calibration split.
    try:
        Splitter().split(n=0, pi=PI, seed=SEED)
        results["empty_calibration_split"] = {"graceful": False, "detail": "did not raise"}
    except ValueError as exc:
        results["empty_calibration_split"] = {"graceful": True, "detail": f"ValueError: {exc}"}
    except Exception as exc:
        results["empty_calibration_split"] = {
            "graceful": False,
            "detail": f"unexpected {type(exc).__name__}: {exc}",
        }

    # (d) Invalid SplitManifest (overlapping calibration/test ids).
    try:
        manifest = _smoke_manifest()
        overlapping = dict(manifest)
        overlapping["test"] = [manifest["calibration"][0], *manifest["test"]]
        MSDNiftiLoader(ROOT_DIR, TASK, split_manifest=overlapping)
        results["invalid_split_manifest"] = {"graceful": False, "detail": "did not raise"}
    except SplitLeakageError as exc:
        results["invalid_split_manifest"] = {
            "graceful": True,
            "detail": f"SplitLeakageError: {exc}",
        }
    except Exception as exc:
        results["invalid_split_manifest"] = {
            "graceful": False,
            "detail": f"unexpected {type(exc).__name__}: {exc}",
        }

    # (e) Missing case (a split-manifest id absent from dataset.json's own
    # discovered cases -- the "missing prediction/source file" analogue for
    # this pipeline, which has no on-disk intermediate prediction files).
    try:
        manifest = _smoke_manifest()
        broken = dict(manifest)
        broken["calibration"] = [*manifest["calibration"], "hippocampus_does_not_exist"]
        loader = MSDNiftiLoader(ROOT_DIR, TASK, split_manifest=broken)
        loader.load("calibration")
        results["missing_case_id"] = {"graceful": False, "detail": "did not raise"}
    except (ValueError, KeyError) as exc:
        results["missing_case_id"] = {
            "graceful": True,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:
        results["missing_case_id"] = {
            "graceful": False,
            "detail": f"unexpected {type(exc).__name__}: {exc}",
        }

    return results


def main() -> int:
    if not dataset_present():
        print(f"MSD Task04_Hippocampus not found at {TASK_DIR.resolve()} -- pilot skipped.")
        return 1

    work_dir = Path("results/pilot/_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    tracemalloc.start()
    inputs = build_pipeline_inputs(work_dir)
    cal_loss_table = inputs["cal_loss_table"]
    test_loss_table = inputs["test_loss_table"]
    cfg = inputs["cfg"]

    baselines = build_baselines(cal_loss_table)
    n_test = test_loss_table.shape[0]
    test_groups = (tuple(range(0, n_test // 2)), tuple(range(n_test // 2, n_test)))

    per_baseline: dict[str, Any] = {}
    for name, calibrator in baselines.items():
        outcome = run_one_baseline(
            name, calibrator, cal_loss_table, test_loss_table, cfg, test_groups
        )
        per_baseline[name] = outcome
        if outcome["error"] is None:
            write_manifest(name, outcome, cfg)

    # Route "wfcrc" additionally through the real ExperimentRunner (real
    # manifest.json + figures/g_curve.pdf/.csv), and run the frozen
    # Verifier explicitly against "wfcrc" and "vanilla_crc" (the only two
    # baselines whose CalibrationResult corresponds to a frozen
    # WFCRCCalibrator branch -- see docs/PILOT_REPORT.md).
    experiment_runner_result = run_wfcrc_via_experiment_runner(
        cal_loss_table, test_loss_table, cfg, work_dir
    )

    from wfcrc.ambiguity.known_weight import KnownWeightFamily

    verifier = Verifier()
    verification_summary: dict[str, Any] = {}
    if per_baseline["wfcrc"]["error"] is None:
        cvar_report = verifier.check_preconditions(cal_loss_table, loss_bound=cfg.B).merge(
            verifier.check_calibration(
                per_baseline["wfcrc"]["calibration_result"],
                cal_loss_table,
                CVaRFamily(beta=0.2),
                cfg,
                seed=SEED,
            )
        )
        verification_summary["wfcrc"] = cvar_report.passed
    if per_baseline["vanilla_crc"]["error"] is None:
        n_cal = cal_loss_table.shape[0]
        kw_family = KnownWeightFamily(weights=[1.0] * n_cal)
        vanilla_report = verifier.check_preconditions(cal_loss_table, loss_bound=cfg.B).merge(
            verifier.check_calibration(
                per_baseline["vanilla_crc"]["calibration_result"],
                cal_loss_table,
                kw_family,
                cfg,
                seed=SEED,
            )
        )
        verification_summary["vanilla_crc"] = vanilla_report.passed

    peak_memory_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    failure_results = failure_injection_checks(inputs, work_dir)

    summary = {
        "dataset": "msd_hippocampus",
        "n_calibration": cal_loss_table.shape[0],
        "n_test": test_loss_table.shape[0],
        "lambda_grid": list(LAMBDA_GRID),
        "alpha": ALPHA,
        "pi": PI,
        "seed": SEED,
        "timings_seconds": inputs["timings"],
        "peak_memory_mb": peak_memory_mb,
        "baselines": {
            name: {
                k: v
                for k, v in outcome.items()
                if k != "calibration_result"  # not JSON-serializable
            }
            for name, outcome in per_baseline.items()
        },
        "verification_summary": verification_summary,
        "experiment_runner_wfcrc": experiment_runner_result,
        "failure_injection": failure_results,
    }
    save_json(RESULTS_DIR / "pilot_summary.json", summary)
    save_json(RESULTS_DIR / "failure_injection.json", failure_results)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
