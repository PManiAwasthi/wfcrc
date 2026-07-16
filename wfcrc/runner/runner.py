"""``ExperimentRunner`` — config-driven orchestration with checkpointing, sweeps, and resume.

Per the Implementation Blueprint (§6, §12) and the MS5 Implementation
Specification (C2, M15): orchestrates `calibrate -> verify -> metrics ->
plot -> write manifest`, enforcing the verify STOP-gate before any metric
is exposed, with resumable checkpoints and isolated, seeded sweep cells.

**Scope decision (confirmed with the user before implementation — see
`CLAIMS_TRACEABILITY.md`).** The Blueprint's own stage list is `load config
-> build/load loss tables (cal+test) -> calibrate -> verify -> metrics ->
plot -> write manifest`. Building loss tables from `config.data`/
`config.model`/`config.sets`/`config.loss` requires resolving those name
strings to concrete `DatasetLoader`/`ScoreProvider`/`PredictionSetConstructor`/
`LossEvaluator` instances; no such dataset/model registry exists anywhere in
this repository (`wfcrc.datasets` is ABC-contracts-only per the MS4 scope
decision — no real dataset/model is available in this environment).
`ExperimentRunner.run` therefore takes already-built `cal_loss_table`/
`test_loss_table` `LossTable` objects directly (exactly like
`wfcrc.evaluation.experiment.run_experiment` already does), and resolves
only `config.family` to a concrete `AmbiguityFamily` — via the already-frozen
`wfcrc.ambiguity.FAMILIES` registry (MS2), not a new one. This keeps the
runner's own scope entirely inside already-built primitives; the
`config`-driven dataset stage remains future work once a real dataset/model
is on hand.

**Verify STOP-gate.** `run_experiment` (MS4, frozen) always computes metrics
and attaches a `VerificationReport` — it does not itself gate metric
exposure on verification passing (nothing in its own scope requires that).
The gate is this module's own responsibility: `run()` inspects the report's
`VerificationReport.passed` *before* checkpointing or returning anything,
raising `VerificationError` if it failed. Metrics are computed internally by
`run_experiment` as an unavoidable consequence of reusing it wholesale
(composing frozen code rather than reimplementing calibrate+verify+metrics
here), but a failed gate means they are never checkpointed, never written
to the manifest, and never returned to the caller — "no downstream metrics"
in effect, even though they were computed once, internally, and discarded.

**Figures.** Most of the paper's F1-F8 figures (Experiment Blueprint §26)
aggregate *many* calibration runs (risk vs alpha, vs severity, vs group) and
have no canonical single-run form; they are downstream of a sweep's
collected metrics, not this module's concern. The one figure genuinely
tied to a *single* calibration run is `plot_g_curve` (dual families only):
`run()` recomputes `g(lambda)` across the whole grid — mirroring exactly
the frozen dual-branch computation `WFCRCCalibrator`/`Verifier` already
perform (same family API, same Algorithm Spec §7 steps 3-6 formula, applied
read-only for plotting; `lambda_hat` itself is never re-selected) — and
renders it via `wfcrc.visualization.plots.plot_g_curve`. Finite-group/
known-weight families have no g-curve concept in the frozen spec and
produce no figures.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from wfcrc.ambiguity import FAMILIES
from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.pipeline import VerifierLike
from wfcrc.calibration.splitter import Splitter
from wfcrc.config.loader import load_config
from wfcrc.config.schema import CalibrationConfig, Config, FamilyConfig
from wfcrc.constants import TEXT_ENCODING
from wfcrc.evaluation.experiment import run_experiment
from wfcrc.evaluation.verifier import VerificationReport, Verifier
from wfcrc.exceptions import FamilyError, RunnerError, VerificationError, WFCRCError
from wfcrc.runner.checkpointer import Checkpointer, stage_key
from wfcrc.utils.io import atomic_write, ensure_dir, load_json, save_json
from wfcrc.utils.logging import get_logger
from wfcrc.utils.reproducibility import get_environment_fingerprint, get_git_commit
from wfcrc.utils.seeds import derive_seed
from wfcrc.visualization.base import FigureFile, FigureSpec
from wfcrc.visualization.plots import plot_g_curve

__all__ = [
    "ExperimentRunner",
    "Manifest",
    "ResultBundle",
    "SweepCellFailure",
    "SweepConfig",
]


def _build_family(cfg: FamilyConfig) -> AmbiguityFamily:
    """Build a concrete `AmbiguityFamily` from a `FamilyConfig`, via the frozen `FAMILIES` registry.

    Args:
        cfg: The validated family configuration section.

    Returns:
        A fresh instance of the concrete family class `cfg.type` names.

    Raises:
        FamilyError: If `cfg.type` is not a key of `FAMILIES`, or its
            required parameter (per `wfcrc.config.schema.FAMILY_REQUIRED_FIELD`)
            is `None` (should not happen for a `FamilyConfig` produced by
            `load_config`, which already enforces this).
    """
    cls = FAMILIES.get(cfg.type)
    if cls is None:
        raise FamilyError(f"unsupported family type: {cfg.type!r}")
    if cfg.type == "cvar":
        if cfg.beta is None:
            raise FamilyError("family.type='cvar' requires family.beta")
        return cls(beta=cfg.beta)  # type: ignore[call-arg]
    if cfg.type == "kl":
        if cfg.rho is None:
            raise FamilyError("family.type='kl' requires family.rho")
        return cls(rho=cfg.rho)  # type: ignore[call-arg]
    if cfg.type == "finite_group":
        if cfg.masks is None:
            raise FamilyError("family.type='finite_group' requires family.masks")
        return cls(masks=cfg.masks)  # type: ignore[call-arg]
    if cfg.weights is None:
        raise FamilyError("family.type='known_weight' requires family.weights")
    return cls(weights=cfg.weights)  # type: ignore[call-arg]


def _dual_g_curve(
    loss_table: LossTable, family: DualAmbiguityFamily, cfg: CalibrationConfig, seed: int
) -> tuple[list[float], list[float]]:
    """Recompute `g(lambda)`, for :func:`~wfcrc.visualization.plots.plot_g_curve`.

    Read-only: mirrors exactly the frozen dual-branch computation
    `WFCRCCalibrator._calibrate_dual`/`Verifier._check_dual` already
    perform (Algorithm Spec §7 steps 3-6) — `CalibrationResult` itself only
    exposes `g` at the deployed `lambda_hat`, not the whole curve, and
    neither of those methods is a public API this module may call
    directly. Never re-selects a threshold: `lambda_hat` is already fixed
    by the `CalibrationResult` this curve accompanies.

    Args:
        loss_table: The calibration `LossTable` (same one calibration ran on).
        family: The dual ambiguity family (`cvar`/`kl`) calibration used.
        cfg: The calibration config calibration used.
        seed: The seed calibration used (reproduces the same A/B split).

    Returns:
        `(lambda_grid, g_values)`, both plain `list[float]`, same length.
    """
    n = loss_table.shape[0]
    a_idx, b_idx = Splitter().split(n, cfg.pi, seed)
    n_b = len(b_idx)
    lambda_grid = loss_table.lambda_grid
    theta_by_lambda = {
        float(lam): family.estimate_dual(loss_table.column(float(lam))[a_idx])
        for lam in lambda_grid
    }
    b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in lambda_grid)
    g_values: list[float] = []
    for lam in lambda_grid:
        theta = theta_by_lambda[float(lam)]
        l_tilde = family.transform(loss_table.column(float(lam))[b_idx], theta)
        r_hat = float(np.mean(l_tilde))
        g_values.append((n_b / (n_b + 1)) * r_hat + b_tilde / (n_b + 1))
    return [float(lam) for lam in lambda_grid], g_values


def _persist_run_inputs(
    run_dir: Path,
    config: Config,
    cal_loss_table: LossTable,
    test_loss_table: LossTable,
    groups: Sequence[Sequence[int]] | None,
    make_figures: bool,
) -> None:
    """Persist everything :meth:`ExperimentRunner.resume` needs to rehydrate this run.

    Args:
        run_dir: The run directory (already created).
        config: The run configuration.
        cal_loss_table: The calibration `LossTable`.
        test_loss_table: The test `LossTable`.
        groups: Optional per-group row indices, or `None`.
        make_figures: Whether figure generation was requested.
    """
    cal_loss_table.save(run_dir / "cal_loss_table.json")
    test_loss_table.save(run_dir / "test_loss_table.json")
    atomic_write(run_dir / "config.yaml", config.to_yaml().encode(TEXT_ENCODING))
    if groups is not None:
        save_json(run_dir / "groups.json", [list(g) for g in groups])
    save_json(run_dir / "run_options.json", {"make_figures": make_figures})


@dataclasses.dataclass(frozen=True)
class Manifest:
    """The JSON-serializable run manifest written to `<run_dir>/manifest.json` (IB §17).

    Attributes:
        config_hash: Reproducibility content hash (config + seed + family).
        seed: The base seed this run was computed with.
        family_type: The ambiguity family type used.
        family_params: A plain snapshot of the family's parameters.
        git_commit: The current git commit hash, or `None` if unavailable.
        environment: Interpreter/platform/package version fingerprint.
        n_a: Size of calibration block `A` (dual branch only).
        n_b: Size of calibration block `B` (dual branch only).
        b_tilde: The transformed-loss bound (dual branch only).
        r_hat_b: The uninflated worst-case empirical risk at `lambda_hat`
            (dual branch only).
        lambda_hat: The selected threshold.
        empty_flag: Whether the empty-selection fallback fired.
        diagnostics: Branch-specific extra detail (see `CalibrationResult`).
        verification_passed: Whether every verification check passed.
            Always `True` for a manifest that was actually written, since a
            failing gate halts `run()` before this manifest is built (see
            module docstring); recorded explicitly anyway so the manifest
            is self-describing.
        metrics: The realized-risk/efficiency metrics computed on the test
            `LossTable`.
        figure_paths: `{figure_name: file_path}` for every figure rendered.
    """

    config_hash: str
    seed: int
    family_type: str
    family_params: Mapping[str, Any]
    git_commit: str | None
    environment: Mapping[str, str]
    n_a: int | None
    n_b: int | None
    b_tilde: float | None
    r_hat_b: float | None
    lambda_hat: float
    empty_flag: bool
    diagnostics: Mapping[str, Any]
    verification_passed: bool | None
    metrics: Mapping[str, Any]
    figure_paths: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Render this manifest as a plain, JSON-serializable dict."""
        return {
            "config_hash": self.config_hash,
            "seed": self.seed,
            "family_type": self.family_type,
            "family_params": dict(self.family_params),
            "git_commit": self.git_commit,
            "environment": dict(self.environment),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "b_tilde": self.b_tilde,
            "r_hat_b": self.r_hat_b,
            "lambda_hat": self.lambda_hat,
            "empty_flag": self.empty_flag,
            "diagnostics": dict(self.diagnostics),
            "verification_passed": self.verification_passed,
            "metrics": dict(self.metrics),
            "figure_paths": dict(self.figure_paths),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Reconstruct a `Manifest` from :meth:`to_dict`'s output.

        Args:
            data: A mapping with exactly `to_dict`'s keys.

        Returns:
            The reconstructed `Manifest`.

        Raises:
            TypeError: If `data` is missing a required key or has an
                unexpected one.
        """
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class ResultBundle:
    """Everything produced by one :meth:`ExperimentRunner.run` (or `resume`) call.

    Attributes:
        run_dir: The run directory this bundle was produced in.
        manifest: The `Manifest` written to `<run_dir>/manifest.json`.
        metrics: The same metrics dict as `manifest.metrics` (as live
            Python objects, not yet round-tripped through JSON).
        figures: `{figure_name: FigureFile}` for every figure rendered.
        verification: The live `VerificationReport`, if this call actually
            ran verification (a fresh, non-checkpointed computation); `None`
            if this result came from an existing checkpoint (verification
            already ran on the original computation, and always passed —
            see `Manifest.verification_passed` — since a failing gate is
            never checkpointed in the first place).
    """

    run_dir: Path
    manifest: Manifest
    metrics: Mapping[str, Any]
    figures: Mapping[str, FigureFile]
    verification: VerificationReport | None


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    """A grid over `alpha` / ambiguity-family parameters / seed (Implementation Blueprint §12).

    Dataset is not a sweep dimension in this milestone's runner (see module
    docstring): `cal_loss_table`/`test_loss_table` are fixed across every
    cell of a given :meth:`ExperimentRunner.run_sweep` call.

    Attributes:
        alphas: Target risk levels to sweep; defaults to
            `[base_config.calibration.alpha]` if `None`.
        family_param_grid: Per-cell `dataclasses.replace` overrides for
            `base_config.family` (e.g. `[{"beta": 0.1}, {"beta": 0.3}]`);
            defaults to `[{}]` (no override) if `None`.
        seeds: Base seed values to sweep; defaults to `[base_config.seed]`
            if `None`. Each cell's *actual* seed is derived from
            `(cell_index, seed)` (see :meth:`ExperimentRunner.run_sweep`),
            so cells never share an identical split even if two cells name
            the same raw seed.
    """

    alphas: Sequence[float] | None = None
    family_param_grid: Sequence[Mapping[str, Any]] | None = None
    seeds: Sequence[int] | None = None


@dataclasses.dataclass(frozen=True)
class SweepCellFailure:
    """A recorded (not raised) failure for one sweep cell (MS5 spec C2 item 8: "record + continue").

    Attributes:
        index: The cell's position in the sweep's Cartesian-product iteration order.
        run_dir: The cell's (still-created) isolated run directory.
        params: `{"alpha": ..., "family_overrides": ..., "seed": ...}` for this cell.
        error: `str(exception)` describing why this cell failed.
    """

    index: int
    run_dir: Path
    params: Mapping[str, Any]
    error: str


class ExperimentRunner:
    """Config-driven orchestration: calibrate -> verify -> metrics -> plot -> manifest.

    Composes already-frozen components rather than reimplementing them:
    :func:`wfcrc.evaluation.experiment.run_experiment` (MS4) for
    calibrate+verify+metrics, :mod:`wfcrc.visualization.plots` (this
    milestone) for the one single-run figure, and
    :class:`~wfcrc.runner.checkpointer.Checkpointer` (wrapping the frozen
    `wfcrc.utils.cache.Cache`, MS1) for resumable stage boundaries.
    """

    def __init__(self, *, verifier: VerifierLike | None = None) -> None:
        """Initialize the runner.

        Args:
            verifier: The verifier enforcing the STOP-gate; defaults to a
                fresh `wfcrc.evaluation.verifier.Verifier()`. Unlike
                `run_experiment`'s own optional `verifier` parameter, the
                runner always verifies — there would be nothing to gate on
                otherwise.
        """
        self._verifier: VerifierLike = verifier if verifier is not None else Verifier()

    def run(
        self,
        config: Config,
        cal_loss_table: LossTable,
        test_loss_table: LossTable,
        *,
        run_dir: str | Path,
        groups: Sequence[Sequence[int]] | None = None,
        make_figures: bool = True,
        figure_spec: FigureSpec | None = None,
        force_recompute: bool = False,
    ) -> ResultBundle:
        """Run one config-driven experiment end to end.

        Stages: resolve `config.family` -> calibrate+verify+metrics (via
        `run_experiment`, checkpointed as `"experiment"`) -> the `g`-curve
        figure for dual families (checkpointed as `"figures"`) -> write
        `manifest.json`. A failing verify STOP-gate raises
        `VerificationError` before either checkpoint is written and before
        any metric is returned.

        Args:
            config: The run configuration (`config.family`/`.calibration`/
                `.seed`/`.runner` are read; `.data`/`.model`/`.sets`/`.loss`
                are recorded in the persisted config for provenance but not
                resolved to concrete objects — see module docstring).
            cal_loss_table: The calibration `LossTable`.
            test_loss_table: The held-out test `LossTable`.
            run_dir: Destination run directory (created if missing).
            groups: Optional per-group row indices into `test_loss_table`,
                forwarded to `run_experiment` for `per_group_risk`.
            make_figures: Whether to render the `g`-curve figure (dual
                families only; ignored for finite-group/known-weight).
            figure_spec: Format/dpi/figsize for the figure, if rendered.
            force_recompute: Bypass every checkpoint and recompute from
                scratch, overwriting existing entries.

        Returns:
            The resulting `ResultBundle`.

        Raises:
            VerificationError: If any verification check failed (the STOP-gate).
            FamilyError: If `config.family.type` cannot be resolved to a
                concrete `AmbiguityFamily`.
            ValueError: Propagated from `run_experiment` (e.g. mismatched `lambda_grid`).
        """
        run_dir = ensure_dir(run_dir)
        logger = get_logger(run_dir, config.runner.log_level)
        checkpointer = Checkpointer(run_dir, force_recompute=force_recompute)

        _persist_run_inputs(run_dir, config, cal_loss_table, test_loss_table, groups, make_figures)

        family = _build_family(config.family)
        logger.info(
            "run_start", config_hash=config.hash(), family_type=family.family_type, seed=config.seed
        )

        live_verification: list[VerificationReport | None] = [None]

        def _compute_experiment() -> dict[str, Any]:
            report = run_experiment(
                cal_loss_table,
                test_loss_table,
                family,
                config.calibration,
                seed=config.seed,
                verifier=self._verifier,
                groups=groups,
            )
            # `self._verifier` is always a real verifier (see __init__), so
            # `run_experiment(..., verifier=self._verifier, ...)` always
            # returns a populated `verification`; this assert is for mypy's
            # type narrowing only, not a runtime guard against reachable
            # behavior (same pattern as `Verifier.check_calibration`'s own
            # asserts, `CLAIMS_TRACEABILITY.md` §6).
            assert report.verification is not None
            live_verification[0] = report.verification
            if not report.verification.passed:
                failing = [item.name for item in report.verification.items if not item.passed]
                logger.error("verify_gate_failed", failing_checks=failing)
                raise VerificationError(
                    f"verify STOP-gate failed for run_dir={run_dir}: {', '.join(failing)}"
                )
            logger.info(
                "calibration_result",
                lambda_hat=report.calibration.lambda_hat,
                n_a=report.calibration.n_a,
                n_b=report.calibration.n_b,
                b_tilde=report.calibration.b_tilde,
                empty_flag=report.calibration.empty_flag,
            )
            return {
                "config_hash": report.config_hash,
                "seed": report.seed,
                "family_type": report.family_type,
                "family_params": report.family_params,
                "n_a": report.calibration.n_a,
                "n_b": report.calibration.n_b,
                "b_tilde": report.calibration.b_tilde,
                "r_hat_b": report.calibration.r_hat_b,
                "lambda_hat": report.calibration.lambda_hat,
                "empty_flag": report.calibration.empty_flag,
                "diagnostics": dict(report.calibration.diagnostics),
                "verification_passed": report.verification.passed,
                "metrics": report.metrics,
            }

        experiment_key = stage_key(
            "experiment",
            config.hash(),
            cal_loss_table.values,
            cal_loss_table.lambda_grid,
            test_loss_table.values,
            test_loss_table.lambda_grid,
            None if groups is None else [list(g) for g in groups],
        )
        experiment_state = checkpointer.get_or_compute(experiment_key, _compute_experiment)

        figures = self._render_figures(
            checkpointer,
            experiment_key,
            experiment_state,
            cal_loss_table,
            family,
            config,
            run_dir,
            make_figures,
            figure_spec,
            logger,
        )

        manifest = Manifest(
            config_hash=experiment_state["config_hash"],
            seed=experiment_state["seed"],
            family_type=experiment_state["family_type"],
            family_params=experiment_state["family_params"],
            git_commit=get_git_commit(),
            environment=get_environment_fingerprint(),
            n_a=experiment_state["n_a"],
            n_b=experiment_state["n_b"],
            b_tilde=experiment_state["b_tilde"],
            r_hat_b=experiment_state["r_hat_b"],
            lambda_hat=experiment_state["lambda_hat"],
            empty_flag=experiment_state["empty_flag"],
            diagnostics=experiment_state["diagnostics"],
            verification_passed=experiment_state["verification_passed"],
            metrics=experiment_state["metrics"],
            figure_paths={name: str(figure.path) for name, figure in figures.items()},
        )
        save_json(run_dir / "manifest.json", manifest.to_dict())
        logger.info("manifest_written", path=str(run_dir / "manifest.json"))
        logger.close()

        return ResultBundle(
            run_dir=run_dir,
            manifest=manifest,
            metrics=experiment_state["metrics"],
            figures=figures,
            verification=live_verification[0],
        )

    @staticmethod
    def _render_figures(
        checkpointer: Checkpointer,
        experiment_key: str,
        experiment_state: Mapping[str, Any],
        cal_loss_table: LossTable,
        family: AmbiguityFamily,
        config: Config,
        run_dir: Path,
        make_figures: bool,
        figure_spec: FigureSpec | None,
        logger: Any,
    ) -> dict[str, FigureFile]:
        """Render (or reload the checkpoint of) the single-run `g`-curve figure.

        Returns an empty dict for finite-group/known-weight families (no
        g-curve concept in the frozen spec) or when `make_figures` is `False`.
        """
        if not make_figures or not isinstance(family, DualAmbiguityFamily):
            return {}

        resolved_spec = figure_spec if figure_spec is not None else FigureSpec()
        figures_dir = run_dir / "figures"
        figures_key = stage_key("figures", experiment_key, resolved_spec.format)

        def _compute_figures() -> dict[str, dict[str, str]]:
            lambda_grid, g_values = _dual_g_curve(
                cal_loss_table, family, config.calibration, config.seed
            )
            figure_file = plot_g_curve(
                lambda_grid,
                g_values,
                config.calibration.alpha,
                experiment_state["lambda_hat"],
                figures_dir / "g_curve",
                spec=resolved_spec,
            )
            logger.info("figure_written", name="g_curve", path=str(figure_file.path))
            return {
                "g_curve": {
                    "path": str(figure_file.path),
                    "sidecar_path": str(figure_file.sidecar_path),
                    "source_hash": figure_file.source_hash,
                }
            }

        figure_state = checkpointer.get_or_compute(figures_key, _compute_figures)
        return {
            name: FigureFile(
                path=Path(entry["path"]),
                sidecar_path=Path(entry["sidecar_path"]),
                source_hash=entry["source_hash"],
            )
            for name, entry in figure_state.items()
        }

    def run_sweep(
        self,
        base_config: Config,
        cal_loss_table: LossTable,
        test_loss_table: LossTable,
        sweep_config: SweepConfig,
        *,
        run_dir: str | Path,
        groups: Sequence[Sequence[int]] | None = None,
        make_figures: bool = True,
        figure_spec: FigureSpec | None = None,
    ) -> list[ResultBundle | SweepCellFailure]:
        """Run every cell of `sweep_config`'s Cartesian-product grid as an isolated sub-run.

        Each cell gets its own subdirectory (`<run_dir>/cell_<index>_<hash>`)
        and a seed derived from `(cell_index, raw_seed)` — distinct even if
        two cells name the same raw seed value in `sweep_config.seeds` — so
        cells are never accidentally correlated. A cell that raises a
        `wfcrc.exceptions.WFCRCError` or `ValueError` (e.g. a failed verify
        STOP-gate, or an invalid swept `alpha`) is recorded as a
        `SweepCellFailure` at that position rather than aborting the sweep
        (MS5 spec C2 item 8: "sweep-cell failure -> record + continue").

        Args:
            base_config: The configuration every cell overrides `alpha`/
                `family`-params/`seed` on.
            cal_loss_table: The calibration `LossTable`, fixed across every cell.
            test_loss_table: The test `LossTable`, fixed across every cell.
            sweep_config: The grid to sweep.
            run_dir: Root directory for the sweep; each cell gets a subdirectory.
            groups: Forwarded to every cell's `run()` call.
            make_figures: Forwarded to every cell's `run()` call.
            figure_spec: Forwarded to every cell's `run()` call.

        Returns:
            One entry per grid cell, in Cartesian-product order: a
            `ResultBundle` on success, or a `SweepCellFailure` if that cell
            raised.
        """
        run_dir = ensure_dir(run_dir)
        alphas = (
            sweep_config.alphas
            if sweep_config.alphas is not None
            else [base_config.calibration.alpha]
        )
        family_grid = (
            sweep_config.family_param_grid if sweep_config.family_param_grid is not None else [{}]
        )
        seeds = sweep_config.seeds if sweep_config.seeds is not None else [base_config.seed]

        results: list[ResultBundle | SweepCellFailure] = []
        grid = product(alphas, family_grid, seeds)
        for index, (alpha, family_overrides, seed) in enumerate(grid):
            params = {"alpha": alpha, "family_overrides": dict(family_overrides), "seed": seed}
            cell_hash = stage_key("sweep_cell", index, params)
            cell_dir = run_dir / f"cell_{index:04d}_{cell_hash[:12]}"
            try:
                if not (0.0 < alpha < base_config.calibration.B):
                    raise RunnerError(
                        f"sweep cell {index}: alpha={alpha} must satisfy "
                        f"0 < alpha < B={base_config.calibration.B}"
                    )
                cell_calibration = dataclasses.replace(base_config.calibration, alpha=alpha)
                cell_family = dataclasses.replace(base_config.family, **family_overrides)
                derived_seed = derive_seed(f"runner.sweep.cell.{index}", seed)
                cell_config = dataclasses.replace(
                    base_config,
                    calibration=cell_calibration,
                    family=cell_family,
                    seed=derived_seed,
                )
                bundle = self.run(
                    cell_config,
                    cal_loss_table,
                    test_loss_table,
                    run_dir=cell_dir,
                    groups=groups,
                    make_figures=make_figures,
                    figure_spec=figure_spec,
                )
                results.append(bundle)
            except (WFCRCError, ValueError) as exc:
                results.append(
                    SweepCellFailure(index=index, run_dir=cell_dir, params=params, error=str(exc))
                )
        return results

    def resume(self, run_dir: str | Path) -> ResultBundle:
        """Resume a run from `run_dir`, restarting from its last completed stage.

        Rehydrates the exact inputs a previous `run()` call persisted
        (config, calibration/test loss tables, groups, figure options) and
        calls `run()` again with `force_recompute=False`; already-completed
        stages are skipped via their existing checkpoints (see
        `Checkpointer.get_or_compute`), and any remaining stage completes.

        Args:
            run_dir: A directory previously passed to `run()` as `run_dir`.

        Returns:
            The resulting `ResultBundle`.

        Raises:
            RunnerError: If `run_dir` has no resumable run (missing
                `config.yaml` or either persisted loss table).
        """
        run_dir = Path(run_dir)
        config_path = run_dir / "config.yaml"
        cal_path = run_dir / "cal_loss_table.json"
        test_path = run_dir / "test_loss_table.json"
        if not (config_path.exists() and cal_path.exists() and test_path.exists()):
            raise RunnerError(
                f"'{run_dir}' has no resumable run (missing config.yaml/"
                "cal_loss_table.json/test_loss_table.json)"
            )

        config = load_config([config_path])
        cal_loss_table = LossTable.load(cal_path)
        test_loss_table = LossTable.load(test_path)

        groups_path = run_dir / "groups.json"
        groups: Sequence[Sequence[int]] | None = (
            load_json(groups_path) if groups_path.exists() else None
        )
        options_path = run_dir / "run_options.json"
        options = load_json(options_path) if options_path.exists() else {"make_figures": True}

        return self.run(
            config,
            cal_loss_table,
            test_loss_table,
            run_dir=run_dir,
            groups=groups,
            make_figures=bool(options.get("make_figures", True)),
            force_recompute=False,
        )
