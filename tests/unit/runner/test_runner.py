"""Unit tests for :mod:`wfcrc.runner.runner`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.runner._helpers import calibration_config, full_config, monotone_loss_table
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.ambiguity.known_weight import KnownWeightFamily
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import FamilyConfig
from wfcrc.evaluation import experiment as experiment_module
from wfcrc.evaluation.verifier import CheckResult, VerificationReport, Verifier
from wfcrc.exceptions import FamilyError, RunnerError, VerificationError
from wfcrc.runner.runner import (
    ExperimentRunner,
    Manifest,
    ResultBundle,
    SweepCellFailure,
    SweepConfig,
    _build_family,
    _dual_g_curve,
    _persist_run_inputs,
)
from wfcrc.visualization.base import FigureFile, FigureSpec


class _AlwaysFailVerifier:
    """A `VerifierLike` stub whose calibration check always fails."""

    def check_preconditions(
        self, loss_table: LossTable, *, loss_bound: float
    ) -> VerificationReport:
        return VerificationReport(
            items=(CheckResult(name="fake_precondition", passed=True, detail="ok"),)
        )

    def check_calibration(
        self, result: Any, loss_table: LossTable, family: Any, cfg: Any, *, seed: int
    ) -> VerificationReport:
        return VerificationReport(
            items=(CheckResult(name="fake_failure", passed=False, detail="forced failure"),)
        )


# ---------------------------------------------------------------------------
# _build_family
# ---------------------------------------------------------------------------


class TestBuildFamily:
    def test_cvar(self) -> None:
        family = _build_family(FamilyConfig(type="cvar", beta=0.2))
        assert isinstance(family, CVaRFamily)
        assert family.beta == 0.2

    def test_kl(self) -> None:
        family = _build_family(FamilyConfig(type="kl", rho=0.1))
        assert family.family_type == "kl"

    def test_finite_group(self) -> None:
        family = _build_family(FamilyConfig(type="finite_group", masks=((0, 1), (2, 3))))
        assert isinstance(family, FiniteGroupFamily)

    def test_known_weight(self) -> None:
        family = _build_family(FamilyConfig(type="known_weight", weights=(1.0, 1.0, 1.0)))
        assert isinstance(family, KnownWeightFamily)

    def test_unsupported_type_raises(self) -> None:
        bad_cfg = FamilyConfig(type="cvar", beta=0.2)
        object.__setattr__(bad_cfg, "type", "not_a_family")
        with pytest.raises(FamilyError, match="unsupported family type"):
            _build_family(bad_cfg)

    def test_cvar_missing_beta_raises(self) -> None:
        with pytest.raises(FamilyError, match=r"requires family\.beta"):
            _build_family(FamilyConfig(type="cvar", beta=None))

    def test_kl_missing_rho_raises(self) -> None:
        with pytest.raises(FamilyError, match=r"requires family\.rho"):
            _build_family(FamilyConfig(type="kl", rho=None))

    def test_finite_group_missing_masks_raises(self) -> None:
        with pytest.raises(FamilyError, match=r"requires family\.masks"):
            _build_family(FamilyConfig(type="finite_group", masks=None))

    def test_known_weight_missing_weights_raises(self) -> None:
        with pytest.raises(FamilyError, match=r"requires family\.weights"):
            _build_family(FamilyConfig(type="known_weight", weights=None))


# ---------------------------------------------------------------------------
# _dual_g_curve
# ---------------------------------------------------------------------------


class TestDualGCurve:
    def test_matches_manual_computation(self) -> None:
        table = monotone_loss_table(n=40, seed=0)
        family = CVaRFamily(beta=0.2)
        cfg = calibration_config(table)

        lambda_grid, g_values = _dual_g_curve(table, family, cfg, seed=0)

        from wfcrc.calibration.splitter import Splitter

        a_idx, b_idx = Splitter().split(table.shape[0], cfg.pi, 0)
        n_b = len(b_idx)
        theta_by_lambda = {
            float(lam): family.estimate_dual(table.column(float(lam))[a_idx])
            for lam in table.lambda_grid
        }
        b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in table.lambda_grid)
        expected = []
        for lam in table.lambda_grid:
            theta = theta_by_lambda[float(lam)]
            l_tilde = family.transform(table.column(float(lam))[b_idx], theta)
            r_hat = float(np.mean(l_tilde))
            expected.append((n_b / (n_b + 1)) * r_hat + b_tilde / (n_b + 1))

        assert lambda_grid == [float(x) for x in table.lambda_grid]
        assert g_values == pytest.approx(expected)

    def test_deterministic_given_same_seed(self) -> None:
        table = monotone_loss_table(n=40, seed=0)
        family = CVaRFamily(beta=0.2)
        cfg = calibration_config(table)
        first = _dual_g_curve(table, family, cfg, seed=5)
        second = _dual_g_curve(table, family, cfg, seed=5)
        assert first == second

    def test_g_is_monotone_nonincreasing(self) -> None:
        table = monotone_loss_table(n=40, seed=0)
        family = CVaRFamily(beta=0.2)
        cfg = calibration_config(table)
        _, g_values = _dual_g_curve(table, family, cfg, seed=0)
        assert all(g_values[i] >= g_values[i + 1] - 1e-9 for i in range(len(g_values) - 1))


# ---------------------------------------------------------------------------
# _persist_run_inputs
# ---------------------------------------------------------------------------


class TestPersistRunInputs:
    def test_writes_expected_files(self, tmp_path: Path) -> None:
        table = monotone_loss_table(n=20, seed=0)
        cfg = full_config(table, tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _persist_run_inputs(run_dir, cfg, table, table, None, True)

        assert (run_dir / "cal_loss_table.json").exists()
        assert (run_dir / "test_loss_table.json").exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "run_options.json").exists()
        assert not (run_dir / "groups.json").exists()

        options = json.loads((run_dir / "run_options.json").read_text())
        assert options == {"make_figures": True}

    def test_writes_groups_when_given(self, tmp_path: Path) -> None:
        table = monotone_loss_table(n=20, seed=0)
        cfg = full_config(table, tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _persist_run_inputs(run_dir, cfg, table, table, [[0, 1], [2, 3]], False)

        groups = json.loads((run_dir / "groups.json").read_text())
        assert groups == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def _sample(self) -> Manifest:
        return Manifest(
            config_hash="abc",
            seed=0,
            family_type="cvar",
            family_params={"beta": 0.2},
            git_commit=None,
            environment={"python_version": "3.12.0"},
            n_a=10,
            n_b=10,
            b_tilde=1.0,
            r_hat_b=0.1,
            lambda_hat=0.5,
            empty_flag=False,
            diagnostics={},
            verification_passed=True,
            metrics={"realized_marginal_risk": 0.1},
            figure_paths={"g_curve": "x.pdf"},
        )

    def test_to_dict_is_json_serializable(self) -> None:
        manifest = self._sample()
        json.dumps(manifest.to_dict())

    def test_from_dict_roundtrips(self) -> None:
        manifest = self._sample()
        restored = Manifest.from_dict(manifest.to_dict())
        assert restored == manifest


# ---------------------------------------------------------------------------
# ExperimentRunner.run
# ---------------------------------------------------------------------------


class TestRunFreshDualFamily:
    def test_produces_manifest_metrics_and_figure(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=60, seed=0)
        test = monotone_loss_table(n=40, seed=1)
        cfg = full_config(cal, tmp_path)
        run_dir = tmp_path / "run"

        runner = ExperimentRunner()
        bundle = runner.run(cfg, cal, test, run_dir=run_dir)

        assert isinstance(bundle, ResultBundle)
        assert bundle.run_dir == run_dir
        assert bundle.manifest.verification_passed is True
        assert "realized_marginal_risk" in bundle.metrics
        assert "g_curve" in bundle.figures
        assert isinstance(bundle.figures["g_curve"], FigureFile)
        assert bundle.figures["g_curve"].path.exists()
        assert bundle.verification is not None
        assert bundle.verification.passed is True

        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists()
        on_disk = json.loads(manifest_path.read_text())
        assert on_disk == bundle.manifest.to_dict()
        assert on_disk["figure_paths"]["g_curve"] == str(bundle.figures["g_curve"].path)

    def test_deterministic_across_separate_run_dirs(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=60, seed=0)
        test = monotone_loss_table(n=40, seed=1)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()

        first = runner.run(cfg, cal, test, run_dir=tmp_path / "a")
        second = runner.run(cfg, cal, test, run_dir=tmp_path / "b")

        assert first.manifest.lambda_hat == second.manifest.lambda_hat
        assert first.manifest.config_hash == second.manifest.config_hash
        assert first.metrics == second.metrics

    def test_make_figures_false_skips_figure(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()

        bundle = runner.run(cfg, cal, test, run_dir=tmp_path / "run", make_figures=False)

        assert bundle.figures == {}
        assert bundle.manifest.figure_paths == {}

    def test_custom_figure_spec_is_used(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()

        bundle = runner.run(
            cfg, cal, test, run_dir=tmp_path / "run", figure_spec=FigureSpec(format="svg")
        )
        assert bundle.figures["g_curve"].path.suffix == ".svg"

    def test_groups_forwarded_to_metrics(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()

        bundle = runner.run(
            cfg, cal, test, run_dir=tmp_path / "run", groups=[list(range(15)), list(range(15, 30))]
        )
        assert "per_group_risk" in bundle.metrics


class TestRunNonDualFamilies:
    def test_finite_group_produces_no_figures(self, tmp_path: Path) -> None:
        table = monotone_loss_table(n=20, seed=0)
        family_cfg = FamilyConfig(type="finite_group", masks=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)))
        cfg = full_config(table, tmp_path, family=family_cfg)
        runner = ExperimentRunner()

        bundle = runner.run(cfg, table, table, run_dir=tmp_path / "run")

        assert bundle.figures == {}
        assert bundle.manifest.figure_paths == {}
        assert bundle.manifest.n_a is None
        assert bundle.manifest.n_b is None

    def test_known_weight_produces_no_figures(self, tmp_path: Path) -> None:
        table = monotone_loss_table(n=10, seed=0)
        family_cfg = FamilyConfig(type="known_weight", weights=tuple(1.0 for _ in range(10)))
        cfg = full_config(table, tmp_path, family=family_cfg)
        runner = ExperimentRunner()

        bundle = runner.run(cfg, table, table, run_dir=tmp_path / "run")

        assert bundle.figures == {}


class TestVerifyStopGate:
    def test_failing_verification_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        run_dir = tmp_path / "run"
        runner = ExperimentRunner(verifier=_AlwaysFailVerifier())

        with pytest.raises(VerificationError):
            runner.run(cfg, cal, test, run_dir=run_dir)

        assert not (run_dir / "manifest.json").exists()
        assert not (run_dir / "checkpoints").exists() or not any(
            (run_dir / "checkpoints").iterdir()
        )

    def test_default_verifier_is_a_real_verifier(self) -> None:
        runner = ExperimentRunner()
        assert isinstance(runner._verifier, Verifier)


class TestResume:
    def test_resume_skips_recomputation_of_experiment_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        run_dir = tmp_path / "run"
        runner = ExperimentRunner()

        first = runner.run(cfg, cal, test, run_dir=run_dir)

        call_count = {"n": 0}
        original = experiment_module.run_experiment

        def counting_run_experiment(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(experiment_module, "run_experiment", counting_run_experiment)
        # The runner imported run_experiment by name into its own module
        # namespace, so patch it there too.
        monkeypatch.setattr("wfcrc.runner.runner.run_experiment", counting_run_experiment)

        resumed = runner.resume(run_dir)

        assert call_count["n"] == 0
        assert resumed.manifest.lambda_hat == first.manifest.lambda_hat
        assert resumed.manifest.config_hash == first.manifest.config_hash
        assert resumed.verification is None
        assert "g_curve" in resumed.figures

    def test_resume_raises_when_nothing_to_resume(self, tmp_path: Path) -> None:
        runner = ExperimentRunner()
        with pytest.raises(RunnerError, match="no resumable run"):
            runner.resume(tmp_path / "empty")

    def test_resume_restores_groups_and_options(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        run_dir = tmp_path / "run"
        runner = ExperimentRunner()

        runner.run(
            cfg,
            cal,
            test,
            run_dir=run_dir,
            groups=[list(range(15)), list(range(15, 30))],
            make_figures=False,
        )
        resumed = runner.resume(run_dir)

        assert "per_group_risk" in resumed.metrics
        assert resumed.figures == {}

    def test_resume_completes_a_previously_skipped_figures_stage(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        test = monotone_loss_table(n=30, seed=1)
        cfg = full_config(cal, tmp_path)
        run_dir = tmp_path / "run"
        runner = ExperimentRunner()

        first = runner.run(cfg, cal, test, run_dir=run_dir, make_figures=False)
        assert first.figures == {}

        # Simulate "figures were requested later": flip the persisted option
        # and resume, exactly like re-invoking `resume` after changing one's
        # mind, without recomputing the already-checkpointed experiment stage.
        options_path = run_dir / "run_options.json"
        options_path.write_text(json.dumps({"make_figures": True}), encoding="utf-8")

        resumed = runner.resume(run_dir)
        assert "g_curve" in resumed.figures
        assert resumed.manifest.lambda_hat == first.manifest.lambda_hat


# ---------------------------------------------------------------------------
# ExperimentRunner.run_sweep
# ---------------------------------------------------------------------------


class TestRunSweep:
    def test_sweeps_full_grid_with_isolated_dirs(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=60, seed=0)
        test = monotone_loss_table(n=40, seed=1)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()
        sweep = SweepConfig(alphas=[0.2, 0.4], family_param_grid=[{"beta": 0.1}, {"beta": 0.3}])

        results = runner.run_sweep(cfg, cal, test, sweep, run_dir=tmp_path / "sweep")

        assert len(results) == 4
        assert all(isinstance(r, ResultBundle) for r in results)
        run_dirs = {r.run_dir for r in results}
        assert len(run_dirs) == 4

    def test_default_grid_uses_base_config_values(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()
        sweep = SweepConfig()

        results = runner.run_sweep(cfg, cal, cal, sweep, run_dir=tmp_path / "sweep")

        assert len(results) == 1

    def test_seeds_are_derived_and_distinct_even_when_raw_seed_repeats(
        self, tmp_path: Path
    ) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()
        sweep = SweepConfig(alphas=[0.2, 0.4], seeds=[0])

        results = runner.run_sweep(cfg, cal, cal, sweep, run_dir=tmp_path / "sweep")

        assert len(results) == 2
        seeds_used = {r.manifest.seed for r in results}  # type: ignore[union-attr]
        assert len(seeds_used) == 2

    def test_invalid_alpha_is_recorded_not_raised(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()
        sweep = SweepConfig(alphas=[0.2, 5.0])  # 5.0 >= B=1.0, invalid

        results = runner.run_sweep(cfg, cal, cal, sweep, run_dir=tmp_path / "sweep")

        assert len(results) == 2
        failures = [r for r in results if isinstance(r, SweepCellFailure)]
        assert len(failures) == 1
        assert "alpha" in failures[0].error

    def test_verify_failure_cell_is_recorded_not_raised(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner(verifier=_AlwaysFailVerifier())
        sweep = SweepConfig(alphas=[0.2])

        results = runner.run_sweep(cfg, cal, cal, sweep, run_dir=tmp_path / "sweep")

        assert len(results) == 1
        assert isinstance(results[0], SweepCellFailure)

    def test_family_param_grid_actually_overrides(self, tmp_path: Path) -> None:
        cal = monotone_loss_table(n=40, seed=0)
        cfg = full_config(cal, tmp_path)
        runner = ExperimentRunner()
        sweep = SweepConfig(family_param_grid=[{"beta": 0.05}, {"beta": 0.5}])

        results = runner.run_sweep(cfg, cal, cal, sweep, run_dir=tmp_path / "sweep")

        betas = {r.manifest.family_params["beta"] for r in results}  # type: ignore[union-attr]
        assert betas == {0.05, 0.5}
