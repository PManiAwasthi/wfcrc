"""Unit tests for ``scripts/reproduce.py`` (`make reproduce`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts import reproduce


class TestBuildReferenceInputs:
    def test_deterministic(self) -> None:
        config_a, cal_a, test_a = reproduce.build_reference_inputs()
        config_b, cal_b, test_b = reproduce.build_reference_inputs()

        assert config_a.hash() == config_b.hash()
        assert (cal_a.values == cal_b.values).all()
        assert (test_a.values == test_b.values).all()

    def test_seed_matches_reference_seed(self) -> None:
        config, _, _ = reproduce.build_reference_inputs()
        assert config.seed == reproduce.REFERENCE_SEED


class TestCompareToGolden:
    def _base(self) -> dict[str, object]:
        return {
            "config_hash": "abc",
            "family_type": "cvar",
            "empty_flag": False,
            "verification_passed": True,
            "lambda_hat": 0.5,
            "n_a": 10,
            "n_b": 10,
            "b_tilde": 1.0,
            "r_hat_b": 0.2,
            "metrics": {"realized_marginal_risk": 0.1, "effective_sizes": {"n_a": 10.0}},
        }

    def test_identical_matches(self) -> None:
        fresh = self._base()
        golden = self._base()
        assert reproduce.compare_to_golden(fresh, golden) == []

    def test_within_tolerance_matches(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["lambda_hat"] = golden["lambda_hat"] + 1e-12  # type: ignore[operator]
        assert reproduce.compare_to_golden(fresh, golden) == []

    def test_exact_key_mismatch_is_reported(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["family_type"] = "kl"
        mismatches = reproduce.compare_to_golden(fresh, golden)
        assert any("family_type" in m for m in mismatches)

    def test_numeric_key_mismatch_is_reported(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["lambda_hat"] = 0.9
        mismatches = reproduce.compare_to_golden(fresh, golden)
        assert any("lambda_hat" in m for m in mismatches)

    def test_numeric_key_none_vs_value_is_reported(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["n_a"] = None
        mismatches = reproduce.compare_to_golden(fresh, golden)
        assert any("n_a" in m for m in mismatches)

    def test_numeric_key_both_none_matches(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["n_a"] = None
        golden["n_a"] = None
        assert reproduce.compare_to_golden(fresh, golden) == []

    def test_nested_metrics_dict_mismatch_is_reported(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["metrics"]["effective_sizes"] = {"n_a": 99.0}  # type: ignore[index]
        mismatches = reproduce.compare_to_golden(fresh, golden)
        assert any("effective_sizes" in m for m in mismatches)

    def test_nested_metrics_dict_match(self) -> None:
        fresh = self._base()
        golden = self._base()
        assert reproduce.compare_to_golden(fresh, golden) == []

    def test_scalar_metric_mismatch_is_reported(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["metrics"]["realized_marginal_risk"] = 0.9  # type: ignore[index]
        mismatches = reproduce.compare_to_golden(fresh, golden)
        assert any("realized_marginal_risk" in m for m in mismatches)

    def test_metric_present_only_in_one_side_is_reported(self) -> None:
        fresh = self._base()
        golden = self._base()
        fresh["metrics"]["extra"] = 0.5  # type: ignore[index]
        mismatches = reproduce.compare_to_golden(fresh, golden)
        assert any("extra" in m for m in mismatches)


class TestMainCli:
    def test_compares_against_committed_golden_file(self) -> None:
        assert reproduce.GOLDEN_PATH.exists(), "committed golden file is required for this test"
        exit_code = reproduce.main([])
        assert exit_code == 0

    def test_write_golden_then_compare_roundtrips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden_copy = tmp_path / "golden.json"
        monkeypatch.setattr(reproduce, "GOLDEN_PATH", golden_copy)

        assert reproduce.main(["--write-golden"]) == 0
        assert golden_copy.exists()
        assert reproduce.main([]) == 0

    def test_missing_golden_file_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(reproduce, "GOLDEN_PATH", tmp_path / "missing.json")
        assert reproduce.main([]) == 1

    def test_mismatch_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden_copy = tmp_path / "golden.json"
        monkeypatch.setattr(reproduce, "GOLDEN_PATH", golden_copy)
        assert reproduce.main(["--write-golden"]) == 0

        from wfcrc.utils.io import load_json, save_json

        tampered = load_json(golden_copy)
        tampered["lambda_hat"] = tampered["lambda_hat"] + 10.0
        save_json(golden_copy, tampered)

        assert reproduce.main([]) == 1

    def test_cli_subprocess_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/reproduce.py"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
