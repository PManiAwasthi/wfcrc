"""Unit tests for :mod:`wfcrc.evaluation.verifier`.

Test names mirror the Algorithm Specification §20 verification checklist
items directly. Failure-branch tests use two techniques to craft
violations without touching any frozen module:

- ``dataclasses.replace`` on an otherwise-correct `CalibrationResult` to
  corrupt exactly one reported field (isolates checks that only compare
  `result`'s own fields against an independent recomputation).
- A tiny local ``_IdentityTransformFamily`` (a legitimate, if trivial,
  `DualAmbiguityFamily`: `c=0`, `t(z)=z`, so `transform(z)=z` and
  `btil(B)=B`) paired with a deliberately non-monotone or out-of-bound
  loss table, for the two checks (`g` monotonicity, `L_tilde<=B_tilde`)
  that are re-derived purely from `family`+`loss_table` and do not read
  `result` at all.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from tests.unit.evaluation._helpers import calibration_config, monotone_loss_table
from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.ambiguity.known_weight import KnownWeightFamily
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig
from wfcrc.evaluation.verifier import CheckResult, VerificationReport, Verifier
from wfcrc.exceptions import FamilyError, VerificationError

# ---------------------------------------------------------------------------
# CheckResult / VerificationReport
# ---------------------------------------------------------------------------


def test_report_passed_is_true_when_every_item_passes() -> None:
    report = VerificationReport(items=(CheckResult("a", True, "ok"), CheckResult("b", True, "ok")))
    assert report.passed is True


def test_report_passed_is_false_when_any_item_fails() -> None:
    report = VerificationReport(
        items=(CheckResult("a", True, "ok"), CheckResult("b", False, "bad"))
    )
    assert report.passed is False


def test_report_passed_is_true_for_an_empty_report() -> None:
    assert VerificationReport().passed is True


def test_assert_ok_does_not_raise_when_passed() -> None:
    VerificationReport(items=(CheckResult("a", True, "ok"),)).assert_ok()


def test_assert_ok_raises_verification_error_naming_failing_items() -> None:
    report = VerificationReport(
        items=(
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "bad"),
            CheckResult("c", False, "bad"),
        )
    )
    with pytest.raises(VerificationError, match="b, c"):
        report.assert_ok()


def test_merge_concatenates_items_from_both_reports() -> None:
    first = VerificationReport(items=(CheckResult("a", True, "ok"),))
    second = VerificationReport(items=(CheckResult("b", False, "bad"),))
    merged = first.merge(second)
    assert [item.name for item in merged.items] == ["a", "b"]
    assert merged.passed is False


# ---------------------------------------------------------------------------
# check_preconditions
# ---------------------------------------------------------------------------


def test_check_preconditions_passes_on_a_valid_table() -> None:
    table = monotone_loss_table()
    report = Verifier().check_preconditions(table, loss_bound=1.0)
    assert report.passed is True
    names = {item.name for item in report.items}
    assert names == {"p2_monotone_nonincreasing", "p2_bounded"}


def test_check_preconditions_single_lambda_column_is_trivially_monotone() -> None:
    values = np.array([[0.5], [0.2]])
    table = LossTable(values=values, lambda_grid=np.array([0.5]))
    report = Verifier().check_preconditions(table, loss_bound=1.0)
    assert report.passed is True


def test_check_preconditions_detects_a_non_monotone_row() -> None:
    values = np.array([[0.1, 0.9, 0.05]])
    table = LossTable(values=values, lambda_grid=np.array([0.0, 0.5, 1.0]))
    report = Verifier().check_preconditions(table, loss_bound=1.0)
    item = next(i for i in report.items if i.name == "p2_monotone_nonincreasing")
    assert item.passed is False
    assert report.passed is False


def test_check_preconditions_detects_an_out_of_bound_entry() -> None:
    values = np.array([[2.0, 1.0, 0.5]])
    table = LossTable(values=values, lambda_grid=np.array([0.0, 0.5, 1.0]))
    report = Verifier().check_preconditions(table, loss_bound=1.0)
    item = next(i for i in report.items if i.name == "p2_bounded")
    assert item.passed is False


# ---------------------------------------------------------------------------
# check_calibration: dual branch, happy path
# ---------------------------------------------------------------------------


def _calibrate_dual(table: LossTable, cfg: CalibrationConfig, seed: int = 0) -> CalibrationResult:
    return WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=seed)


def test_check_calibration_dual_branch_passes_on_a_genuine_result() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    report = Verifier().check_calibration(result, table, family, cfg, seed=0)
    assert report.passed is True
    names = {item.name for item in report.items}
    assert "reproducibility" in names
    assert "dual_split_disjoint_and_sizes_match" in names
    assert "dual_b_tilde_finite_and_matches" in names
    assert "dual_transform_bounded_by_b_tilde" in names
    assert "dual_g_monotone_nonincreasing" in names
    assert "dual_lambda_hat_is_argmin_or_empty_default" in names
    assert "finite_group_checks" in names  # not-applicable marker
    assert "known_weight_checks" in names  # not-applicable marker


def test_check_calibration_dual_branch_passes_when_empty_selection() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table, alpha=-1.0)  # infeasible target -> empty selection
    family = CVaRFamily(beta=0.2)
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.empty_flag is True
    report = Verifier().check_calibration(result, table, family, cfg, seed=0)
    assert report.passed is True


# ---------------------------------------------------------------------------
# check_calibration: dual branch, crafted violations via result corruption
# ---------------------------------------------------------------------------


def test_check_calibration_detects_wrong_lambda_hat() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    result = _calibrate_dual(table, cfg)
    other_lambda = next(float(lam) for lam in table.lambda_grid if lam != result.lambda_hat)
    corrupted = dataclasses.replace(result, lambda_hat=other_lambda)

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    assert report.passed is False
    repro = next(i for i in report.items if i.name == "reproducibility")
    assert repro.passed is False
    argmin = next(i for i in report.items if i.name == "dual_lambda_hat_is_argmin_or_empty_default")
    assert argmin.passed is False


def test_check_calibration_detects_wrong_n_a_n_b() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    result = _calibrate_dual(table, cfg)
    corrupted = dataclasses.replace(result, n_a=(result.n_a or 0) + 1)

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    split_item = next(i for i in report.items if i.name == "dual_split_disjoint_and_sizes_match")
    assert split_item.passed is False


def test_check_calibration_detects_wrong_b_tilde() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    result = _calibrate_dual(table, cfg)
    corrupted = dataclasses.replace(result, b_tilde=(result.b_tilde or 0.0) + 5.0)

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    item = next(i for i in report.items if i.name == "dual_b_tilde_finite_and_matches")
    assert item.passed is False


# ---------------------------------------------------------------------------
# check_calibration: dual branch, crafted violations via family/table manipulation
# ---------------------------------------------------------------------------


class _IdentityTransformFamily(DualAmbiguityFamily):
    """Trivial, legitimate `DualAmbiguityFamily`: `transform(z) = z`, `btil(B) = B`.

    Used only to isolate the two checks (`L_tilde<=B_tilde`, `g`
    monotonicity) that are re-derived purely from `family`+`loss_table`
    and never read `result` — everything downstream of `transform` then
    tracks the loss table's raw values directly, which the test
    constructs to deliberately violate the property under test.
    """

    @property
    def family_type(self) -> Any:
        return "cvar"

    def estimate_dual(self, losses_a_col: NDArray[np.float64]) -> float:
        return 0.0

    def c(self, theta: Any) -> float:
        del theta
        return 0.0

    def t(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
        del theta
        return np.asarray(z, dtype=np.float64)


def _dummy_dual_result(n_a: int, n_b: int) -> CalibrationResult:
    """A placeholder result for checks that never read `result`'s fields."""
    return CalibrationResult(
        lambda_hat=0.0, empty_flag=False, n_a=n_a, n_b=n_b, b_tilde=1.0, r_hat_b=0.0
    )


def test_check_calibration_detects_transform_exceeding_b_tilde() -> None:
    # Every row shares the same value at lambda=0 (2.0) that exceeds cfg.B=1.0.
    values = np.array([[2.0, 0.5], [2.0, 0.5]])
    table = LossTable(values=values, lambda_grid=np.array([0.0, 1.0]))
    cfg = CalibrationConfig(alpha=0.3, B=1.0, pi=0.5, lambda_grid=(0.0, 1.0))
    family = _IdentityTransformFamily()

    verifier = Verifier()
    report = verifier.check_calibration(_dummy_dual_result(1, 1), table, family, cfg, seed=0)
    item = next(i for i in report.items if i.name == "dual_transform_bounded_by_b_tilde")
    assert item.passed is False


def test_check_calibration_detects_non_monotone_g() -> None:
    # Every row shares the same non-monotone pattern across lambda, so any
    # A/B split sees the same (non-monotone) column means.
    pattern = [0.5, 0.9, 0.1]
    values = np.tile(pattern, (10, 1))
    table = LossTable(values=values, lambda_grid=np.array([0.0, 0.5, 1.0]))
    cfg = CalibrationConfig(alpha=0.3, B=1.0, pi=0.5, lambda_grid=(0.0, 0.5, 1.0))
    family = _IdentityTransformFamily()

    report = Verifier().check_calibration(_dummy_dual_result(5, 5), table, family, cfg, seed=0)
    item = next(i for i in report.items if i.name == "dual_g_monotone_nonincreasing")
    assert item.passed is False


# ---------------------------------------------------------------------------
# check_calibration: finite-group branch
# ---------------------------------------------------------------------------


def test_check_calibration_finite_group_branch_passes_on_a_genuine_result() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    family = FiniteGroupFamily(masks=[tuple(range(0, 10)), tuple(range(10, 20))])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    report = Verifier().check_calibration(result, table, family, cfg, seed=0)
    assert report.passed is True
    names = {item.name for item in report.items}
    assert "finite_group_lambda_hat_matches_max_over_groups" in names
    assert "finite_group_uses_n_g" in names
    assert "dual_branch_checks" in names  # not-applicable marker
    assert "known_weight_checks" in names  # not-applicable marker


def test_check_calibration_finite_group_branch_passes_when_empty_selection() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table, alpha=-1.0)  # infeasible target -> empty selection
    family = FiniteGroupFamily(masks=[tuple(range(0, 10)), tuple(range(10, 20))])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    report = Verifier().check_calibration(result, table, family, cfg, seed=0)
    assert report.passed is True


def test_check_calibration_finite_group_detects_wrong_lambda_hat() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    family = FiniteGroupFamily(masks=[tuple(range(0, 10)), tuple(range(10, 20))])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    other_lambda = next(float(lam) for lam in table.lambda_grid if lam != result.lambda_hat)
    corrupted = dataclasses.replace(result, lambda_hat=other_lambda)

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    check_name = "finite_group_lambda_hat_matches_max_over_groups"
    item = next(i for i in report.items if i.name == check_name)
    assert item.passed is False


def test_check_calibration_finite_group_detects_wrong_n_g_diagnostics() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    family = FiniteGroupFamily(masks=[tuple(range(0, 10)), tuple(range(10, 20))])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    corrupted = dataclasses.replace(result, diagnostics={"per_group": []})

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    item = next(i for i in report.items if i.name == "finite_group_uses_n_g")
    assert item.passed is False


def test_check_calibration_raises_family_error_for_malformed_finite_group_family() -> None:
    class _FakeFiniteGroup(AmbiguityFamily):
        @property
        def family_type(self) -> Any:
            return "finite_group"

    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    dummy_result = CalibrationResult(lambda_hat=0.0, empty_flag=False)
    with pytest.raises(FamilyError, match="groups"):
        Verifier().check_calibration(dummy_result, table, _FakeFiniteGroup(), cfg, seed=0)


# ---------------------------------------------------------------------------
# check_calibration: known-weight branch
# ---------------------------------------------------------------------------


def test_check_calibration_known_weight_branch_passes_on_a_genuine_result() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(20))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    report = Verifier().check_calibration(result, table, family, cfg, seed=0)
    assert report.passed is True
    names = {item.name for item in report.items}
    assert "known_weight_uses_full_n" in names
    assert "dual_branch_checks" in names
    assert "finite_group_checks" in names


def test_check_calibration_known_weight_branch_passes_when_empty_selection() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table, alpha=-1.0)  # infeasible target -> empty selection
    family = KnownWeightFamily(weights=np.ones(20))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    report = Verifier().check_calibration(result, table, family, cfg, seed=0)
    assert report.passed is True


def test_check_calibration_known_weight_detects_wrong_lambda_hat() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(20))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    other_lambda = next(float(lam) for lam in table.lambda_grid if lam != result.lambda_hat)
    corrupted = dataclasses.replace(result, lambda_hat=other_lambda)

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    item = next(i for i in report.items if i.name == "known_weight_uses_full_n")
    assert item.passed is False


def test_check_calibration_known_weight_detects_wrong_n_diagnostics() -> None:
    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(20))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    corrupted = dataclasses.replace(result, diagnostics={"n": -1, "weight_sum": 20.0})

    report = Verifier().check_calibration(corrupted, table, family, cfg, seed=0)
    item = next(i for i in report.items if i.name == "known_weight_uses_full_n")
    assert item.passed is False


def test_check_calibration_raises_family_error_for_malformed_known_weight_family() -> None:
    class _FakeKnownWeight(AmbiguityFamily):
        @property
        def family_type(self) -> Any:
            return "known_weight"

    table = monotone_loss_table(n=20)
    cfg = calibration_config(table)
    dummy_result = CalibrationResult(lambda_hat=0.0, empty_flag=False)
    with pytest.raises(FamilyError, match="weights"):
        Verifier().check_calibration(dummy_result, table, _FakeKnownWeight(), cfg, seed=0)


def test_check_calibration_raises_family_error_for_malformed_dual_family() -> None:
    class _FakeDual(AmbiguityFamily):
        @property
        def family_type(self) -> Any:
            return "cvar"

    table = monotone_loss_table()
    cfg = calibration_config(table)
    dummy_result = CalibrationResult(lambda_hat=0.0, empty_flag=False)
    with pytest.raises(FamilyError, match="DualAmbiguityFamily"):
        Verifier().check_calibration(dummy_result, table, _FakeDual(), cfg, seed=0)


# ---------------------------------------------------------------------------
# tol constructor parameter
# ---------------------------------------------------------------------------


def test_custom_tol_is_used_by_check_preconditions() -> None:
    # A row that increases by exactly 0.05 should fail at tight tol but pass at loose tol.
    values = np.array([[0.5, 0.55]])
    table = LossTable(values=values, lambda_grid=np.array([0.0, 1.0]))
    tight = Verifier(tol=1e-9).check_preconditions(table, loss_bound=1.0)
    loose = Verifier(tol=0.1).check_preconditions(table, loss_bound=1.0)
    assert tight.passed is False
    assert loose.passed is True
