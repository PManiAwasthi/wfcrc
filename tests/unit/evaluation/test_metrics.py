"""Unit tests for :mod:`wfcrc.evaluation.metrics`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wfcrc.ambiguity.base import AmbiguityFamily
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.evaluation.metrics import (
    CI,
    TestResult,
    bootstrap_ci,
    coverage,
    duality_gap,
    effective_sizes,
    holm_correct,
    mean_set_size,
    one_sided_risk_test,
    paired_wilcoxon,
    per_group_risk,
    realized_marginal_risk,
    realized_worst_case_risk,
)
from wfcrc.exceptions import FamilyError
from wfcrc.prediction_sets.classification import ThresholdSets

# ---------------------------------------------------------------------------
# realized_worst_case_risk / realized_marginal_risk
# ---------------------------------------------------------------------------


def _test_table() -> LossTable:
    lambda_grid = np.array([0.0, 0.5, 1.0])
    values = np.array(
        [
            [0.9, 0.5, 0.1],
            [0.8, 0.4, 0.2],
            [1.0, 0.6, 0.0],
            [0.7, 0.3, 0.1],
        ]
    )
    return LossTable(values=values, lambda_grid=lambda_grid)


def test_realized_worst_case_risk_matches_manual_dual_computation() -> None:
    table = _test_table()
    family = CVaRFamily(beta=0.2)
    result = CalibrationResult(lambda_hat=0.5, empty_flag=False)

    risk = realized_worst_case_risk(result, table, family)

    col = table.column(0.5)
    theta = family.estimate_dual(col)
    expected = float(np.mean(family.transform(col, theta)))
    assert risk == pytest.approx(expected)


def test_realized_worst_case_risk_rejects_non_dual_family() -> None:
    table = _test_table()
    family = FiniteGroupFamily(masks=[[0, 1], [2, 3]])
    result = CalibrationResult(lambda_hat=0.5, empty_flag=False)
    with pytest.raises(FamilyError, match="dual families"):
        realized_worst_case_risk(result, table, family)


def test_realized_marginal_risk_is_the_plain_column_mean() -> None:
    table = _test_table()
    result = CalibrationResult(lambda_hat=1.0, empty_flag=False)
    assert realized_marginal_risk(result, table) == pytest.approx(np.mean([0.1, 0.2, 0.0, 0.1]))


# ---------------------------------------------------------------------------
# per_group_risk
# ---------------------------------------------------------------------------


def test_per_group_risk_computes_mean_per_group() -> None:
    table = _test_table()
    result = CalibrationResult(lambda_hat=0.0, empty_flag=False)
    groups = [[0, 1], [2, 3]]
    output = per_group_risk(result, table, groups)
    assert output == {0: pytest.approx(np.mean([0.9, 0.8])), 1: pytest.approx(np.mean([1.0, 0.7]))}


def test_per_group_risk_rejects_empty_groups_list() -> None:
    table = _test_table()
    result = CalibrationResult(lambda_hat=0.0, empty_flag=False)
    with pytest.raises(ValueError, match="groups must be non-empty"):
        per_group_risk(result, table, [])


def test_per_group_risk_rejects_an_empty_group() -> None:
    table = _test_table()
    result = CalibrationResult(lambda_hat=0.0, empty_flag=False)
    with pytest.raises(ValueError, match="group 1 is empty"):
        per_group_risk(result, table, [[0, 1], []])


# ---------------------------------------------------------------------------
# mean_set_size / coverage
# ---------------------------------------------------------------------------


def test_mean_set_size_matches_manual_cardinality_average() -> None:
    constructor = ThresholdSets()
    scores = [np.array([0.9, 0.1, 0.1]), np.array([0.9, 0.9, 0.1])]
    lam = 0.5  # threshold = 0.5
    result = mean_set_size(constructor, scores, lam)
    expected = np.mean([np.sum(constructor.construct(s, lam)) for s in scores])
    assert result == pytest.approx(expected)


def test_mean_set_size_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="scores must be non-empty"):
        mean_set_size(ThresholdSets(), [], 0.5)


def test_coverage_matches_manual_miscoverage_complement() -> None:
    constructor = ThresholdSets()
    scores = [np.array([0.9, 0.1]), np.array([0.1, 0.1])]
    labels = [np.array([True, False]), np.array([True, False])]
    lam = 0.2  # threshold = 0.8: first score covers, second does not
    result = coverage(constructor, scores, labels, lam)
    assert result == pytest.approx(0.5)


def test_coverage_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="scores must be non-empty"):
        coverage(ThresholdSets(), [], [], 0.5)


def test_coverage_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"scores has .* entries but labels has"):
        coverage(ThresholdSets(), [np.array([0.5])], [], 0.5)


# ---------------------------------------------------------------------------
# effective_sizes
# ---------------------------------------------------------------------------


def test_effective_sizes_reads_dual_branch_fields() -> None:
    result = CalibrationResult(lambda_hat=0.5, empty_flag=False, n_a=10, n_b=20)
    assert effective_sizes(result) == {"n_a": 10.0, "n_b": 20.0}


def test_effective_sizes_reads_per_group_diagnostics() -> None:
    result = CalibrationResult(
        lambda_hat=0.5,
        empty_flag=False,
        diagnostics={"per_group": [{"group": 0, "n_g": 5}, {"group": 1, "n_g": 7}]},
    )
    output = effective_sizes(result)
    assert output == {"n_g_0": 5.0, "n_g_1": 7.0}


def test_effective_sizes_computes_kish_n_eff_from_weights() -> None:
    result = CalibrationResult(lambda_hat=0.5, empty_flag=False)
    weights = np.array([1.0, 1.0, 1.0, 1.0])  # uniform weights -> n_eff == n
    output = effective_sizes(result, weights=weights)
    assert output["n_eff"] == pytest.approx(4.0)


def test_effective_sizes_returns_empty_dict_when_nothing_applies() -> None:
    result = CalibrationResult(lambda_hat=0.5, empty_flag=False)
    assert effective_sizes(result) == {}


# ---------------------------------------------------------------------------
# duality_gap
# ---------------------------------------------------------------------------


def test_duality_gap_is_surrogate_minus_realized() -> None:
    assert duality_gap(0.3, 0.1) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_the_sample_mean() -> None:
    values = np.linspace(0.0, 1.0, 50)
    ci = bootstrap_ci(values, seed=0)
    assert isinstance(ci, CI)
    assert ci.lo <= np.mean(values) <= ci.hi
    assert ci.level == 0.95


def test_bootstrap_ci_is_deterministic_given_the_same_seed() -> None:
    values = [0.1, 0.2, 0.15, 0.3, 0.25]
    first = bootstrap_ci(values, seed=42)
    second = bootstrap_ci(values, seed=42)
    assert first == second


def test_bootstrap_ci_degenerates_to_a_point_for_a_single_value() -> None:
    ci = bootstrap_ci([0.5], seed=0, n_resamples=10)
    assert ci.lo == pytest.approx(0.5)
    assert ci.hi == pytest.approx(0.5)


def test_bootstrap_ci_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="values must be non-empty"):
        bootstrap_ci([], seed=0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_ci_rejects_level_outside_open_unit_interval(level: float) -> None:
    with pytest.raises(ValueError, match="level must be in"):
        bootstrap_ci([0.1, 0.2], level=level, seed=0)


def test_bootstrap_ci_rejects_non_positive_n_resamples() -> None:
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        bootstrap_ci([0.1, 0.2], seed=0, n_resamples=0)


# ---------------------------------------------------------------------------
# one_sided_risk_test
# ---------------------------------------------------------------------------


def test_one_sided_risk_test_large_p_value_when_mean_is_below_alpha() -> None:
    risks = [0.05, 0.06, 0.04, 0.05, 0.055]
    result = one_sided_risk_test(risks, alpha=0.5)
    assert isinstance(result, TestResult)
    assert result.p_value > 0.9


def test_one_sided_risk_test_small_p_value_when_mean_is_far_above_alpha() -> None:
    risks = [0.9, 0.91, 0.89, 0.92, 0.895]
    result = one_sided_risk_test(risks, alpha=0.1)
    assert result.p_value < 0.05


def test_one_sided_risk_test_matches_normal_cdf_by_hand() -> None:
    risks = [0.1, 0.2, 0.3, 0.4]
    alpha = 0.15
    result = one_sided_risk_test(risks, alpha)
    std = np.std(risks, ddof=1)
    z = (np.mean(risks) - alpha) / (std / math.sqrt(len(risks)))
    expected_p = 1.0 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert result.statistic == pytest.approx(z)
    assert result.p_value == pytest.approx(expected_p)


def test_one_sided_risk_test_rejects_fewer_than_two_entries() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        one_sided_risk_test([0.1], alpha=0.2)


def test_one_sided_risk_test_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        one_sided_risk_test([0.2, 0.2, 0.2], alpha=0.2)


# ---------------------------------------------------------------------------
# paired_wilcoxon
# ---------------------------------------------------------------------------


def test_paired_wilcoxon_symmetric_around_zero_gives_small_statistic() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=30)
    b = a.copy()  # identical -> all zero differences discarded is a different case;
    b = a + rng.normal(0, 1e-9, size=30)  # negligible, near-symmetric noise
    result = paired_wilcoxon(a.tolist(), b.tolist())
    assert isinstance(result, TestResult)
    assert result.p_value > 0.05


def test_paired_wilcoxon_detects_a_systematic_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=30)
    b = a + 0.5  # b systematically larger
    result = paired_wilcoxon(a.tolist(), b.tolist())
    assert result.p_value < 0.01
    assert result.statistic < 0.0  # W+ (positive a-b ranks) is small


def test_paired_wilcoxon_handles_ties_via_average_ranks() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.0, 1.0, 3.0, 4.0, 3.0]  # |diff| has a tie (2.0, 2.0)
    result = paired_wilcoxon(a, b)
    assert isinstance(result.statistic, float)


def test_paired_wilcoxon_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        paired_wilcoxon([1.0, 2.0], [1.0])


def test_paired_wilcoxon_rejects_all_zero_differences() -> None:
    with pytest.raises(ValueError, match="no nonzero differences"):
        paired_wilcoxon([1.0, 2.0], [1.0, 2.0])


def test_paired_wilcoxon_handles_a_single_nonzero_difference() -> None:
    # n=1: var_w = 1*2*3/24 - 0/48 = 0.25 > 0 (the formula's variance is
    # provably always positive -- see the `assert` in the source, which
    # this exercises at its smallest possible n).
    result = paired_wilcoxon([5.0], [1.0])
    assert result.statistic != 0.0


# ---------------------------------------------------------------------------
# holm_correct
# ---------------------------------------------------------------------------


def test_holm_correct_matches_hand_computed_example() -> None:
    pvals = [0.01, 0.2, 0.03, 0.5]
    corrected = holm_correct(pvals)
    assert corrected == pytest.approx([0.04, 0.4, 0.09, 0.5])


def test_holm_correct_is_monotone_nondecreasing_in_sorted_order() -> None:
    pvals = [0.5, 0.001, 0.3, 0.2, 0.4]
    corrected = holm_correct(pvals)
    order = np.argsort(pvals)
    sorted_corrected = np.asarray(corrected)[order]
    assert np.all(np.diff(sorted_corrected) >= -1e-12)


def test_holm_correct_clips_at_one() -> None:
    corrected = holm_correct([0.9, 0.9, 0.9])
    assert all(p <= 1.0 for p in corrected)


def test_holm_correct_single_pvalue_is_unchanged() -> None:
    assert holm_correct([0.03]) == pytest.approx([0.03])


def test_holm_correct_rejects_empty_pvals() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        holm_correct([])


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_holm_correct_rejects_pvals_outside_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_correct([0.1, bad])


# ---------------------------------------------------------------------------
# isinstance guard uses the base AmbiguityFamily type correctly
# ---------------------------------------------------------------------------


def test_realized_worst_case_risk_type_annotation_accepts_ambiguity_family_base() -> None:
    # Static-typing smoke check: AmbiguityFamily is accepted at the type
    # level even though only DualAmbiguityFamily instances pass at runtime.
    family: AmbiguityFamily = CVaRFamily(beta=0.2)
    table = _test_table()
    result = CalibrationResult(lambda_hat=0.5, empty_flag=False)
    assert isinstance(realized_worst_case_risk(result, table, family), float)
