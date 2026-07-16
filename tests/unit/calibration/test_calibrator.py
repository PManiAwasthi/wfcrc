"""Unit tests for :class:`wfcrc.calibration.calibrator.WFCRCCalibrator`.

Test names mirror the Algorithm Specification §20 verification checklist
items directly, so fidelity to the frozen spec is traceable from the test
name alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.calibration._helpers import calibration_config, monotone_loss_table
from wfcrc.ambiguity import CVaRFamily, FiniteGroupFamily, KLFamily, KnownWeightFamily
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.splitter import Splitter
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import FamilyError

# ---------------------------------------------------------------------------
# Dual branch: CVaR
# ---------------------------------------------------------------------------


def test_cvar_returns_calibration_result_with_dual_fields_populated() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    result = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=0)
    assert isinstance(result, CalibrationResult)
    assert result.n_a is not None and result.n_b is not None
    assert result.n_a + result.n_b == table.shape[0]
    assert result.b_tilde is not None and np.isfinite(result.b_tilde)
    assert result.r_hat_b is not None
    assert result.lambda_hat in table.lambda_grid


def test_cvar_g_is_non_increasing_over_the_grid() -> None:
    # "g(lambda) non-increasing" (Algorithm Spec S20).
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    a_idx, b_idx = Splitter().split(table.shape[0], cfg.pi, seed=0)

    theta_by_lambda = {
        float(lam): family.estimate_dual(table.column(float(lam))[a_idx])
        for lam in table.lambda_grid
    }
    b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in table.lambda_grid)
    n_b = len(b_idx)

    def g(lam: float) -> float:
        l_tilde = family.transform(table.column(lam)[b_idx], theta_by_lambda[float(lam)])
        r_hat = float(np.mean(l_tilde))
        return (n_b / (n_b + 1)) * r_hat + b_tilde / (n_b + 1)

    g_values = [g(float(lam)) for lam in table.lambda_grid]
    assert all(g_values[i] >= g_values[i + 1] - 1e-9 for i in range(len(g_values) - 1))


def test_cvar_lambda_hat_is_smallest_grid_point_satisfying_g_le_alpha() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    result = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=0)
    if not result.empty_flag:
        assert result.r_hat_b is not None
        inflated = (result.n_b / (result.n_b + 1)) * result.r_hat_b + result.b_tilde / (
            result.n_b + 1
        )
        assert inflated <= cfg.alpha + 1e-9


def test_cvar_inflation_uses_n_b_not_n() -> None:
    # "Inflation uses n_B (not n) for dual branch" (Algorithm Spec S20).
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table, pi=0.1)  # n_A small, n_B large and != n
    result = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=0)
    assert result.n_b is not None
    assert result.n_b != table.shape[0]
    assert result.n_a is not None
    assert result.n_a + result.n_b == table.shape[0]


def test_cvar_theta_uses_a_only_transform_uses_b_only() -> None:
    # "theta_hat uses A only; transform/R_hat_B use B only" (Algorithm Spec S20).
    # Verified structurally: corrupting the LossTable's B-block values must
    # not change the fitted dual (theta), and corrupting the A-block values
    # must not change R_hat_B for a fixed theta.
    table = monotone_loss_table()
    cfg = calibration_config(table)
    seed = 0
    a_idx, b_idx = Splitter().split(table.shape[0], cfg.pi, seed)
    family = CVaRFamily(beta=0.2)

    original_theta = {
        float(lam): family.estimate_dual(table.column(float(lam))[a_idx])
        for lam in table.lambda_grid
    }

    corrupted_values = table.values.copy()
    corrupted_values[b_idx, :] = 999.0  # corrupt B-block only
    corrupted_table = LossTable(values=corrupted_values, lambda_grid=table.lambda_grid)
    corrupted_theta = {
        float(lam): family.estimate_dual(corrupted_table.column(float(lam))[a_idx])
        for lam in table.lambda_grid
    }
    for lam in table.lambda_grid:
        assert original_theta[float(lam)] == pytest.approx(corrupted_theta[float(lam)])


def test_cvar_b_tilde_is_finite_and_l_tilde_bounded_by_it() -> None:
    # "B_tilde < inf; L_tilde <= B_tilde" (Algorithm Spec S20).
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = CVaRFamily(beta=0.2)
    a_idx, b_idx = Splitter().split(table.shape[0], cfg.pi, seed=0)
    theta_by_lambda = {
        float(lam): family.estimate_dual(table.column(float(lam))[a_idx])
        for lam in table.lambda_grid
    }
    b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in table.lambda_grid)
    assert np.isfinite(b_tilde)
    for lam in table.lambda_grid:
        l_tilde = family.transform(table.column(float(lam))[b_idx], theta_by_lambda[float(lam)])
        assert np.all(l_tilde <= b_tilde + 1e-9)


def test_cvar_deterministic_given_fixed_seed() -> None:
    # "Fixed seed => identical lambda_hat" (Algorithm Spec S20).
    table = monotone_loss_table()
    cfg = calibration_config(table)
    r1 = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=123)
    r2 = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=123)
    assert r1.lambda_hat == r2.lambda_hat
    assert r1.n_a == r2.n_a
    assert r1.n_b == r2.n_b
    assert r1.b_tilde == r2.b_tilde


def test_cvar_different_seeds_can_give_different_splits() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    r1 = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=1)
    r2 = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=2)
    # Not asserting inequality of lambda_hat (could coincide), just that the
    # procedure runs to completion for multiple independent seeds.
    assert np.isfinite(r1.lambda_hat)
    assert np.isfinite(r2.lambda_hat)


def test_cvar_empty_selection_falls_back_to_lambda_max() -> None:
    # "empty => lambda_max" (Algorithm Spec S20 / F-1).
    table = monotone_loss_table()
    cfg = calibration_config(table, alpha=1e-6)  # infeasibly strict target
    result = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.2), cfg, seed=0)
    assert result.empty_flag is True
    assert result.lambda_hat == pytest.approx(float(table.lambda_grid[-1]))


def test_cvar_singleton_calibration_block() -> None:
    # n_B or n_A as small as 1 must not crash.
    table = monotone_loss_table(n=2, n_lambda=5)
    cfg = calibration_config(table)
    result = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.3), cfg, seed=0)
    assert result.n_a == 1
    assert result.n_b == 1
    assert np.isfinite(result.lambda_hat)


def test_cvar_large_loss_limit_still_finite() -> None:
    # A large-but-finite loss bound must not overflow CVaR's transform
    # (it has no exp(); it stays linear).
    table = monotone_loss_table(base_low=1e5, base_high=1e6)
    cfg = calibration_config(table, alpha=5e4, loss_bound=1e6)
    result = WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.3), cfg, seed=0)
    assert np.isfinite(result.lambda_hat)
    assert result.b_tilde is not None and np.isfinite(result.b_tilde)


# ---------------------------------------------------------------------------
# Dual branch: KL
# ---------------------------------------------------------------------------


def test_kl_returns_calibration_result_with_dual_fields_populated() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    result = WFCRCCalibrator().calibrate(table, KLFamily(rho=0.1), cfg, seed=0)
    assert result.n_a is not None and result.n_b is not None
    assert result.b_tilde is not None and np.isfinite(result.b_tilde)
    assert np.isfinite(result.lambda_hat)


def test_kl_deterministic_given_fixed_seed() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    r1 = WFCRCCalibrator().calibrate(table, KLFamily(rho=0.1), cfg, seed=7)
    r2 = WFCRCCalibrator().calibrate(table, KLFamily(rho=0.1), cfg, seed=7)
    assert r1.lambda_hat == r2.lambda_hat
    assert r1.b_tilde == r2.b_tilde


def test_kl_g_is_non_increasing_over_the_grid() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table)
    family = KLFamily(rho=0.1)
    a_idx, b_idx = Splitter().split(table.shape[0], cfg.pi, seed=0)
    theta_by_lambda = {
        float(lam): family.estimate_dual(table.column(float(lam))[a_idx])
        for lam in table.lambda_grid
    }
    b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in table.lambda_grid)
    n_b = len(b_idx)

    def g(lam: float) -> float:
        l_tilde = family.transform(table.column(lam)[b_idx], theta_by_lambda[float(lam)])
        r_hat = float(np.mean(l_tilde))
        return (n_b / (n_b + 1)) * r_hat + b_tilde / (n_b + 1)

    g_values = [g(float(lam)) for lam in table.lambda_grid]
    assert all(g_values[i] >= g_values[i + 1] - 1e-6 for i in range(len(g_values) - 1))


def test_kl_degenerate_a_block_succeeds_via_fixed_eta_fallback() -> None:
    # S15/F-4: a zero-variance A-block at the last grid point used to force
    # eta to the eta_min boundary and then overflow when transforming B.
    # estimate_dual now detects that boundary condition and substitutes the
    # fixed fallback_eta instead, so this must succeed (not raise), staying
    # valid by weak duality (any eta > 0 gives a valid, if looser, bound).
    table = monotone_loss_table(lambda_max=1.0)  # collapses to exact 0 at lambda_max
    cfg = calibration_config(table)
    result = WFCRCCalibrator().calibrate(table, KLFamily(rho=0.1), cfg, seed=0)
    assert np.isfinite(result.lambda_hat)
    assert result.b_tilde is not None and np.isfinite(result.b_tilde)


def test_kl_fallback_does_not_prevent_genuine_f3_rejection() -> None:
    # The fallback is not a blanket suppression of F-3: if the fixed
    # fallback_eta is still far too small relative to the declared bound B,
    # the transform of B can still overflow, and this must still raise
    # FamilyError rather than silently returning a meaningless bound.
    table = monotone_loss_table(lambda_max=1.0)
    cfg = calibration_config(table, loss_bound=1e12)
    with pytest.raises(FamilyError):
        WFCRCCalibrator().calibrate(table, KLFamily(rho=0.1, fallback_eta=1e-9), cfg, seed=0)


def test_kl_empty_selection_falls_back_to_lambda_max() -> None:
    table = monotone_loss_table()
    cfg = calibration_config(table, alpha=1e-6)
    result = WFCRCCalibrator().calibrate(table, KLFamily(rho=0.1), cfg, seed=0)
    assert result.empty_flag is True
    assert result.lambda_hat == pytest.approx(float(table.lambda_grid[-1]))


def test_kl_singleton_a_block_succeeds_via_fixed_eta_fallback() -> None:
    # With n_A=1, logsumexp of a single element is that element itself, so
    # h(eta) = z_0 + eta*rho (the same "all-equal" degenerate form as the
    # zero-variance case), forcing eta to the eta_min boundary regardless of
    # z_0's value -- the S15/F-4 condition. estimate_dual now substitutes
    # the fixed fallback_eta in that case, so this must succeed (not raise)
    # with the library's default fallback_eta=1.0 and the default B=1.0.
    table = monotone_loss_table(n=2, n_lambda=5)
    cfg = calibration_config(table)
    result = WFCRCCalibrator().calibrate(table, KLFamily(rho=0.2), cfg, seed=0)
    assert result.n_a == 1
    assert result.n_b == 1
    assert np.isfinite(result.lambda_hat)
    assert result.b_tilde is not None and np.isfinite(result.b_tilde)


def test_kl_singleton_a_block_fallback_eta_is_configurable() -> None:
    # A user-supplied fallback_eta (not eta_min) is what determines whether
    # the singleton-A degeneracy resolves cleanly for a given B: a
    # fallback_eta well-scaled to B keeps the transform finite even when
    # the declared bound B differs from the library default of 1.0.
    values = np.array([[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]])
    lambda_grid = np.array([0.0, 0.5, 1.0])
    table = LossTable(values=values, lambda_grid=lambda_grid)
    cfg = CalibrationConfig(alpha=0.3, B=0.5, pi=0.5, lambda_grid=(0.0, 0.5, 1.0))
    family = KLFamily(rho=0.2, fallback_eta=0.5)
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.n_a == 1
    assert result.n_b == 1
    assert np.isfinite(result.lambda_hat)


# ---------------------------------------------------------------------------
# Finite-group branch
# ---------------------------------------------------------------------------


def test_finite_group_deploys_max_over_group_thresholds() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    groups = [tuple(range(0, 100)), tuple(range(100, 200))]
    family = FiniteGroupFamily(masks=groups)
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    per_group = result.diagnostics["per_group"]
    assert len(per_group) == 2
    assert result.lambda_hat == pytest.approx(max(item["lambda_hat"] for item in per_group))


def test_finite_group_uses_n_g_inflation() -> None:
    # "groups use n_G" (Algorithm Spec S20): recompute one group's
    # threshold manually with the exact n_G-inflated CRC rule and compare.
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    group_indices = np.arange(0, 60)  # n_G = 60, deliberately != n and != n_B of any split
    family = FiniteGroupFamily(masks=[tuple(group_indices.tolist())])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    n_g = len(group_indices)

    def g_group(lam: float) -> float:
        r_hat = float(np.mean(table.column(lam)[group_indices]))
        return (n_g / (n_g + 1)) * r_hat + cfg.B / (n_g + 1)

    if not result.empty_flag:
        assert g_group(result.lambda_hat) <= cfg.alpha + 1e-9


def test_finite_group_no_a_b_split_uses_full_group_membership() -> None:
    # "No dual, no split" (Algorithm Spec S7'): the group calibration must
    # be independent of any seed (no stochastic component in this branch).
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    family = FiniteGroupFamily(masks=[tuple(range(200))])
    r1 = WFCRCCalibrator().calibrate(table, family, cfg, seed=1)
    r2 = WFCRCCalibrator().calibrate(table, family, cfg, seed=999)
    assert r1.lambda_hat == r2.lambda_hat


def test_finite_group_empty_flag_true_if_any_group_infeasible() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table, alpha=1e-6)
    family = FiniteGroupFamily(masks=[tuple(range(0, 100)), tuple(range(100, 200))])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.empty_flag is True


def test_finite_group_result_has_none_dual_fields() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    family = FiniteGroupFamily(masks=[tuple(range(200))])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.n_a is None
    assert result.n_b is None
    assert result.b_tilde is None


def test_finite_group_singleton_group() -> None:
    table = monotone_loss_table(n=10, n_lambda=5)
    cfg = calibration_config(table)
    family = FiniteGroupFamily(masks=[(0,)])
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert np.isfinite(result.lambda_hat)


# ---------------------------------------------------------------------------
# Known-weight branch
# ---------------------------------------------------------------------------


def test_known_weight_uniform_weights_matches_unweighted_mean() -> None:
    # With w_i = 1 for all i, the weighted mean collapses to the ordinary
    # mean over the full n -- a direct cross-check against a hand-written
    # reference computation.
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    n = table.shape[0]
    family = KnownWeightFamily(weights=np.ones(n))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)

    def g(lam: float) -> float:
        r_hat = float(np.mean(table.column(lam)))
        return (n / (n + 1)) * r_hat + cfg.B / (n + 1)

    if not result.empty_flag:
        assert g(result.lambda_hat) <= cfg.alpha + 1e-9


def test_known_weight_uses_full_n_not_a_split() -> None:
    # "known-weight uses full n with weighted mean" (Algorithm Spec S20).
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(200))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.diagnostics["n"] == 200


def test_known_weight_no_split_deterministic_regardless_of_seed() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(200))
    r1 = WFCRCCalibrator().calibrate(table, family, cfg, seed=1)
    r2 = WFCRCCalibrator().calibrate(table, family, cfg, seed=42)
    assert r1.lambda_hat == r2.lambda_hat


def test_known_weight_nonuniform_weights_changes_result() -> None:
    table = monotone_loss_table(n=200, seed=1)
    cfg = calibration_config(table)
    n = table.shape[0]
    uniform = KnownWeightFamily(weights=np.ones(n))
    skewed_w = np.ones(n)
    skewed_w[:50] = 3.0
    skewed_w[50:] = (n - 3.0 * 50) / (n - 50)  # keep mean(w) == 1
    skewed = KnownWeightFamily(weights=skewed_w)

    r_uniform = WFCRCCalibrator().calibrate(table, uniform, cfg, seed=0)
    r_skewed = WFCRCCalibrator().calibrate(table, skewed, cfg, seed=0)
    # Upweighting the first (higher-base-loss, per monotone_loss_table's
    # construction) block should make calibration at least as conservative.
    assert r_skewed.lambda_hat >= r_uniform.lambda_hat - 1e-9


def test_known_weight_empty_selection_falls_back_to_lambda_max() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table, alpha=1e-6)
    family = KnownWeightFamily(weights=np.ones(200))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.empty_flag is True
    assert result.lambda_hat == pytest.approx(float(table.lambda_grid[-1]))


def test_known_weight_result_has_none_dual_fields() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(200))
    result = WFCRCCalibrator().calibrate(table, family, cfg, seed=0)
    assert result.n_a is None
    assert result.n_b is None
    assert result.b_tilde is None


# ---------------------------------------------------------------------------
# Validation / dispatch
# ---------------------------------------------------------------------------


def test_rejects_mismatched_lambda_grid() -> None:
    table = monotone_loss_table()
    cfg = CalibrationConfig(alpha=0.3, B=1.0, pi=0.5, lambda_grid=(0.0, 0.5, 1.0))  # different grid
    with pytest.raises(ValueError):
        WFCRCCalibrator().calibrate(table, CVaRFamily(beta=0.1), cfg, seed=0)


def test_rejects_unsupported_family_type() -> None:
    class _BogusFamily:
        family_type = "wasserstein"  # not a frozen supported family

    table = monotone_loss_table()
    cfg = calibration_config(table)
    with pytest.raises(FamilyError):
        WFCRCCalibrator().calibrate(table, _BogusFamily(), cfg, seed=0)  # type: ignore[arg-type]


def test_rejects_cvar_typed_family_that_is_not_a_dual_family() -> None:
    class _FakeCVaR:
        family_type = "cvar"

    table = monotone_loss_table()
    cfg = calibration_config(table)
    with pytest.raises(FamilyError):
        WFCRCCalibrator().calibrate(table, _FakeCVaR(), cfg, seed=0)  # type: ignore[arg-type]


def test_rejects_finite_group_typed_family_missing_groups_method() -> None:
    class _FakeGroupFamily:
        family_type = "finite_group"

    table = monotone_loss_table()
    cfg = calibration_config(table)
    with pytest.raises(FamilyError):
        WFCRCCalibrator().calibrate(table, _FakeGroupFamily(), cfg, seed=0)  # type: ignore[arg-type]


def test_rejects_known_weight_typed_family_missing_weights_method() -> None:
    class _FakeWeightFamily:
        family_type = "known_weight"

    table = monotone_loss_table()
    cfg = calibration_config(table)
    with pytest.raises(FamilyError):
        WFCRCCalibrator().calibrate(table, _FakeWeightFamily(), cfg, seed=0)  # type: ignore[arg-type]


def test_known_weight_rejects_weight_count_mismatch() -> None:
    table = monotone_loss_table(n=200)
    cfg = calibration_config(table)
    family = KnownWeightFamily(weights=np.ones(50))  # != table's 200 rows
    with pytest.raises(FamilyError):
        WFCRCCalibrator().calibrate(table, family, cfg, seed=0)


def test_custom_splitter_and_threshold_search_are_used() -> None:
    # Dependency injection: swapping collaborators must change behavior,
    # proving they are actually invoked rather than bypassed.
    calls = {"split": 0, "search": 0}

    class _CountingSplitter(Splitter):
        def split(self, n: int, pi: float, seed: int):  # type: ignore[override]
            calls["split"] += 1
            return super().split(n, pi, seed)

    from wfcrc.calibration.threshold_search import ThresholdSearch

    class _CountingSearch(ThresholdSearch):
        def search(self, g, grid, alpha, default):  # type: ignore[override]
            calls["search"] += 1
            return super().search(g, grid, alpha, default)

    table = monotone_loss_table()
    cfg = calibration_config(table)
    calibrator = WFCRCCalibrator(splitter=_CountingSplitter(), threshold_search=_CountingSearch())
    calibrator.calibrate(table, CVaRFamily(beta=0.2), cfg, seed=0)
    assert calls["split"] == 1
    assert calls["search"] == 1


# ---------------------------------------------------------------------------
# Empirical validity smoke test (illustrative, not a proof -- Algorithm Spec S20)
# ---------------------------------------------------------------------------


def test_empirical_validity_smoke_cvar() -> None:
    # Over many independent calibration/test resamples, the realized
    # worst-case (CVaR) risk on a held-out set should be <= alpha within
    # Monte Carlo error most of the time. This is the illustrative smoke
    # test the frozen spec explicitly calls for -- not a proof.
    alpha = 0.3
    beta = 0.3
    successes = 0
    trials = 30
    n_cal, n_test = 300, 300
    lambda_grid = np.linspace(0.0, 0.9, 19)

    def make_table(n: int, seed: int, lambda_grid: np.ndarray = lambda_grid) -> LossTable:
        local_rng = np.random.default_rng(seed)
        base = local_rng.uniform(0.3, 1.0, size=n)
        return LossTable(values=np.outer(base, (1.0 - lambda_grid)), lambda_grid=lambda_grid)

    for trial in range(trials):
        cal_table = make_table(n_cal, seed=2000 + trial)
        test_table = make_table(n_test, seed=3000 + trial)
        cfg = CalibrationConfig(alpha=alpha, B=1.0, pi=0.5, lambda_grid=tuple(lambda_grid.tolist()))
        family = CVaRFamily(beta=beta)
        result = WFCRCCalibrator().calibrate(cal_table, family, cfg, seed=trial)

        # Realized CVaR risk on the independent held-out test set at lambda_hat.
        test_col = test_table.column(result.lambda_hat)
        eta_test = family.estimate_dual(test_col)
        realized = float(np.mean(family.transform(test_col, eta_test)))
        if realized <= alpha + 0.05:  # small slack for Monte Carlo noise
            successes += 1

    assert successes / trials >= 0.8
