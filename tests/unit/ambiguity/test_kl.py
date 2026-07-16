"""Unit tests for :class:`wfcrc.ambiguity.kl.KLFamily`."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.ambiguity.kl import (
    KLDualParams,
    KLFamily,
    _bracket_and_minimize,
    _golden_section_minimize,
)
from wfcrc.exceptions import FamilyError
from wfcrc.utils.numerics import logsumexp


def _h(z: np.ndarray, rho: float, eta: float) -> float:
    """Reference (test-local) recomputation of the profiled dual objective."""
    return float(eta * (float(logsumexp(z / eta)) - np.log(z.size) + rho))


def test_family_type() -> None:
    assert KLFamily(rho=0.1).family_type == "kl"


@pytest.mark.parametrize("rho", [0.0, -0.1, -5.0])
def test_constructor_rejects_nonpositive_rho(rho: float) -> None:
    with pytest.raises(FamilyError):
        KLFamily(rho=rho)


@pytest.mark.parametrize("eta_min", [0.0, -1e-6])
def test_constructor_rejects_nonpositive_eta_min(eta_min: float) -> None:
    with pytest.raises(FamilyError):
        KLFamily(rho=0.1, eta_min=eta_min)


def test_estimate_dual_rejects_empty_column() -> None:
    family = KLFamily(rho=0.1)
    with pytest.raises(FamilyError):
        family.estimate_dual(np.array([]))


def test_estimate_dual_matches_brute_force_grid_search() -> None:
    rng = np.random.default_rng(0)
    z = rng.uniform(0.0, 1.0, size=200)
    rho = 0.1
    family = KLFamily(rho=rho)
    theta = family.estimate_dual(z)

    grid = np.logspace(-6, 3, 4000)
    grid_values = [_h(z, rho, e) for e in grid]
    grid_best = float(np.min(grid_values))

    assert _h(z, rho, theta.eta) == pytest.approx(grid_best, abs=1e-4)


@pytest.mark.parametrize("rho", [0.01, 0.1, 0.5, 2.0])
def test_estimate_dual_beats_random_etas(rho: float) -> None:
    rng = np.random.default_rng(1)
    z = rng.uniform(-1.0, 1.0, size=100)
    family = KLFamily(rho=rho)
    theta = family.estimate_dual(z)
    h_star = _h(z, rho, theta.eta)
    for eta in rng.uniform(family.eta_min, 50.0, size=25):
        assert h_star <= _h(z, rho, eta) + 1e-8


def test_estimate_dual_degenerate_all_equal_losses_triggers_fallback() -> None:
    # h(eta) = c + eta*rho is strictly increasing for constant z, so the
    # unconstrained minimizer would sit at the eta_min boundary -- which is
    # exactly the S15/F-4 dual-non-convergence condition. estimate_dual
    # must therefore substitute fallback_eta rather than returning the
    # boundary eta_min itself.
    eta_min = 1e-9
    fallback_eta = 2.5
    family = KLFamily(rho=0.2, eta_min=eta_min, fallback_eta=fallback_eta)
    z = np.full(50, 0.42)
    theta = family.estimate_dual(z)
    assert theta.eta == pytest.approx(fallback_eta)
    # mu recovers the constant exactly regardless of which eta is used,
    # since z is degenerate (constant): logsumexp(z/eta) - log(n) = z/eta
    # for any eta, so mu = eta * (z/eta) = z.
    assert theta.mu == pytest.approx(0.42, abs=1e-6)


def test_estimate_dual_singleton_column() -> None:
    # n_A=1 is always a degenerate point mass (S7' derivation: logsumexp of
    # one element is that element, so h(eta) = z_0 + eta*rho, strictly
    # increasing) -- this must trigger the fallback unconditionally.
    fallback_eta = 3.3
    family = KLFamily(rho=0.3, fallback_eta=fallback_eta)
    theta = family.estimate_dual(np.array([0.77]))
    assert theta.eta == pytest.approx(fallback_eta)
    assert theta.mu == pytest.approx(0.77, abs=1e-6)
    assert np.isfinite(theta.eta)
    assert np.isfinite(theta.mu)


def test_fallback_is_not_triggered_for_a_genuine_interior_optimum() -> None:
    # Regression guard against over-triggering: data with real spread and a
    # clear interior minimizer (as already exercised by the grid-search
    # cross-check test) must NOT be diverted to fallback_eta.
    fallback_eta = 999.0  # deliberately implausible, so any accidental use is obvious
    rng = np.random.default_rng(0)
    z = rng.uniform(0.0, 1.0, size=200)
    family = KLFamily(rho=0.1, fallback_eta=fallback_eta)
    theta = family.estimate_dual(z)
    assert theta.eta != pytest.approx(fallback_eta)
    assert theta.eta == pytest.approx(0.635104, abs=1e-4)  # matches the known grid-search value


def test_near_constant_calibration_block_triggers_fallback() -> None:
    # Not exactly constant (genuine floating-point variance is present),
    # but small enough that the KL-ball radius still pushes the optimizer
    # to the eta_min boundary. The fallback must trigger here too, not
    # just for the exactly-degenerate case.
    eta_min = 1e-9
    fallback_eta = 4.0
    family = KLFamily(rho=0.2, eta_min=eta_min, fallback_eta=fallback_eta)
    rng = np.random.default_rng(0)
    z = 0.5 + rng.uniform(-1e-10, 1e-10, size=50)  # near-constant, tiny genuine spread
    theta = family.estimate_dual(z)
    assert theta.eta == pytest.approx(fallback_eta)


def test_fallback_activation_is_exact_boundary_equality() -> None:
    # The trigger condition is that the unconstrained search returns
    # exactly eta_min (proven, by convexity, to occur iff h is
    # non-decreasing everywhere on [eta_min, infinity) -- see
    # _bracket_and_minimize). Confirm the private minimizer itself reports
    # the boundary for this degenerate input, motivating why estimate_dual's
    # equality check is exact rather than tolerance-based.
    z = np.full(20, 0.3)
    rho = 0.15
    eta_min = 1e-9

    def h(eta: float) -> float:
        return _h(z, rho, eta)

    assert _bracket_and_minimize(h, eta_min) == eta_min


def test_fallback_behaviour_is_deterministic() -> None:
    # Repeated calls on identical degenerate input must produce
    # bit-identical fallback results (no hidden state, no randomness).
    family = KLFamily(rho=0.2, fallback_eta=2.0)
    z = np.full(30, 0.6)
    theta1 = family.estimate_dual(z)
    theta2 = family.estimate_dual(z)
    assert theta1 == theta2
    assert theta1.eta == theta2.eta
    assert theta1.mu == theta2.mu


def test_fallback_eta_must_be_positive() -> None:
    with pytest.raises(FamilyError):
        KLFamily(rho=0.1, fallback_eta=0.0)
    with pytest.raises(FamilyError):
        KLFamily(rho=0.1, fallback_eta=-1.0)


def test_fallback_preserves_weak_duality_pointwise_domination() -> None:
    # The pointwise-domination property already proven generically for any
    # (eta, mu, rho>0) (test_dual_transform_pointwise_dominates_raw_loss)
    # must still hold for the SPECIFIC theta produced by the fallback path.
    family = KLFamily(rho=0.2, fallback_eta=1.5)
    z_a = np.full(10, 0.4)  # degenerate A-block -> triggers fallback
    theta = family.estimate_dual(z_a)
    z_b = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    dominated = family.transform(z_b, theta)
    assert np.all(dominated >= z_b - 1e-9)


def test_c_formula() -> None:
    family = KLFamily(rho=0.25)
    theta = KLDualParams(eta=2.0, mu=1.5)
    assert family.c(theta) == pytest.approx(1.5 + 2.0 * 0.25)


def test_t_formula_known_value() -> None:
    family = KLFamily(rho=0.1)
    theta = KLDualParams(eta=1.0, mu=0.0)
    # t(z=1; eta=1, mu=0) = 1*(exp(1) - 1)
    result = float(family.t(1.0, theta))
    assert result == pytest.approx(np.exp(1.0) - 1.0)


def test_t_at_z_equal_mu_is_zero() -> None:
    family = KLFamily(rho=0.1)
    theta = KLDualParams(eta=0.5, mu=0.3)
    result = float(family.t(0.3, theta))
    assert result == pytest.approx(0.0, abs=1e-12)


def test_transform_equals_c_plus_t() -> None:
    family = KLFamily(rho=0.2)
    theta = KLDualParams(eta=0.8, mu=0.1)
    z = np.array([0.0, 0.3, 0.6])
    expected = family.c(theta) + family.t(z, theta)
    np.testing.assert_allclose(family.transform(z, theta), expected)


def test_btil_equals_transform_at_bound() -> None:
    family = KLFamily(rho=0.1)
    theta = family.estimate_dual(np.linspace(0.0, 1.0, 40))
    assert family.btil(theta, 1.0) == pytest.approx(float(family.transform(1.0, theta)))


def test_btil_finite_for_realistic_inputs() -> None:
    family = KLFamily(rho=0.05)
    theta = family.estimate_dual(np.linspace(0.0, 1.0, 100))
    assert np.isfinite(family.btil(theta, 1.0))


def test_transform_raises_on_engineered_overflow() -> None:
    family = KLFamily(rho=0.1)
    # An eta far smaller than any value estimate_dual would ever return,
    # paired with a z far from mu, forces exp() to overflow to +inf.
    bad_theta = KLDualParams(eta=1e-300, mu=0.0)
    with pytest.raises(FamilyError):
        family.transform(1000.0, bad_theta)


def test_weak_duality_upper_bound_by_max_z() -> None:
    # h(eta*) must never exceed max(z): the KL dual value at the optimum is
    # at most the pointwise max (the eta -> 0+ limit), by convexity.
    rng = np.random.default_rng(2)
    z = rng.uniform(0.0, 1.0, size=150)
    family = KLFamily(rho=0.3)
    theta = family.estimate_dual(z)
    assert _h(z, family.rho, theta.eta) <= float(np.max(z)) + 1e-9


def test_dual_transform_pointwise_dominates_raw_loss() -> None:
    # c(theta) + t(z, theta) >= z for ANY eta>0, mu, rho>0 (from exp(x)>=1+x);
    # this is the pointwise inequality underlying weak duality for KL.
    rng = np.random.default_rng(3)
    for _ in range(30):
        rho = rng.uniform(0.01, 2.0)
        family = KLFamily(rho=rho)
        eta = rng.uniform(0.1, 5.0)
        mu = rng.uniform(-2.0, 2.0)
        theta = KLDualParams(eta=eta, mu=mu)
        z = rng.uniform(-2.0, 2.0, size=20)
        dominated = family.transform(z, theta)
        assert np.all(dominated >= z - 1e-9)


def test_deterministic_given_same_inputs() -> None:
    family = KLFamily(rho=0.1)
    z = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    theta1 = family.estimate_dual(z)
    theta2 = family.estimate_dual(z)
    assert theta1 == theta2


def test_bracket_and_minimize_on_known_quadratic() -> None:
    # Generic sanity check of the private minimizer against a textbook
    # convex function with a known closed-form minimizer, independent of
    # the KL-specific application. lo > 0, matching the function's documented
    # contract (real callers always pass eta_min > 0, validated at
    # KLFamily construction).
    def quadratic(x: float) -> float:
        return (x - 5.0) ** 2

    result = _bracket_and_minimize(quadratic, lo=1e-9)
    assert result == pytest.approx(5.0, abs=1e-6)


def test_bracket_and_minimize_finds_small_interior_minimum() -> None:
    # Regression test: an interior minimum well below the old fixed probe
    # point (1.0) must still be found, not mistaken for a boundary solution
    # by comparing lo against a far-away point.
    def quadratic(x: float) -> float:
        return (x - 0.05) ** 2

    result = _bracket_and_minimize(quadratic, lo=1e-12)
    assert result == pytest.approx(0.05, abs=1e-6)


def test_bracket_and_minimize_boundary_solution() -> None:
    # A strictly increasing function on [lo, inf) must return lo itself.
    def increasing(x: float) -> float:
        return x

    result = _bracket_and_minimize(increasing, lo=3.0)
    assert result == pytest.approx(3.0)


def test_bracket_and_minimize_falls_back_when_never_bracketed() -> None:
    # A pathological, ever-decreasing function cannot be bracketed within
    # the expansion budget; the fallback path must still return a finite
    # result (rather than crash or loop forever). lo != 1.0 so the initial
    # h(lo) vs h(start=max(lo,1.0)) comparison is a genuine two-point check,
    # not a degenerate a==b comparison.
    def ever_decreasing(x: float) -> float:
        return -x

    result = _bracket_and_minimize(ever_decreasing, lo=0.5)
    assert np.isfinite(result)
    assert result >= 0.5


def test_golden_section_runs_to_iteration_budget_on_huge_bracket() -> None:
    # An interval wide enough that 200 golden-section iterations cannot
    # shrink it below the convergence tolerance must still terminate
    # (via the iteration budget, not the early-exit tolerance check) and
    # return a finite value inside the bracket.
    def quadratic(x: float) -> float:
        return (x - 5.0) ** 2

    result = _golden_section_minimize(quadratic, lo=0.0, hi=1e40)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1e40
