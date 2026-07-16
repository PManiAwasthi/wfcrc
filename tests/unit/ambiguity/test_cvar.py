"""Unit tests for :class:`wfcrc.ambiguity.cvar.CVaRFamily`."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.exceptions import FamilyError
from wfcrc.utils.numerics import quantile


def test_family_type() -> None:
    assert CVaRFamily(beta=0.1).family_type == "cvar"


@pytest.mark.parametrize("beta", [0.0, 1.0, -0.1, 1.1])
def test_constructor_rejects_out_of_range_beta(beta: float) -> None:
    with pytest.raises(FamilyError):
        CVaRFamily(beta=beta)


def test_estimate_dual_matches_quantile_directly() -> None:
    family = CVaRFamily(beta=0.4)
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    theta = family.estimate_dual(z)
    assert theta == pytest.approx(float(quantile(z, q=0.6)))


def test_estimate_dual_rejects_empty_column() -> None:
    family = CVaRFamily(beta=0.1)
    with pytest.raises(FamilyError):
        family.estimate_dual(np.array([]))


def test_c_returns_eta_itself() -> None:
    family = CVaRFamily(beta=0.2)
    assert family.c(0.7) == pytest.approx(0.7)
    assert family.c(-1.5) == pytest.approx(-1.5)


def test_t_zero_below_eta() -> None:
    family = CVaRFamily(beta=0.5)
    result = family.t(np.array([0.0, 0.3, 0.5]), theta=0.5)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


def test_t_known_value_above_eta() -> None:
    family = CVaRFamily(beta=0.25)
    # t(1.0; eta=0.2) = (1/0.25) * (1.0 - 0.2) = 4 * 0.8 = 3.2
    result = family.t(1.0, theta=0.2)
    assert float(result) == pytest.approx(3.2)


def test_t_scalar_and_array_consistent() -> None:
    family = CVaRFamily(beta=0.3)
    scalar_result = float(family.t(0.9, theta=0.4))
    array_result = family.t(np.array([0.9]), theta=0.4)
    assert scalar_result == pytest.approx(float(array_result[0]))


def test_transform_equals_c_plus_t() -> None:
    family = CVaRFamily(beta=0.3)
    theta = 0.4
    z = np.array([0.1, 0.5, 0.9])
    expected = family.c(theta) + family.t(z, theta)
    np.testing.assert_allclose(family.transform(z, theta), expected)


def test_btil_equals_transform_at_bound() -> None:
    family = CVaRFamily(beta=0.3)
    theta = family.estimate_dual(np.array([0.1, 0.4, 0.9]))
    b = 1.0
    assert family.btil(theta, b) == pytest.approx(float(family.transform(b, theta)))


def test_btil_is_finite_for_bounded_inputs() -> None:
    family = CVaRFamily(beta=0.05)
    theta = family.estimate_dual(np.linspace(0.0, 1.0, 50))
    assert np.isfinite(family.btil(theta, 1.0))


def test_dual_transform_pointwise_dominates_raw_loss() -> None:
    # c(theta) + t(z, theta) >= z for every eta and every z, for any beta in
    # (0,1). This is the pointwise property that underlies weak duality for
    # the CVaR family, and is exact (not just in expectation), so it can be
    # checked directly.
    rng = np.random.default_rng(0)
    for _ in range(30):
        beta = rng.uniform(0.01, 0.99)
        family = CVaRFamily(beta=beta)
        eta = rng.uniform(-2.0, 2.0)
        z = rng.uniform(-2.0, 2.0, size=20)
        dominated = family.transform(z, eta)
        assert np.all(dominated >= z - 1e-12)


def test_deterministic_given_same_inputs() -> None:
    family = CVaRFamily(beta=0.2)
    z = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    theta1 = family.estimate_dual(z)
    theta2 = family.estimate_dual(z)
    assert theta1 == theta2


def test_large_loss_limit_transform_scales_correctly() -> None:
    # As z -> B (a large bound), t grows linearly at rate 1/beta above eta.
    family = CVaRFamily(beta=0.1)
    eta = 0.5
    small_b, large_b = 10.0, 1000.0
    t_small = float(family.t(small_b, eta))
    t_large = float(family.t(large_b, eta))
    assert t_large > t_small
    assert t_large == pytest.approx((large_b - eta) / family.beta)
