"""Unit tests for :mod:`wfcrc.utils.numerics`."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.utils.numerics import clamp, logsumexp, quantile, safe_div, weighted_logsumexp


def _reference_logsumexp(x: np.ndarray) -> float:
    """Naive reference implementation (may overflow for large inputs)."""
    return float(np.log(np.sum(np.exp(x))))


def test_logsumexp_matches_reference_within_tolerance() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    assert logsumexp(x) == pytest.approx(_reference_logsumexp(x), abs=1e-12)


def test_logsumexp_overflow_stability() -> None:
    x = np.array([1000.0, 1000.0, 1000.0])
    # Naive exp(1000) overflows to inf; the stable version must not.
    result = logsumexp(x)
    assert np.isfinite(result)
    assert result == pytest.approx(1000.0 + np.log(3.0), abs=1e-12)


def test_logsumexp_single_element() -> None:
    assert logsumexp(np.array([3.0])) == pytest.approx(3.0, abs=1e-12)


def test_logsumexp_all_equal() -> None:
    x = np.full(10, 2.0)
    assert logsumexp(x) == pytest.approx(2.0 + np.log(10.0), abs=1e-12)


def test_logsumexp_neg_inf_entries_ignored() -> None:
    x = np.array([0.0, -np.inf, -np.inf])
    assert logsumexp(x) == pytest.approx(0.0, abs=1e-12)


def test_logsumexp_all_neg_inf_returns_neg_inf() -> None:
    x = np.array([-np.inf, -np.inf])
    assert logsumexp(x) == -np.inf


def test_logsumexp_rejects_nan() -> None:
    with pytest.raises(ValueError):
        logsumexp(np.array([1.0, np.nan]))


def test_logsumexp_rejects_pos_inf() -> None:
    with pytest.raises(ValueError):
        logsumexp(np.array([1.0, np.inf]))


def test_logsumexp_rejects_empty() -> None:
    with pytest.raises(ValueError):
        logsumexp(np.array([]))


def test_logsumexp_axis_reduction() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = logsumexp(x, axis=1)
    expected = np.array([np.log(2.0), 1.0 + np.log(2.0)])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_logsumexp_is_bit_deterministic() -> None:
    x = np.linspace(-5, 5, 37)
    first = logsumexp(x)
    second = logsumexp(x)
    assert first == second


def test_weighted_logsumexp_matches_manual_computation() -> None:
    x = np.array([0.0, 1.0, 2.0])
    w = np.array([1.0, 0.5, 0.0])
    result = weighted_logsumexp(x, w)
    expected = np.log(1.0 * np.exp(0.0) + 0.5 * np.exp(1.0) + 0.0 * np.exp(2.0))
    assert result == pytest.approx(expected, abs=1e-12)


def test_weighted_logsumexp_rejects_negative_weight() -> None:
    with pytest.raises(ValueError):
        weighted_logsumexp(np.array([0.0]), np.array([-1.0]))


def test_clamp_within_bounds_unchanged() -> None:
    assert clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)


def test_clamp_saturates_bounds() -> None:
    assert clamp(-1.0, 0.0, 1.0) == pytest.approx(0.0)
    assert clamp(2.0, 0.0, 1.0) == pytest.approx(1.0)


def test_clamp_idempotent() -> None:
    x = np.array([-5.0, 0.5, 5.0])
    once = clamp(x, 0.0, 1.0)
    twice = clamp(once, 0.0, 1.0)
    np.testing.assert_array_equal(once, twice)


def test_clamp_rejects_lo_gt_hi() -> None:
    with pytest.raises(ValueError):
        clamp(0.5, 1.0, 0.0)


def test_clamp_rejects_nan() -> None:
    with pytest.raises(ValueError):
        clamp(np.nan, 0.0, 1.0)


def test_clamp_rejects_inf() -> None:
    with pytest.raises(ValueError):
        clamp(np.inf, 0.0, 1.0)


def test_safe_div_normal_case() -> None:
    assert safe_div(1.0, 2.0) == pytest.approx(0.5)


def test_safe_div_at_b_near_zero_is_finite() -> None:
    result = safe_div(1.0, 0.0)
    assert np.isfinite(result)


def test_safe_div_preserves_sign_near_zero() -> None:
    positive = safe_div(1.0, 1e-20, eps=1e-6)
    negative = safe_div(1.0, -1e-20, eps=1e-6)
    assert positive > 0
    assert negative < 0


def test_safe_div_rejects_nonpositive_eps() -> None:
    with pytest.raises(ValueError):
        safe_div(1.0, 1.0, eps=0.0)


def test_quantile_matches_numpy_on_known_array() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert quantile(x, q) == pytest.approx(np.quantile(x, q), abs=1e-12)


def test_quantile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError):
        quantile(np.array([1.0, 2.0]), 1.5)


def test_quantile_rejects_empty_array() -> None:
    with pytest.raises(ValueError):
        quantile(np.array([]), 0.5)


def test_quantile_single_element() -> None:
    assert quantile(np.array([7.0]), 0.3) == pytest.approx(7.0)
