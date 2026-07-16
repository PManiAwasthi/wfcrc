"""Unit tests for :class:`wfcrc.ambiguity.known_weight.KnownWeightFamily`."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.ambiguity.known_weight import KnownWeightFamily
from wfcrc.exceptions import FamilyError


def test_family_type() -> None:
    family = KnownWeightFamily(weights=[1.0, 1.0, 1.0, 1.0])
    assert family.family_type == "known_weight"


def test_weights_round_trip() -> None:
    family = KnownWeightFamily(weights=[0.5, 1.5, 1.0, 1.0])
    np.testing.assert_allclose(family.weights(), [0.5, 1.5, 1.0, 1.0])


def test_weights_returns_defensive_copy() -> None:
    family = KnownWeightFamily(weights=[1.0, 1.0])
    returned = family.weights()
    returned[0] = 999.0
    np.testing.assert_allclose(family.weights(), [1.0, 1.0])


def test_rejects_empty_weights() -> None:
    with pytest.raises(FamilyError):
        KnownWeightFamily(weights=[])


def test_rejects_negative_weight() -> None:
    with pytest.raises(FamilyError):
        KnownWeightFamily(weights=[1.0, -0.5, 1.5])


def test_rejects_non_finite_weight() -> None:
    with pytest.raises(FamilyError):
        KnownWeightFamily(weights=[1.0, np.nan, 1.0])
    with pytest.raises(FamilyError):
        KnownWeightFamily(weights=[1.0, np.inf, 1.0])


def test_rejects_mean_far_from_one() -> None:
    with pytest.raises(FamilyError):
        KnownWeightFamily(weights=[2.0, 2.0, 2.0])


def test_accepts_mean_within_tolerance() -> None:
    # mean = 1.0 + 1e-9, well within the default tolerance.
    weights = [1.0 + 2e-9, 1.0, 1.0 - 2e-9]
    family = KnownWeightFamily(weights=weights)
    assert np.mean(family.weights()) == pytest.approx(1.0, abs=1e-6)


def test_custom_mean_tolerance() -> None:
    weights = [1.1, 1.0, 0.9]  # mean = 1.0 exactly, trivially within any tol
    family = KnownWeightFamily(weights=weights, mean_tol=1e-9)
    np.testing.assert_allclose(family.weights(), weights)


def test_zero_weight_allowed() -> None:
    # A weight of exactly zero is a valid (if extreme) nonnegative weight.
    weights = [0.0, 1.5, 1.5]
    family = KnownWeightFamily(weights=weights)
    np.testing.assert_allclose(family.weights(), weights)


def test_deterministic_construction() -> None:
    weights = [0.8, 1.2, 1.0]
    a = KnownWeightFamily(weights=weights).weights()
    b = KnownWeightFamily(weights=weights).weights()
    np.testing.assert_array_equal(a, b)
