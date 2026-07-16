"""Unit tests for :mod:`wfcrc.ambiguity.base` (shared dual-family plumbing)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from wfcrc.ambiguity.base import DualAmbiguityFamily
from wfcrc.config.schema import FamilyType
from wfcrc.exceptions import FamilyError


class _ToyDualFamily(DualAmbiguityFamily):
    """Minimal concrete family isolating `transform`/`btil`'s shared logic
    from any real family's mathematics."""

    def __init__(self, *, blow_up: bool = False) -> None:
        self.blow_up = blow_up

    @property
    def family_type(self) -> FamilyType:
        return "cvar"  # reuse an existing tag; irrelevant to this test

    def estimate_dual(self, losses_a_col: NDArray[np.float64]) -> Any:
        return 1.0

    def c(self, theta: Any) -> float:
        return 10.0

    def t(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
        if self.blow_up:
            return np.asarray(np.inf)
        return np.asarray(z, dtype=np.float64) * 2.0


def test_transform_composes_c_and_t() -> None:
    family = _ToyDualFamily()
    result = family.transform(np.array([1.0, 2.0, 3.0]), theta=None)
    np.testing.assert_allclose(result, [12.0, 14.0, 16.0])


def test_btil_calls_transform_at_bound() -> None:
    family = _ToyDualFamily()
    assert family.btil(theta=None, loss_bound=5.0) == pytest.approx(10.0 + 10.0)


def test_transform_raises_family_error_on_non_finite_result() -> None:
    family = _ToyDualFamily(blow_up=True)
    with pytest.raises(FamilyError):
        family.transform(np.array([1.0]), theta=None)


def test_btil_raises_family_error_on_non_finite_result() -> None:
    family = _ToyDualFamily(blow_up=True)
    with pytest.raises(FamilyError):
        family.btil(theta=None, loss_bound=1.0)


def test_transform_rejects_nan_result() -> None:
    class _NanFamily(DualAmbiguityFamily):
        @property
        def family_type(self) -> FamilyType:
            return "kl"

        def estimate_dual(self, losses_a_col: NDArray[np.float64]) -> Any:
            return None

        def c(self, theta: Any) -> float:
            return 0.0

        def t(self, z: ArrayLike, theta: Any) -> NDArray[np.float64]:
            return np.asarray(np.nan)

    with pytest.raises(FamilyError):
        _NanFamily().transform(1.0, theta=None)
