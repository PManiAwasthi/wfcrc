"""Unit tests for :mod:`wfcrc.utils.seeds`."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from wfcrc.exceptions import ReproducibilityError
from wfcrc.utils import seeds as seeds_module
from wfcrc.utils.seeds import derive_seed, rng_for, set_global_seed


@pytest.fixture(autouse=True)
def _reset_global_seed() -> Iterator[None]:
    """Ensure each test starts with no global seed set."""
    seeds_module._global_seed = None
    yield
    seeds_module._global_seed = None


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed("component.a", 42) == derive_seed("component.a", 42)


def test_derive_seed_distinct_names_give_distinct_seeds() -> None:
    a = derive_seed("component.a", 42)
    b = derive_seed("component.b", 42)
    assert a != b


def test_derive_seed_distinct_bases_give_distinct_seeds() -> None:
    a = derive_seed("component.a", 1)
    b = derive_seed("component.a", 2)
    assert a != b


def test_derive_seed_rejects_empty_name() -> None:
    with pytest.raises(ReproducibilityError):
        derive_seed("", 42)


def test_derive_seed_rejects_invalid_base() -> None:
    with pytest.raises(ReproducibilityError):
        derive_seed("component.a", -1)
    with pytest.raises(ReproducibilityError):
        derive_seed("component.a", 1.5)  # type: ignore[arg-type]


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ReproducibilityError):
        set_global_seed(-1)


def test_set_global_seed_rejects_non_int() -> None:
    with pytest.raises(ReproducibilityError):
        set_global_seed(1.5)  # type: ignore[arg-type]
    with pytest.raises(ReproducibilityError):
        set_global_seed(True)  # type: ignore[arg-type]


def test_set_global_seed_accepts_zero() -> None:
    set_global_seed(0)  # must not raise


def test_rng_for_requires_global_seed() -> None:
    with pytest.raises(ReproducibilityError):
        rng_for("component.a")


def test_rng_for_is_reproducible_given_same_global_seed() -> None:
    set_global_seed(123)
    first = rng_for("component.a").standard_normal(10)
    seeds_module._global_seed = None
    set_global_seed(123)
    second = rng_for("component.a").standard_normal(10)
    np.testing.assert_array_equal(first, second)


def test_rng_for_cross_name_independence() -> None:
    set_global_seed(123)
    a = rng_for("component.a").standard_normal(10)
    b = rng_for("component.b").standard_normal(10)
    assert not np.array_equal(a, b)


def test_rng_for_returns_independent_generators() -> None:
    set_global_seed(7)
    gen1 = rng_for("component.a")
    gen2 = rng_for("component.a")
    # Same derivation -> same seed -> same stream, but distinct objects.
    assert gen1 is not gen2
    np.testing.assert_array_equal(gen1.standard_normal(5), gen2.standard_normal(5))


def test_large_seed_accepted() -> None:
    from wfcrc.constants import MAX_SEED

    set_global_seed(MAX_SEED)
    rng_for("component.a")  # must not raise
