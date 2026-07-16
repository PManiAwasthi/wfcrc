"""Unit tests for :class:`wfcrc.calibration.splitter.Splitter`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wfcrc.calibration.splitter import Splitter


def test_split_sizes_match_ceil_pi_n() -> None:
    splitter = Splitter()
    a_idx, b_idx = splitter.split(n=100, pi=0.3, seed=0)
    assert len(a_idx) == math.ceil(0.3 * 100)
    assert len(a_idx) + len(b_idx) == 100


def test_split_is_disjoint() -> None:
    splitter = Splitter()
    a_idx, b_idx = splitter.split(n=50, pi=0.4, seed=1)
    assert set(a_idx.tolist()).isdisjoint(set(b_idx.tolist()))


def test_split_covers_all_indices() -> None:
    splitter = Splitter()
    n = 37
    a_idx, b_idx = splitter.split(n=n, pi=0.5, seed=2)
    covered = set(a_idx.tolist()) | set(b_idx.tolist())
    assert covered == set(range(n))


def test_split_is_deterministic_given_same_seed() -> None:
    splitter = Splitter()
    a1, b1 = splitter.split(n=100, pi=0.5, seed=42)
    a2, b2 = splitter.split(n=100, pi=0.5, seed=42)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)


def test_split_differs_across_seeds() -> None:
    splitter = Splitter()
    a1, _ = splitter.split(n=100, pi=0.5, seed=1)
    a2, _ = splitter.split(n=100, pi=0.5, seed=2)
    assert not np.array_equal(a1, a2)


def test_split_indices_are_sorted() -> None:
    splitter = Splitter()
    a_idx, b_idx = splitter.split(n=60, pi=0.3, seed=7)
    assert list(a_idx) == sorted(a_idx.tolist())
    assert list(b_idx) == sorted(b_idx.tolist())


def test_split_dtype_is_int64() -> None:
    splitter = Splitter()
    a_idx, b_idx = splitter.split(n=10, pi=0.5, seed=0)
    assert a_idx.dtype == np.int64
    assert b_idx.dtype == np.int64


@pytest.mark.parametrize("n", [0, 1, -5])
def test_rejects_n_below_two(n: int) -> None:
    with pytest.raises(ValueError):
        Splitter().split(n=n, pi=0.5, seed=0)


@pytest.mark.parametrize("pi", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_pi(pi: float) -> None:
    with pytest.raises(ValueError):
        Splitter().split(n=10, pi=pi, seed=0)


def test_rejects_split_yielding_empty_a() -> None:
    # n=2, pi very small: ceil(0.01 * 2) = 1, which is actually valid (n_A=1).
    # Use n=1-adjacent degenerate combination instead: n=2, pi so small that
    # ceil(pi*n) would be 0 is impossible for pi>0, so test the other edge:
    # pi close to 1 with small n making n_A == n (empty B).
    with pytest.raises(ValueError):
        Splitter().split(n=2, pi=0.999, seed=0)


def test_minimal_n_two_valid_split() -> None:
    a_idx, b_idx = Splitter().split(n=2, pi=0.5, seed=0)
    assert len(a_idx) == 1
    assert len(b_idx) == 1


def test_seed_derivation_uses_frozen_seed_utility(monkeypatch: pytest.MonkeyPatch) -> None:
    # The splitter must route through wfcrc.utils.seeds.derive_seed (not a
    # bare/raw use of the given seed), per the frozen seed-utility policy.
    calls = []
    import wfcrc.calibration.splitter as splitter_module

    original = splitter_module.derive_seed

    def spy(name: str, base: int) -> int:
        calls.append((name, base))
        return original(name, base)

    monkeypatch.setattr(splitter_module, "derive_seed", spy)
    Splitter().split(n=10, pi=0.5, seed=99)
    assert calls == [("calibration.split", 99)]


def test_large_n_split() -> None:
    a_idx, b_idx = Splitter().split(n=10_000, pi=0.5, seed=0)
    assert len(a_idx) + len(b_idx) == 10_000
    assert set(a_idx.tolist()).isdisjoint(set(b_idx.tolist()))
