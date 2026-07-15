"""Unit tests for :mod:`wfcrc.utils.cache`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfcrc.exceptions import CacheError
from wfcrc.utils.cache import Cache, make_key


def test_make_key_is_deterministic() -> None:
    assert make_key("a", 1, {"b": 2}) == make_key("a", 1, {"b": 2})


def test_make_key_distinguishes_different_parts() -> None:
    assert make_key("a", 1) != make_key("a", 2)


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    calls = []

    def compute() -> dict[str, int]:
        calls.append(1)
        return {"result": 42}

    key = make_key("query", 1)
    assert not cache.exists(key)

    first = cache.get_or_compute(key, compute)
    assert first == {"result": 42}
    assert len(calls) == 1

    second = cache.get_or_compute(key, compute)
    assert second == {"result": 42}
    assert len(calls) == 1  # compute_fn not called again on hit


def test_get_or_compute_returns_identical_value_on_hit(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = make_key("k")

    def _fail() -> None:
        raise AssertionError("should not run")

    value = cache.get_or_compute(key, lambda: {"a": [1, 2, 3]})
    cached = cache.get_or_compute(key, _fail)
    assert value == cached


def test_cache_stores_numpy_arrays(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = make_key("arr")
    arr = np.arange(10, dtype=np.float64)
    cache.put(key, arr)
    loaded = cache.load(key)
    np.testing.assert_array_equal(loaded, arr)


def test_put_twice_is_consistent(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = make_key("k")
    cache.put(key, {"v": 1})
    cache.put(key, {"v": 1})
    assert cache.load(key) == {"v": 1}


def test_put_overwrite_changes_value(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = make_key("k")
    cache.put(key, {"v": 1})
    cache.put(key, {"v": 2})
    assert cache.load(key) == {"v": 2}


def test_force_recompute_bypasses_existing_entry(tmp_path: Path) -> None:
    cache = Cache(tmp_path, force_recompute=True)
    key = make_key("k")
    cache.put(key, {"v": "old"})

    calls = []

    def compute() -> dict[str, str]:
        calls.append(1)
        return {"v": "new"}

    result = cache.get_or_compute(key, compute)
    assert result == {"v": "new"}
    assert len(calls) == 1
    assert cache.load(key) == {"v": "new"}


def test_load_missing_key_raises(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    with pytest.raises(CacheError):
        cache.load(make_key("missing"))


def test_load_corrupt_entry_raises(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = make_key("k")
    cache.put(key, {"v": 1})
    # Corrupt the on-disk JSON entry directly.
    (tmp_path / f"{key}.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CacheError):
        cache.load(key)


def test_get_or_compute_raises_on_corrupt_entry_by_default(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = make_key("k")
    cache.put(key, {"v": 1})
    (tmp_path / f"{key}.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CacheError):
        cache.get_or_compute(key, lambda: {"v": 2})


def test_missing_cache_dir_is_created(tmp_path: Path) -> None:
    nested = tmp_path / "does" / "not" / "exist"
    cache = Cache(nested)
    assert nested.is_dir()
    assert cache.dir == nested


def test_exists_false_for_unknown_key(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert not cache.exists(make_key("nope"))
