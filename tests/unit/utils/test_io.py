"""Unit tests for :mod:`wfcrc.utils.io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfcrc.exceptions import SerializationError
from wfcrc.utils.io import (
    atomic_write,
    content_hash,
    ensure_dir,
    load_array,
    load_json,
    save_array,
    save_json,
)


def test_content_hash_is_deterministic() -> None:
    obj = {"a": 1, "b": [1, 2, 3], "c": {"d": 4.5}}
    assert content_hash(obj) == content_hash(obj)


def test_content_hash_is_key_order_invariant() -> None:
    obj_a = {"a": 1, "b": 2, "c": {"x": 1, "y": 2}}
    obj_b = {"c": {"y": 2, "x": 1}, "b": 2, "a": 1}
    assert content_hash(obj_a) == content_hash(obj_b)


def test_content_hash_differs_for_different_values() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_content_hash_width_controls_length() -> None:
    assert len(content_hash({"a": 1}, width=8)) == 8
    assert len(content_hash({"a": 1}, width=32)) == 32


def test_content_hash_handles_numpy_arrays_and_scalars() -> None:
    obj = {"arr": np.array([1, 2, 3], dtype=np.int64), "scalar": np.float64(1.5)}
    assert content_hash(obj) == content_hash(obj)


def test_content_hash_handles_non_float_numpy_scalar() -> None:
    # np.float64 is a subclass of Python's `float`, so json's C encoder
    # serializes it natively without invoking our `default` callback.
    # np.int64 is not a subclass of `int`, so this exercises the
    # `isinstance(obj, np.generic)` branch specifically.
    obj = {"count": np.int64(7)}
    assert content_hash(obj) == content_hash({"count": 7})


def test_content_hash_rejects_non_serializable() -> None:
    class Unserializable:
        pass

    with pytest.raises(TypeError):
        content_hash({"a": Unserializable()})


def test_content_hash_empty_object() -> None:
    assert content_hash({}) == content_hash({})
    assert content_hash([]) != content_hash({})


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.bin"
    atomic_write(target, b"hello world")
    assert target.read_bytes() == b"hello world"


def test_atomic_write_leaves_no_partial_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.bin"

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr("os.replace", _boom)
    with pytest.raises(OSError):
        atomic_write(target, b"data")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_overwrite_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    atomic_write(target, b"first")
    atomic_write(target, b"second")
    assert target.read_bytes() == b"second"


def test_save_load_json_round_trip(tmp_path: Path) -> None:
    obj = {"alpha": 0.1, "nested": {"z": 1, "a": 2}, "list": [1, 2, 3]}
    path = tmp_path / "obj.json"
    save_json(path, obj)
    assert load_json(path) == obj


def test_save_load_json_numpy_round_trip(tmp_path: Path) -> None:
    obj = {"arr": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)}
    path = tmp_path / "obj.json"
    save_json(path, obj)
    loaded = load_json(path)
    np.testing.assert_array_equal(loaded["arr"], obj["arr"])
    assert loaded["arr"].dtype == obj["arr"].dtype


def test_load_json_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SerializationError):
        load_json(tmp_path / "does_not_exist.json")


def test_load_json_corrupt_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SerializationError):
        load_json(path)


def test_save_load_array_round_trip(tmp_path: Path) -> None:
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "arr.npz"
    save_array(path, arr)
    loaded = load_array(path)
    np.testing.assert_array_equal(loaded, arr)


def test_load_array_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SerializationError):
        load_array(tmp_path / "missing.npz")


def test_ensure_dir_creates_nested_path(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.is_dir()
    assert result == target


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a"
    ensure_dir(target)
    ensure_dir(target)
    assert target.is_dir()
