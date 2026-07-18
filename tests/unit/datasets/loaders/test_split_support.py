"""Unit tests for :mod:`wfcrc.datasets.loaders._split_support` (DI-2).

The shared split-manifest mechanism every concrete loader family reuses.
These tests exercise it directly (the per-loader suites additionally cover
it end-to-end through each loader's own ``__init__``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcrc.datasets.loaders._split_support import (
    MANIFEST_FIELD,
    SPLIT_NAMES,
    build_manifest,
    read_split_manifest,
    validate_manifest_ids,
)
from wfcrc.exceptions import SerializationError, SplitLeakageError


def test_split_names_and_manifest_field_are_consistent() -> None:
    assert SPLIT_NAMES == ("train", "calibration", "test")
    assert set(MANIFEST_FIELD) == set(SPLIT_NAMES)
    assert set(MANIFEST_FIELD.values()) == {"train_ids", "cal_ids", "test_ids"}


def test_read_split_manifest_from_mapping_coerces_ids_to_str() -> None:
    out = read_split_manifest({"train": [1, 2], "calibration": [3], "test": [4]})
    assert out == {"train": ["1", "2"], "calibration": ["3"], "test": ["4"]}


def test_read_split_manifest_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"train": ["a"], "calibration": ["b"], "test": ["c"]}), encoding="utf-8"
    )
    assert read_split_manifest(path) == {"train": ["a"], "calibration": ["b"], "test": ["c"]}
    # A str path works identically to a Path.
    assert read_split_manifest(str(path)) == read_split_manifest(path)


def test_read_split_manifest_missing_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split manifest file not found"):
        read_split_manifest(tmp_path / "nope.json")


def test_read_split_manifest_unparsable_file_raises_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SerializationError, match="could not read/parse split manifest"):
        read_split_manifest(path)


def test_read_split_manifest_non_mapping_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        read_split_manifest([1, 2, 3])  # type: ignore[arg-type]


def test_read_split_manifest_missing_split_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing required split"):
        read_split_manifest({"train": [], "test": []})


def test_read_split_manifest_extra_split_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unrecognized split name"):
        read_split_manifest({"train": [], "calibration": [], "test": [], "extra": []})


def test_validate_manifest_ids_accepts_known_ids() -> None:
    manifest = {"train": ["a"], "calibration": ["b"], "test": ["c"]}
    validate_manifest_ids(manifest, ["a", "b", "c", "d"], pool_description="pool")


def test_validate_manifest_ids_rejects_unknown_id() -> None:
    manifest = {"train": ["a"], "calibration": ["zzz"], "test": ["c"]}
    with pytest.raises(ValueError, match=r"calibration.*not present in the discovered pool.*zzz"):
        validate_manifest_ids(manifest, ["a", "b", "c"], pool_description="pool")


def test_validate_manifest_ids_coerces_known_ids_to_str() -> None:
    # Known ids given as ints still match the str-coerced manifest ids.
    manifest = {"train": ["1"], "calibration": ["2"], "test": ["3"]}
    validate_manifest_ids(manifest, [1, 2, 3], pool_description="pool")


def test_build_manifest_constructs_split_manifest() -> None:
    manifest = build_manifest({"train": ["a"], "calibration": ["b"], "test": ["c"]})
    assert manifest.train_ids == ("a",)
    assert manifest.cal_ids == ("b",)
    assert manifest.test_ids == ("c",)


def test_build_manifest_rejects_overlap_via_frozen_gate() -> None:
    with pytest.raises(SplitLeakageError):
        build_manifest({"train": ["a"], "calibration": ["a"], "test": ["c"]})
