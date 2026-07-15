"""Unit tests for :mod:`wfcrc.utils.logging`."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from wfcrc.utils.logging import get_logger


def _read_records(log_path: Path) -> list[dict[str, object]]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_get_logger_produces_valid_jsonl(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, console=io.StringIO())
    logger.event("run_start", seed=42, config_hash="abc123")
    logger.close()

    records = _read_records(logger.log_path)
    assert len(records) == 1
    assert records[0]["kind"] == "run_start"
    assert records[0]["seed"] == 42
    assert records[0]["config_hash"] == "abc123"


def test_level_filtering_drops_below_threshold(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, level="WARNING", console=io.StringIO())
    logger.info("should_be_dropped")
    logger.warn("should_be_kept")
    logger.close()

    records = _read_records(logger.log_path)
    assert len(records) == 1
    assert records[0]["kind"] == "should_be_kept"
    assert records[0]["level"] == "WARNING"


def test_error_level_recorded(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, console=io.StringIO())
    logger.error("failure", reason="disk full")
    logger.close()

    records = _read_records(logger.log_path)
    assert records[0]["level"] == "ERROR"
    assert records[0]["reason"] == "disk full"


def test_field_ordering_is_deterministic(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, console=io.StringIO())
    logger.event("e", zeta=1, alpha=2, mu=3)
    logger.close()

    line = logger.log_path.read_text(encoding="utf-8").splitlines()[0]
    keys = list(json.loads(line, object_pairs_hook=lambda pairs: pairs))
    key_names = [k for k, _ in keys]
    assert key_names == ["timestamp", "level", "kind", "alpha", "mu", "zeta"]


def test_timestamp_is_separable_for_golden_diff(tmp_path: Path) -> None:
    logger_a = get_logger(tmp_path / "run_a", console=io.StringIO())
    logger_a.event("calibration_result", lam=0.5, n_b=100)
    logger_a.close()

    logger_b = get_logger(tmp_path / "run_b", console=io.StringIO())
    logger_b.event("calibration_result", lam=0.5, n_b=100)
    logger_b.close()

    record_a = _read_records(logger_a.log_path)[0]
    record_b = _read_records(logger_b.log_path)[0]
    del record_a["timestamp"]
    del record_b["timestamp"]
    assert record_a == record_b


def test_unicode_fields_round_trip(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, console=io.StringIO())
    logger.event("message", text="日本語 café naïve — em—dash")
    logger.close()

    records = _read_records(logger.log_path)
    assert records[0]["text"] == "日本語 café naïve — em—dash"


def test_long_message_round_trips(tmp_path: Path) -> None:
    long_text = "x" * 20_000
    logger = get_logger(tmp_path, console=io.StringIO())
    logger.event("message", text=long_text)
    logger.close()

    records = _read_records(logger.log_path)
    assert records[0]["text"] == long_text


def test_numpy_scalar_and_array_fields_are_encoded(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, console=io.StringIO())
    logger.event("e", count=np.int64(7), arr=np.array([1, 2, 3]))
    logger.close()

    records = _read_records(logger.log_path)
    assert records[0]["count"] == 7
    assert records[0]["arr"] == [1, 2, 3]


def test_unserializable_field_falls_back_to_str(tmp_path: Path) -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque-repr"

    logger = get_logger(tmp_path, console=io.StringIO())
    logger.event("e", obj=Opaque())
    logger.close()

    records = _read_records(logger.log_path)
    assert records[0]["obj"] == "opaque-repr"


def test_unwritable_dir_raises(tmp_path: Path) -> None:
    # Create a plain file where the run directory should be, so mkdir fails.
    blocker = tmp_path / "run_dir"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(OSError):
        get_logger(blocker, console=io.StringIO())


def test_invalid_level_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        get_logger(tmp_path, level="NOT_A_LEVEL", console=io.StringIO())


def test_event_invalid_level_raises(tmp_path: Path) -> None:
    logger = get_logger(tmp_path, console=io.StringIO())
    try:
        with pytest.raises(ValueError):
            logger.event("e", level="NOT_A_LEVEL")
    finally:
        logger.close()


def test_context_manager_closes_handle(tmp_path: Path) -> None:
    with get_logger(tmp_path, console=io.StringIO()) as logger:
        logger.event("e")
    assert logger._handle.closed
