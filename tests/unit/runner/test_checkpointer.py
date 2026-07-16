"""Unit tests for :mod:`wfcrc.runner.checkpointer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from wfcrc.exceptions import CacheError
from wfcrc.runner.checkpointer import Checkpointer, stage_key


class TestStageKey:
    def test_deterministic(self) -> None:
        assert stage_key("experiment", {"a": 1}) == stage_key("experiment", {"a": 1})

    def test_sensitive_to_stage_name(self) -> None:
        assert stage_key("experiment", {"a": 1}) != stage_key("figures", {"a": 1})

    def test_sensitive_to_parts(self) -> None:
        assert stage_key("experiment", {"a": 1}) != stage_key("experiment", {"a": 2})

    def test_rejects_non_serializable(self) -> None:
        with pytest.raises(TypeError):
            stage_key("experiment", object())


class TestCheckpointer:
    def test_exists_is_false_initially(self, tmp_path: Path) -> None:
        checkpointer = Checkpointer(tmp_path)
        assert checkpointer.exists("k") is False

    def test_save_then_load_roundtrips(self, tmp_path: Path) -> None:
        checkpointer = Checkpointer(tmp_path)
        checkpointer.save({"lambda_hat": 0.5}, "k")
        assert checkpointer.exists("k") is True
        assert checkpointer.load("k") == {"lambda_hat": 0.5}

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        checkpointer = Checkpointer(tmp_path)
        with pytest.raises(CacheError):
            checkpointer.load("missing")

    def test_get_or_compute_computes_once(self, tmp_path: Path) -> None:
        checkpointer = Checkpointer(tmp_path)
        calls = {"n": 0}

        def compute() -> dict[str, int]:
            calls["n"] += 1
            return {"value": 42}

        first = checkpointer.get_or_compute("k", compute)
        second = checkpointer.get_or_compute("k", compute)
        assert first == {"value": 42}
        assert second == {"value": 42}
        assert calls["n"] == 1

    def test_get_or_compute_does_not_persist_on_exception(self, tmp_path: Path) -> None:
        checkpointer = Checkpointer(tmp_path)

        def failing() -> dict[str, int]:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            checkpointer.get_or_compute("k", failing)
        assert checkpointer.exists("k") is False

    def test_force_recompute_always_recomputes(self, tmp_path: Path) -> None:
        checkpointer = Checkpointer(tmp_path, force_recompute=True)
        calls = {"n": 0}

        def compute() -> dict[str, int]:
            calls["n"] += 1
            return {"value": calls["n"]}

        first = checkpointer.get_or_compute("k", compute)
        second = checkpointer.get_or_compute("k", compute)
        assert first == {"value": 1}
        assert second == {"value": 2}
        assert calls["n"] == 2
