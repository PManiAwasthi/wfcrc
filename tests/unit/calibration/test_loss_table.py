"""Unit tests for :class:`wfcrc.calibration.loss_table.LossTable`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfcrc.calibration.loss_table import LossTable


def _make(n: int = 5, t: int = 4) -> LossTable:
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 1.0, size=(n, t))
    lambda_grid = np.linspace(0.0, 1.0, t)
    return LossTable(values=values, lambda_grid=lambda_grid)


def test_shape() -> None:
    table = _make(n=5, t=4)
    assert table.shape == (5, 4)


def test_column_returns_correct_values() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lambda_grid = np.array([0.0, 1.0])
    table = LossTable(values=values, lambda_grid=lambda_grid)
    np.testing.assert_array_equal(table.column(0.0), [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(table.column(1.0), [2.0, 4.0, 6.0])


def test_row_returns_correct_values() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    lambda_grid = np.array([0.0, 1.0])
    table = LossTable(values=values, lambda_grid=lambda_grid)
    np.testing.assert_array_equal(table.row(0), [1.0, 2.0])
    np.testing.assert_array_equal(table.row(1), [3.0, 4.0])


def test_column_rejects_lambda_not_in_grid() -> None:
    table = _make()
    with pytest.raises(ValueError):
        table.column(0.123456789)


def test_rejects_non_2d_values() -> None:
    with pytest.raises(ValueError):
        LossTable(values=np.array([1.0, 2.0, 3.0]), lambda_grid=np.array([0.0, 1.0]))


def test_rejects_non_1d_lambda_grid() -> None:
    with pytest.raises(ValueError):
        LossTable(values=np.ones((3, 2)), lambda_grid=np.ones((2, 1)))


def test_rejects_empty_lambda_grid() -> None:
    with pytest.raises(ValueError):
        LossTable(values=np.ones((3, 0)), lambda_grid=np.array([]))


def test_rejects_mismatched_column_count() -> None:
    with pytest.raises(ValueError):
        LossTable(values=np.ones((3, 5)), lambda_grid=np.array([0.0, 1.0]))


def test_rejects_non_increasing_lambda_grid() -> None:
    with pytest.raises(ValueError):
        LossTable(values=np.ones((3, 3)), lambda_grid=np.array([0.0, 1.0, 0.5]))


def test_rejects_duplicate_lambda_values() -> None:
    with pytest.raises(ValueError):
        LossTable(values=np.ones((3, 3)), lambda_grid=np.array([0.0, 0.5, 0.5]))


def test_save_load_round_trip(tmp_path: Path) -> None:
    table = _make(n=6, t=5)
    path = tmp_path / "loss_table.json"
    table.save(path)
    loaded = LossTable.load(path)
    np.testing.assert_array_equal(loaded.values, table.values)
    np.testing.assert_array_equal(loaded.lambda_grid, table.lambda_grid)


def test_save_load_preserves_column_lookup(tmp_path: Path) -> None:
    table = _make(n=4, t=3)
    path = tmp_path / "loss_table.json"
    table.save(path)
    loaded = LossTable.load(path)
    for lam in table.lambda_grid:
        np.testing.assert_array_equal(loaded.column(float(lam)), table.column(float(lam)))


def test_single_column_table() -> None:
    table = LossTable(values=np.array([[0.5], [0.6]]), lambda_grid=np.array([0.3]))
    assert table.shape == (2, 1)
    np.testing.assert_array_equal(table.column(0.3), [0.5, 0.6])
