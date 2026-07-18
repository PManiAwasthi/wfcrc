"""Shared synthetic fixtures for `tests/unit/baselines/*` (no real data/model)."""

from __future__ import annotations

import numpy as np

from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig

LAMBDA_GRID = np.linspace(0.0, 0.9, 10)


def population_losses(seed: int, n: int) -> np.ndarray:
    """A synthetic `(n, T)` loss table, monotone non-increasing per row.

    Mirrors `tests/unit/calibration/test_negative_controls.py`'s own
    `_population_losses` fixture (same construction), so the promoted
    baselines in `wfcrc.baselines.negative_controls` are exercised against
    the identical kind of population their original test-only harness used.
    """
    rng = np.random.default_rng(seed)
    base = np.clip(rng.pareto(a=2.5, size=n) * 0.15, 0.0, 1.0)
    return np.outer(base, (1.0 - LAMBDA_GRID))


def miscoverage_losses(seed: int, n: int) -> np.ndarray:
    """A synthetic `(n, T)` 0/1 miscoverage-style loss table, monotone non-increasing per row."""
    rng = np.random.default_rng(seed)
    thresholds = rng.uniform(0.05, 0.85, size=n)
    return (LAMBDA_GRID[np.newaxis, :] < thresholds[:, np.newaxis]).astype(np.float64)


def loss_table(values: np.ndarray) -> LossTable:
    return LossTable(values=values, lambda_grid=LAMBDA_GRID)


def cfg(alpha: float = 0.25, pi: float = 0.5, loss_bound: float = 1.0) -> CalibrationConfig:
    return CalibrationConfig(
        alpha=alpha, B=loss_bound, pi=pi, lambda_grid=tuple(LAMBDA_GRID.tolist())
    )
