"""Shared synthetic-data helpers for evaluation/verifier tests."""

from __future__ import annotations

import numpy as np

from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig


def monotone_loss_table(
    n: int = 200,
    lambda_max: float = 0.9,
    n_lambda: int = 21,
    *,
    seed: int = 0,
    base_low: float = 0.3,
    base_high: float = 1.0,
) -> LossTable:
    """A synthetic loss table, monotone non-increasing in lambda per row."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(base_low, base_high, size=n)
    lambda_grid = np.linspace(0.0, lambda_max, n_lambda)
    values = np.outer(base, (1.0 - lambda_grid))
    return LossTable(values=values, lambda_grid=lambda_grid)


def calibration_config(
    loss_table: LossTable, *, alpha: float = 0.3, loss_bound: float = 1.0, pi: float = 0.5
) -> CalibrationConfig:
    """A `CalibrationConfig` matching `loss_table`'s grid exactly."""
    return CalibrationConfig(
        alpha=alpha,
        B=loss_bound,
        pi=pi,
        lambda_grid=tuple(float(x) for x in loss_table.lambda_grid),
    )
