"""Shared synthetic-data helpers for `wfcrc.runner` tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import (
    CalibrationConfig,
    Config,
    DataConfig,
    FamilyConfig,
    LossConfig,
    ModelConfig,
    RunnerConfig,
    SetsConfig,
)


def monotone_loss_table(
    n: int = 60,
    lambda_max: float = 0.9,
    n_lambda: int = 11,
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


def full_config(
    loss_table: LossTable,
    tmp_path: Path,
    *,
    family: FamilyConfig | None = None,
    alpha: float = 0.3,
    seed: int = 0,
) -> Config:
    """A fully populated `Config` for `ExperimentRunner` tests."""
    return Config(
        data=DataConfig(name="synthetic"),
        model=ModelConfig(name="synthetic"),
        sets=SetsConfig(name="threshold"),
        loss=LossConfig(name="miscoverage"),
        family=family if family is not None else FamilyConfig(type="cvar", beta=0.2),
        calibration=calibration_config(loss_table, alpha=alpha),
        runner=RunnerConfig(cache_dir=str(tmp_path / "cache"), log_level="INFO"),
        seed=seed,
    )
