"""Typed, validated, hashable, layered run configuration.

Public entry points: :func:`load_config` to build a
:class:`~wfcrc.config.schema.Config` from layered YAML plus CLI overrides,
and the :class:`~wfcrc.config.schema.Config` type itself (and its section
dataclasses) for type-checked access to configuration values.
"""

from __future__ import annotations

from wfcrc.config.loader import load_config
from wfcrc.config.schema import (
    CalibrationConfig,
    Config,
    DataConfig,
    FamilyConfig,
    FamilyType,
    LossConfig,
    ModelConfig,
    RunnerConfig,
    SetsConfig,
)

__all__ = [
    "CalibrationConfig",
    "Config",
    "DataConfig",
    "FamilyConfig",
    "FamilyType",
    "LossConfig",
    "ModelConfig",
    "RunnerConfig",
    "SetsConfig",
    "load_config",
]
