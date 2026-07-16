"""Config-driven experiment orchestration with checkpointing, sweeps, and resume (M15).

Public API: :class:`~wfcrc.runner.runner.ExperimentRunner`,
:class:`~wfcrc.runner.runner.ResultBundle`,
:class:`~wfcrc.runner.runner.Manifest`,
:class:`~wfcrc.runner.runner.SweepConfig`,
:class:`~wfcrc.runner.runner.SweepCellFailure`, and
:class:`~wfcrc.runner.checkpointer.Checkpointer`. See
:mod:`wfcrc.runner.runner`'s module docstring for this milestone's scope
decision (no config-driven dataset/model resolution; `run()` takes
already-built `LossTable` objects directly).
"""

from __future__ import annotations

from wfcrc.runner.checkpointer import Checkpointer, stage_key
from wfcrc.runner.runner import (
    ExperimentRunner,
    Manifest,
    ResultBundle,
    SweepCellFailure,
    SweepConfig,
)

__all__ = [
    "Checkpointer",
    "ExperimentRunner",
    "Manifest",
    "ResultBundle",
    "SweepCellFailure",
    "SweepConfig",
    "stage_key",
]
