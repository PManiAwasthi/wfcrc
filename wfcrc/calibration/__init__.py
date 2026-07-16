"""Single-split WF-CRC calibration (Algorithm Specification §6-§9, §17).

Public API: :class:`~wfcrc.calibration.splitter.Splitter` (the sole
stochastic step), :class:`~wfcrc.calibration.threshold_search.ThresholdSearch`
(monotone binary search), :class:`~wfcrc.calibration.loss_table.LossTable`
(the calibration input contract),
:class:`~wfcrc.calibration.calibrator.WFCRCCalibrator` /
:class:`~wfcrc.calibration.calibrator.CalibrationResult` (the core
orchestrator — the single integration point of MS2), and
:func:`~wfcrc.calibration.pipeline.run_calibration_pipeline` /
:class:`~wfcrc.calibration.pipeline.PipelineResult` (MS3's thin
`LossTable → WFCRCCalibrator → optional Verifier` composition).
"""

from __future__ import annotations

from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.pipeline import PipelineResult, VerifierLike, run_calibration_pipeline
from wfcrc.calibration.splitter import Splitter
from wfcrc.calibration.threshold_search import ThresholdSearch

__all__ = [
    "CalibrationResult",
    "LossTable",
    "PipelineResult",
    "Splitter",
    "ThresholdSearch",
    "VerifierLike",
    "WFCRCCalibrator",
    "run_calibration_pipeline",
]
