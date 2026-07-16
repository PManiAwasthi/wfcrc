"""Nested prediction-set constructors `C_λ` (Mathematical Spec D1, Algorithm Spec P-1).

Public API: the :class:`~wfcrc.prediction_sets.base.PredictionSetConstructor`
contract and its two frozen concrete implementations —
:class:`~wfcrc.prediction_sets.classification.ThresholdSets` (LAC) and
:class:`~wfcrc.prediction_sets.segmentation.MorphologicalSets` (dilation-margin)
— plus a name-keyed :data:`SETS` registry for config-driven instantiation
(Implementation Blueprint §3), matching the frozen `sets.type∈{threshold,
morphological}` config vocabulary (MS2 Implementation Spec, §C3 item 7).
"""

from __future__ import annotations

from wfcrc.prediction_sets.base import PredictionSetConstructor
from wfcrc.prediction_sets.classification import ThresholdSets
from wfcrc.prediction_sets.segmentation import MorphologicalSets

__all__ = ["SETS", "MorphologicalSets", "PredictionSetConstructor", "ThresholdSets"]

#: Registry mapping a config `sets.type` string to its concrete class.
SETS: dict[str, type[PredictionSetConstructor]] = {
    "threshold": ThresholdSets,
    "morphological": MorphologicalSets,
}
