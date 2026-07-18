"""``wfcrc.baselines`` — comparator baselines for E1-E12, behind the common `Calibrator` interface.

Per the MS9 milestone (`docs/EXPERIMENT_PROTOCOL.md`/`docs/MODEL_POLICY.md`
§1.2): every calibration-layer comparator baseline the frozen Experiment
Blueprint names, implemented behind :class:`~wfcrc.baselines.base.Calibrator`
— the same external shape :class:`~wfcrc.calibration.calibrator.WFCRCCalibrator`
already exposes, so an experiment/runner caller never branches on which
calibrator it is holding. Importing this package populates
:data:`~wfcrc.baselines.base.BASELINES` with every registered name:

- ``"wfcrc"`` — :class:`~wfcrc.baselines.wfcrc_adapter.WFCRCAdapter` (WF-CRC
  itself, adapted to the common interface).
- ``"vanilla_crc"`` — :class:`~wfcrc.baselines.vanilla_crc.VanillaCRC`
  (Angelopoulos et al. 2022).
- ``"lac"`` — :class:`~wfcrc.baselines.lac.SplitConformalLAC` (Sadinle,
  Lei & Wasserman 2019).
- ``"group_conditional"`` — :class:`~wfcrc.baselines.group_conditional.GroupConditionalCRC`
  (finite-group specialization used as the Gibbs-Cherian-Candès proxy).
- ``"robust_fdiv"`` — :class:`~wfcrc.baselines.robust_fdiv.RobustFDivergenceCP`
  (Cauchois-Duchi f-divergence-ball robust CP).
- ``"pooled_k_fold"`` / ``"total_n_inflation"`` / ``"fixed_eta"`` —
  :mod:`~wfcrc.baselines.negative_controls` (the E7 architecture ablations;
  the first two promoted from MS4's test-only harness, the third new).
- ``"temperature_scaled_lac"`` / ``"ensemble_aggregated_lac"`` — thin
  downstream wrappers over already-recalibrated/aggregated scores
  (:mod:`~wfcrc.baselines.scaling`, :mod:`~wfcrc.baselines.ensembles`);
  the score-level fitting/aggregation utilities those two modules also
  export (`fit_temperature`, `fit_selective_threshold`,
  `aggregate_mc_dropout_scores`, `aggregate_deep_ensemble_scores`) operate
  upstream of the `LossTable` boundary and are not themselves `Calibrator`s
  — see each module's own docstring for why.

**Explicitly not implemented** (disclosed, not silently substituted — see
the MS9 final report): AA-CRC and sem-CRC (no algorithm is transcribed
anywhere in the frozen Research Vault — both names appear only as bare,
uncited acronyms in related-work lists, confirmed by an exhaustive
repository/vault search) and Levy-Prokhorov robust-CP (the frozen
`Paper 1 - FRAMEWORK SPECIFICATION.md` itself names Levy-Prokhorov/
Wasserstein families as "future work... open gap," not currently
representable by this repository's ambiguity-family architecture).
"""

from __future__ import annotations

from wfcrc.baselines import (
    ensembles,
    group_conditional,
    lac,
    negative_controls,
    robust_fdiv,
    scaling,
    vanilla_crc,
    wfcrc_adapter,
)
from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.baselines.ensembles import (
    EnsembleAggregatedLAC,
    aggregate_deep_ensemble_scores,
    aggregate_mc_dropout_scores,
)
from wfcrc.baselines.group_conditional import GroupConditionalCRC
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.baselines.negative_controls import (
    FixedEtaWFCRC,
    PooledKFoldWFCRC,
    TotalNInflationWFCRC,
)
from wfcrc.baselines.robust_fdiv import RobustFDivergenceCP
from wfcrc.baselines.scaling import (
    TemperatureScaledLAC,
    apply_selective_threshold,
    apply_temperature,
    fit_selective_threshold,
    fit_temperature,
)
from wfcrc.baselines.vanilla_crc import VanillaCRC
from wfcrc.baselines.wfcrc_adapter import WFCRCAdapter

__all__ = [
    "BASELINES",
    "Calibrator",
    "EnsembleAggregatedLAC",
    "FixedEtaWFCRC",
    "GroupConditionalCRC",
    "PooledKFoldWFCRC",
    "RobustFDivergenceCP",
    "SplitConformalLAC",
    "TemperatureScaledLAC",
    "TotalNInflationWFCRC",
    "VanillaCRC",
    "WFCRCAdapter",
    "aggregate_deep_ensemble_scores",
    "aggregate_mc_dropout_scores",
    "apply_selective_threshold",
    "apply_temperature",
    "ensembles",
    "fit_selective_threshold",
    "fit_temperature",
    "group_conditional",
    "lac",
    "negative_controls",
    "robust_fdiv",
    "scaling",
    "vanilla_crc",
    "wfcrc_adapter",
]
