"""Bounded, monotone task losses `l(set, label)` (Algorithm Spec §2, §5, P-2).

Public API: the :class:`~wfcrc.losses.base.LossEvaluator` contract and its
three frozen concrete implementations — :class:`~wfcrc.losses.fnr.FNRLoss`,
:class:`~wfcrc.losses.fpr.FPRLoss`,
:class:`~wfcrc.losses.miscoverage.MiscoverageLoss` — plus a name-keyed
:data:`LOSSES` registry for config-driven instantiation (Implementation
Blueprint §3).
"""

from __future__ import annotations

from wfcrc.losses.base import LossEvaluator
from wfcrc.losses.fnr import FNRLoss
from wfcrc.losses.fpr import FPRLoss
from wfcrc.losses.miscoverage import MiscoverageLoss

__all__ = ["LOSSES", "FNRLoss", "FPRLoss", "LossEvaluator", "MiscoverageLoss"]

#: Registry mapping a config `loss.name` string to its concrete class.
LOSSES: dict[str, type[LossEvaluator]] = {
    "fnr": FNRLoss,
    "fpr": FPRLoss,
    "miscoverage": MiscoverageLoss,
}
