"""Ambiguity families `Q` (Mathematical Spec A5/D4; Algorithm Spec §7, §7').

Public API: the :class:`~wfcrc.ambiguity.base.AmbiguityFamily` /
:class:`~wfcrc.ambiguity.base.DualAmbiguityFamily` contracts and the four
frozen supported families — :class:`~wfcrc.ambiguity.cvar.CVaRFamily`,
:class:`~wfcrc.ambiguity.kl.KLFamily`,
:class:`~wfcrc.ambiguity.finite_group.FiniteGroupFamily`,
:class:`~wfcrc.ambiguity.known_weight.KnownWeightFamily` — plus a
name-keyed :data:`FAMILIES` registry for config-driven instantiation
(Implementation Blueprint §3). No other ambiguity family is supported
(Wasserstein / Levy-Prokhorov / optimal-transport families are explicitly
future work — Framework Specification §12).
"""

from __future__ import annotations

from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.ambiguity.kl import KLDualParams, KLFamily
from wfcrc.ambiguity.known_weight import KnownWeightFamily

__all__ = [
    "FAMILIES",
    "AmbiguityFamily",
    "CVaRFamily",
    "DualAmbiguityFamily",
    "FiniteGroupFamily",
    "KLDualParams",
    "KLFamily",
    "KnownWeightFamily",
]

#: Registry mapping a config `family.type` string to its concrete class.
FAMILIES: dict[str, type[AmbiguityFamily]] = {
    "cvar": CVaRFamily,
    "kl": KLFamily,
    "finite_group": FiniteGroupFamily,
    "known_weight": KnownWeightFamily,
}
