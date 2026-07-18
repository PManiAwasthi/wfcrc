"""Model Registry — name-keyed lookup from ``config.model.name`` to a concrete
``ScoreProvider`` subclass (MS6.1).

Per the MS6 Architecture Specification (§3.2), this mirrors the
:data:`wfcrc.ambiguity.FAMILIES` registry pattern (MS2) exactly, one layer
over :data:`wfcrc.datasets.registry.DATASETS`. The registry starts empty —
MS6.1 only establishes the pattern; no concrete ``ScoreProvider`` for any
named model exists yet (MS6.4 populates it). The Config Resolver (MS6.6) is
the intended consumer, looking up ``MODELS[config.model.name]`` exactly as
``wfcrc.runner.runner._build_family`` looks up ``FAMILIES[cfg.type]`` today.
"""

from __future__ import annotations

from wfcrc.datasets.score_provider import ScoreProvider

__all__ = ["MODELS"]

#: Registry mapping a config ``model.name`` string to its concrete
#: ``ScoreProvider`` class. Empty until MS6.4 registers the first concrete
#: provider.
MODELS: dict[str, type[ScoreProvider]] = {}
