"""Model Registry — name-keyed lookup from ``config.model.name`` to a concrete
``ScoreProvider`` subclass (MS6.1, populated from MS7).

Per the MS6 Architecture Specification (§3.2), this mirrors the
:data:`wfcrc.ambiguity.FAMILIES` registry pattern (MS2) exactly, one layer
over :data:`wfcrc.datasets.registry.DATASETS`. MS6.1 established the
(then-empty) pattern; MS7 registers the first concrete provider —
``"hippocampus_segmenter"`` ->
:class:`~wfcrc.models.scores.hippocampus_segmenter.HippocampusScoreProvider`,
the minimum vertical-slice model for the first end-to-end pipeline. The
Config Resolver (MS6.6, not yet built) is the intended future consumer,
looking up ``MODELS[config.model.name]`` exactly as
``wfcrc.runner.runner._build_family`` looks up ``FAMILIES[cfg.type]`` today.
"""

from __future__ import annotations

from wfcrc.datasets.score_provider import ScoreProvider
from wfcrc.models.scores.hippocampus_segmenter import HippocampusScoreProvider

__all__ = ["MODELS"]

#: Registry mapping a config ``model.name`` string to its concrete
#: ``ScoreProvider`` class.
MODELS: dict[str, type[ScoreProvider]] = {
    "hippocampus_segmenter": HippocampusScoreProvider,
}
