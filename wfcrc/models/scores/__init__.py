"""Concrete :class:`~wfcrc.datasets.score_provider.ScoreProvider` implementations (MS7).

MS7 implements exactly one, for the minimum end-to-end vertical slice:
:class:`~wfcrc.models.scores.hippocampus_segmenter.HippocampusScoreProvider`.
No other Phase-A model is implemented in this pass.
"""

from __future__ import annotations

from wfcrc.models.scores.hippocampus_segmenter import HippocampusScoreProvider

__all__ = ["HippocampusScoreProvider"]
