"""``ScoreProvider`` — abstract, cached model-score contract (M4).

Per the Implementation Blueprint (§6, `data.ModelScoreProvider`) and the
MS2 Implementation Specification (§C2): produces and caches the pretrained
model's per-example outputs needed to build `C_λ` (e.g. per-pixel softmax
or masks), so the base model runs once. This milestone implements only the
abstract contract — no concrete provider for any specific pretrained model
is built. Running inference from a real checkpoint would need an actual
model and dataset, neither present in this environment; wiring one in is
deferred alongside `datasets`' concrete loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["ScoreArray", "ScoreBatch", "ScoreProvider"]

#: A single example's per-example model output (e.g. `[H,W,K]` softmax or
#: `[K,H,W]` logits for segmentation, `[K]` for classification) — shape is
#: dataset/model-specific, matching
#: :meth:`wfcrc.prediction_sets.base.PredictionSetConstructor.construct`'s
#: own dimension-agnostic `score` input.
ScoreArray = NDArray[np.float64]

#: A batch of :data:`ScoreArray`, one per requested id, in request order.
ScoreBatch = Sequence[NDArray[np.float64]]


class ScoreProvider(ABC):
    """Abstract, cached provider of per-example model outputs."""

    @abstractmethod
    def scores_for(self, id_: Hashable) -> ScoreArray:
        """Return the cached (or freshly computed) score for one example.

        Args:
            id_: An example id (as returned by `Dataset.ids()`).

        Returns:
            The `ScoreArray` for that example.
        """

    @abstractmethod
    def scores_batch(self, ids: Sequence[Hashable]) -> ScoreBatch:
        """Return scores for a batch of example ids, in the given order.

        Args:
            ids: A sequence of example ids.

        Returns:
            One `ScoreArray` per id, in the same order as `ids`.
        """

    @abstractmethod
    def model_fingerprint(self) -> str:
        """Return a stable fingerprint identifying the underlying model.

        Used as part of the cache key (Implementation Blueprint §9) so
        scores from different model checkpoints never collide.
        """
