"""``LossTableBuilder`` — assembles `L[i,λ]` from a dataset (M7).

Per the Implementation Blueprint (§6, `data.LossTableBuilder`) and the MS2
Implementation Specification (§C5): "assemble `L[i,λ] = l(C_λ(score_i),
Y_i)` over calibration examples x λ-grid; it is the only object
calibration consumes (L1a dimension-independence)." This assembly logic is
fully mechanical — it only iterates `(id, label, score)` triples through
an already-frozen `PredictionSetConstructor` and `LossEvaluator` — so it is
implemented concretely here, even though its `Dataset`/`ScoreProvider`
inputs are abstract-only this milestone (see :mod:`wfcrc.datasets.base`/
`.score_provider`): tests exercise it against small synthetic test doubles,
never a real dataset.

The MS2 spec's own "Public API" line
(`LossTableBuilder.build(dataset, constructor, loss, λ_grid)->LossTable`)
omits `score_provider`, but its "Inputs" line (item 3) lists `ScoreProvider`
as a required input alongside `Dataset`/`PredictionSetConstructor`/
`LossEvaluator` — the abbreviated signature line evidently omits it by
oversight, not by design (a `PredictionSetConstructor` cannot build `C_λ`
without a score to build it from). `build()` therefore takes
`score_provider` explicitly, as an additional parameter consistent with
the fuller "Inputs" list.

No disk caching is implemented here (the Implementation Blueprint's §9
cache-key scheme is keyed on `model_fingerprint`, which is only meaningful
once a concrete `ScoreProvider` exists) — this class assembles and returns
a `LossTable` directly; adding a cache layer once a concrete provider
exists is a later, purely additive change.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from wfcrc.calibration.loss_table import LossTable
from wfcrc.datasets.base import Dataset
from wfcrc.datasets.score_provider import ScoreProvider
from wfcrc.losses.base import LossEvaluator
from wfcrc.prediction_sets.base import PredictionSetConstructor

__all__ = ["LossTableBuilder"]


class LossTableBuilder:
    """Assembles a :class:`~wfcrc.calibration.loss_table.LossTable` from a dataset."""

    def build(
        self,
        dataset: Dataset,
        score_provider: ScoreProvider,
        constructor: PredictionSetConstructor,
        loss: LossEvaluator,
        lambda_grid: ArrayLike,
    ) -> LossTable:
        """Assemble `L[i,λ] = l(C_λ(score_i), Y_i)` over `dataset x lambda_grid`.

        Args:
            dataset: The (calibration or test) split to build a table over.
            score_provider: Supplies each example's score via `scores_for`.
            constructor: Builds `C_λ(score)` at each grid point.
            loss: Evaluates `l(C_λ(score), Y)`.
            lambda_grid: The λ-grid to assemble columns over.

        Returns:
            A `LossTable` of shape `(len(dataset), len(lambda_grid))`.

        Raises:
            ValueError: If `dataset` is empty or `lambda_grid` is empty
                (propagated from `LossTable`'s own validation, or raised
                directly for an empty dataset).
        """
        ids = dataset.ids()
        if len(ids) == 0:
            raise ValueError("dataset must be non-empty")
        grid = np.asarray(lambda_grid, dtype=np.float64)
        if grid.size == 0:
            raise ValueError("lambda_grid must be non-empty")

        values = np.empty((len(ids), grid.size), dtype=np.float64)
        for row, id_ in enumerate(ids):
            label = dataset.labels(id_)
            score = score_provider.scores_for(id_)
            for col, lam in enumerate(grid):
                predicted_set = constructor.construct(score, float(lam))
                values[row, col] = loss.evaluate(predicted_set, label)

        return LossTable(values=values, lambda_grid=grid)
