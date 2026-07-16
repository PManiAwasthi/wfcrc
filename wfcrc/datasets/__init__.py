"""Dataset-loading, score-provider, and loss-table-assembly contracts (M3/M4/M7).

Public API: the :class:`~wfcrc.datasets.base.Dataset` /
:class:`~wfcrc.datasets.base.DatasetLoader` abstract contracts, the A1
hygiene gate (:func:`~wfcrc.datasets.base.assert_split_disjoint`,
:class:`~wfcrc.datasets.base.SplitManifest`), the
:class:`~wfcrc.datasets.score_provider.ScoreProvider` abstract contract, and
the concrete :class:`~wfcrc.datasets.loss_table_builder.LossTableBuilder`
that assembles a :class:`~wfcrc.calibration.loss_table.LossTable` from
them. No concrete dataset, model, or score provider for any specific named
dataset is implemented — see each module's docstring for why.
"""

from __future__ import annotations

from wfcrc.datasets.base import Dataset, DatasetLoader, SplitManifest, assert_split_disjoint
from wfcrc.datasets.loss_table_builder import LossTableBuilder
from wfcrc.datasets.score_provider import ScoreArray, ScoreBatch, ScoreProvider

__all__ = [
    "Dataset",
    "DatasetLoader",
    "LossTableBuilder",
    "ScoreArray",
    "ScoreBatch",
    "ScoreProvider",
    "SplitManifest",
    "assert_split_disjoint",
]
