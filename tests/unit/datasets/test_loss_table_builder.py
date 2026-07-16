"""Unit tests for :mod:`wfcrc.datasets.loss_table_builder`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.datasets._helpers import FakeDataset, FakeScoreProvider
from wfcrc.calibration.loss_table import LossTable
from wfcrc.datasets.loss_table_builder import LossTableBuilder
from wfcrc.losses.miscoverage import MiscoverageLoss
from wfcrc.prediction_sets.classification import ThresholdSets


def test_build_returns_a_loss_table_with_correct_shape() -> None:
    dataset = FakeDataset(6)
    provider = FakeScoreProvider(6, seed=0)
    lambda_grid = np.linspace(0.0, 1.0, 4)

    table = LossTableBuilder().build(
        dataset, provider, ThresholdSets(), MiscoverageLoss(), lambda_grid
    )

    assert isinstance(table, LossTable)
    assert table.shape == (6, 4)
    np.testing.assert_array_equal(table.lambda_grid, lambda_grid)


def test_build_values_match_manual_construct_and_evaluate() -> None:
    dataset = FakeDataset(3)
    provider = FakeScoreProvider(3, seed=0)
    lambda_grid = np.array([0.0, 0.5, 1.0])
    constructor = ThresholdSets()
    loss = MiscoverageLoss()

    table = LossTableBuilder().build(dataset, provider, constructor, loss, lambda_grid)

    for row, id_ in enumerate(dataset.ids()):
        label = dataset.labels(id_)
        score = provider.scores_for(id_)
        for col, lam in enumerate(lambda_grid):
            expected = loss.evaluate(constructor.construct(score, float(lam)), label)
            assert table.values[row, col] == pytest.approx(expected)


def test_build_output_is_monotone_non_increasing_per_row() -> None:
    # ThresholdSets (LAC) grows with lambda, and MiscoverageLoss is the
    # loss FNR/miscoverage are frozen to pair monotonically with a growing
    # set family -- an end-to-end sanity check that the assembled table
    # actually exhibits P-2, not just that the builder ran without error.
    dataset = FakeDataset(10)
    provider = FakeScoreProvider(10, seed=3)
    lambda_grid = np.linspace(0.0, 1.0, 11)

    table = LossTableBuilder().build(
        dataset, provider, ThresholdSets(), MiscoverageLoss(), lambda_grid
    )

    for row in range(table.shape[0]):
        assert np.all(np.diff(table.values[row, :]) <= 0.0)


def test_build_rejects_an_empty_dataset() -> None:
    dataset = FakeDataset(0)
    provider = FakeScoreProvider(0)
    with pytest.raises(ValueError, match="dataset must be non-empty"):
        LossTableBuilder().build(
            dataset, provider, ThresholdSets(), MiscoverageLoss(), np.array([0.5])
        )


def test_build_rejects_an_empty_lambda_grid() -> None:
    dataset = FakeDataset(2)
    provider = FakeScoreProvider(2)
    with pytest.raises(ValueError, match="lambda_grid must be non-empty"):
        LossTableBuilder().build(
            dataset, provider, ThresholdSets(), MiscoverageLoss(), np.array([])
        )
