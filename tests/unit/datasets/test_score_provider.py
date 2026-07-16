"""Unit tests for :mod:`wfcrc.datasets.score_provider`."""

from __future__ import annotations

import numpy as np

from tests.unit.datasets._helpers import FakeScoreProvider
from wfcrc.datasets.score_provider import ScoreProvider


def test_fake_score_provider_satisfies_the_contract() -> None:
    provider = FakeScoreProvider(4, seed=0)
    assert isinstance(provider, ScoreProvider)
    assert provider.model_fingerprint() == "fake-model-v1"


def test_scores_for_returns_a_score_array() -> None:
    provider = FakeScoreProvider(4, seed=0)
    score = provider.scores_for(2)
    assert isinstance(score, np.ndarray)
    assert score.shape == (3,)


def test_scores_for_is_deterministic_given_the_same_seed() -> None:
    first = FakeScoreProvider(4, seed=7)
    second = FakeScoreProvider(4, seed=7)
    np.testing.assert_array_equal(first.scores_for(0), second.scores_for(0))


def test_scores_batch_matches_scores_for_in_requested_order() -> None:
    provider = FakeScoreProvider(5, seed=1)
    batch = provider.scores_batch([3, 1, 4])
    assert len(batch) == 3
    np.testing.assert_array_equal(batch[0], provider.scores_for(3))
    np.testing.assert_array_equal(batch[1], provider.scores_for(1))
    np.testing.assert_array_equal(batch[2], provider.scores_for(4))


def test_model_fingerprint_is_configurable() -> None:
    provider = FakeScoreProvider(2, fingerprint="custom-v2")
    assert provider.model_fingerprint() == "custom-v2"
