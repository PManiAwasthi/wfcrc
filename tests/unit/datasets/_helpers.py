"""Shared synthetic `Dataset`/`ScoreProvider` test doubles for `wfcrc.datasets` tests."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from typing import Any

import numpy as np

from wfcrc.datasets.base import Dataset
from wfcrc.datasets.score_provider import ScoreArray, ScoreBatch, ScoreProvider


class FakeDataset(Dataset):
    """A minimal in-memory `Dataset` test double: `n` examples, 3-class one-hot labels."""

    def __init__(self, n: int, *, version: str = "v1", license_: str = "test-license") -> None:
        self._ids: tuple[int, ...] = tuple(range(n))
        self._labels = {i: np.array([i % 3 == k for k in range(3)]) for i in self._ids}
        self._version = version
        self._license = license_

    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        for i in self._ids:
            yield i, None, self._labels[i]

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> Sequence[Hashable]:
        return self._ids

    def labels(self, id_: Hashable) -> Any:
        return self._labels[id_]

    def meta(self) -> dict[str, Any]:
        return {"version": self._version, "license": self._license}


class FakeScoreProvider(ScoreProvider):
    """A minimal in-memory `ScoreProvider` test double."""

    def __init__(self, n: int, *, seed: int = 0, fingerprint: str = "fake-model-v1") -> None:
        rng = np.random.default_rng(seed)
        self._scores = {i: rng.uniform(0.0, 1.0, size=3) for i in range(n)}
        self._fingerprint = fingerprint

    def scores_for(self, id_: Hashable) -> ScoreArray:
        return self._scores[id_]

    def scores_batch(self, ids: Sequence[Hashable]) -> ScoreBatch:
        return [self._scores[i] for i in ids]

    def model_fingerprint(self) -> str:
        return self._fingerprint
