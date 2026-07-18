"""Unit tests for :mod:`wfcrc.models.registry` (MS6.1, populated from MS7)."""

from __future__ import annotations

from wfcrc.datasets.score_provider import ScoreProvider
from wfcrc.models import MODELS as MODELS_FROM_PACKAGE
from wfcrc.models.registry import MODELS
from wfcrc.models.scores.hippocampus_segmenter import HippocampusScoreProvider


def test_models_is_a_dict() -> None:
    assert isinstance(MODELS, dict)


def test_models_contains_exactly_the_ms7_entries() -> None:
    # MS7 implements and registers the minimum end-to-end vertical slice
    # only: one model, for one dataset.
    assert {"hippocampus_segmenter": HippocampusScoreProvider} == MODELS


def test_every_registered_entry_is_a_score_provider_subclass() -> None:
    for name, cls in MODELS.items():
        assert issubclass(
            cls, ScoreProvider
        ), f"MODELS[{name!r}] = {cls!r} is not a ScoreProvider subclass"


def test_models_reexported_from_package_init() -> None:
    assert MODELS_FROM_PACKAGE is MODELS
