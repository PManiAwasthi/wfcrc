"""Unit tests for :mod:`wfcrc.models.registry` (MS6.1)."""

from __future__ import annotations

from wfcrc.datasets.score_provider import ScoreProvider
from wfcrc.models import MODELS as MODELS_FROM_PACKAGE
from wfcrc.models.registry import MODELS


def test_models_is_a_dict() -> None:
    assert isinstance(MODELS, dict)


def test_models_starts_empty() -> None:
    # MS6.1 defines only the registry itself; MS6.4 populates concrete
    # score providers.
    assert MODELS == {}


def test_every_registered_entry_is_a_score_provider_subclass() -> None:
    # Vacuously true while MODELS is empty; guards every future MS6.4
    # registration against accidentally registering a non-ScoreProvider.
    for name, cls in MODELS.items():
        assert issubclass(
            cls, ScoreProvider
        ), f"MODELS[{name!r}] = {cls!r} is not a ScoreProvider subclass"


def test_models_reexported_from_package_init() -> None:
    assert MODELS_FROM_PACKAGE is MODELS
