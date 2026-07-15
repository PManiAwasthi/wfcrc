"""Shared pytest fixtures for the wfcrc test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Iterator[Path]:
    """A throwaway directory representing a single experiment run's root.

    Yields:
        A fresh, empty directory unique to the calling test.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    yield run_dir
