"""Unit tests for :mod:`wfcrc.utils.reproducibility`."""

from __future__ import annotations

from pathlib import Path

import pytest

from wfcrc.utils.reproducibility import get_environment_fingerprint, get_git_commit


def test_get_environment_fingerprint_has_expected_keys() -> None:
    fingerprint = get_environment_fingerprint()
    assert set(fingerprint) == {
        "python_version",
        "platform",
        "wfcrc_version",
        "numpy_version",
    }
    assert all(isinstance(v, str) and v for v in fingerprint.values())


def test_get_environment_fingerprint_is_deterministic_within_process() -> None:
    assert get_environment_fingerprint() == get_environment_fingerprint()


def test_get_git_commit_returns_string_or_none_in_this_repo() -> None:
    commit = get_git_commit(Path(__file__).resolve().parents[3])
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_get_git_commit_returns_none_outside_a_repo(tmp_path: Path) -> None:
    assert get_git_commit(tmp_path) is None


def test_get_git_commit_returns_none_when_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert get_git_commit() is None
