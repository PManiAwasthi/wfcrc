"""Run-provenance capture for reproducibility manifests.

Per the Implementation Blueprint's reproducibility protocol (§17), every run
manifest should record enough environment/version/commit information to
explain *what code and environment* produced a result, complementing
:mod:`wfcrc.utils.seeds` (which explains *what randomness* produced it) and
:meth:`wfcrc.config.schema.Config.hash` (which explains *what parameters*
produced it). This module only reads/reports state — it never mutates
anything.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import numpy

from wfcrc._version import __version__ as _wfcrc_version

__all__ = ["get_environment_fingerprint", "get_git_commit"]


def get_git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable.

    Unavailability (not a git repository, ``git`` not installed, detached
    filesystem, etc.) is treated as a normal, non-fatal outcome — a
    reproducibility manifest should degrade gracefully rather than fail a
    run because provenance metadata is incomplete.

    Args:
        repo_dir: Directory to run ``git`` in; defaults to the current
            working directory.

    Returns:
        The full commit hash as a string, or ``None`` if it could not be
        determined.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def get_environment_fingerprint() -> dict[str, str]:
    """Capture a snapshot of the interpreter/platform/package versions.

    Returns:
        A dict with keys ``python_version``, ``platform``, ``wfcrc_version``,
        and ``numpy_version``, suitable for embedding in a run manifest.
    """
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "wfcrc_version": _wfcrc_version,
        "numpy_version": numpy.__version__,
    }
