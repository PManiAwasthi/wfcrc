"""Run-scoped, deterministic, JSON-lines structured logging.

Distinct from Python's standard :mod:`logging` (which individual wfcrc
modules use internally via ``logging.getLogger(__name__)`` for their own
diagnostics): this module produces the *provenance* record of a run —
one append-only JSONL file per run directory, with a fixed field order and
the timestamp isolated in its own field so two runs with identical inputs
produce byte-identical event streams once timestamps are stripped
("golden-diffable").
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from wfcrc.constants import DEFAULT_LOG_DIR, DEFAULT_LOG_LEVEL, LOG_FILENAME, TEXT_ENCODING
from wfcrc.utils.io import ensure_dir

__all__ = ["Logger", "get_logger"]

#: Numeric severity ordering, mirroring :mod:`logging`'s levels.
_LEVEL_VALUES: Mapping[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}


def _json_default(obj: Any) -> Any:
    """Best-effort encoding for values that plain ``json`` cannot serialize.

    Logging must never crash the computation it is observing, so unknown
    types fall back to ``str(obj)`` rather than raising.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _validate_level(level: str) -> None:
    """Raise :class:`ValueError` if ``level`` is not a recognized severity name."""
    if level not in _LEVEL_VALUES:
        raise ValueError(f"unknown level '{level}', expected one of {sorted(_LEVEL_VALUES)}")


class Logger:
    """A single run's structured JSONL event log plus a console mirror.

    Instances are obtained via :func:`get_logger`, not constructed directly.

    Attributes:
        level: Minimum severity (``DEBUG``/``INFO``/``WARNING``/``ERROR``)
            that will be written; lower-severity events are dropped.
        log_path: Path to the JSONL file this logger appends to.
    """

    def __init__(self, log_path: Path, level: str, *, console: TextIO | None = None) -> None:
        """Initialize the logger and eagerly open its log file.

        Args:
            log_path: Destination JSONL file path (parent must exist).
            level: Minimum severity to record.
            console: Stream to mirror human-readable lines to; defaults to
                ``sys.stdout``.

        Raises:
            ValueError: If ``level`` is not a recognized severity name.
            OSError: If ``log_path`` cannot be opened for writing.
        """
        _validate_level(level)
        self.level = level
        self.log_path = log_path
        self._console = sys.stdout if console is None else console
        self._handle = log_path.open("a", encoding=TEXT_ENCODING)

    def _enabled(self, level: str) -> bool:
        return _LEVEL_VALUES[level] >= _LEVEL_VALUES[self.level]

    def event(self, kind: str, *, level: str = "INFO", **fields: Any) -> None:
        """Record one structured event.

        Args:
            kind: Short, stable event-type label (e.g. ``"calibration_result"``).
            level: Severity of this event; filtered against ``self.level``.
            **fields: Arbitrary structured payload. Keys are sorted before
                writing so the field order is deterministic across
                processes/runs.

        Raises:
            ValueError: If ``level`` is not a recognized severity name.
        """
        _validate_level(level)
        if not self._enabled(level):
            return

        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "kind": kind,
        }
        for key in sorted(fields):
            record[key] = fields[key]

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        self._handle.write(line + "\n")
        self._handle.flush()
        print(line, file=self._console)

    def info(self, kind: str, **fields: Any) -> None:
        """Record an ``INFO``-severity event. See :meth:`event`."""
        self.event(kind, level="INFO", **fields)

    def warn(self, kind: str, **fields: Any) -> None:
        """Record a ``WARNING``-severity event. See :meth:`event`."""
        self.event(kind, level="WARNING", **fields)

    def error(self, kind: str, **fields: Any) -> None:
        """Record an ``ERROR``-severity event. See :meth:`event`."""
        self.event(kind, level="ERROR", **fields)

    def close(self) -> None:
        """Flush and close the underlying log file handle."""
        self._handle.close()

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def get_logger(
    run_dir: str | Path, level: str = DEFAULT_LOG_LEVEL, *, console: TextIO | None = None
) -> Logger:
    """Create a :class:`Logger` writing to ``<run_dir>/<DEFAULT_LOG_DIR>/<LOG_FILENAME>``.

    Args:
        run_dir: Root directory of the current run.
        level: Minimum severity to record (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
        console: Stream to mirror human-readable lines to; defaults to
            ``sys.stdout``.

    Returns:
        A ready-to-use :class:`Logger`.

    Raises:
        ValueError: If ``level`` is not a recognized severity name.
        OSError: If the log directory/file cannot be created or opened
            (e.g. permission denied).
    """
    log_dir = ensure_dir(Path(run_dir) / DEFAULT_LOG_DIR)
    return Logger(log_dir / LOG_FILENAME, level, console=console)
