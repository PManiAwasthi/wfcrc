"""``Checkpointer`` — resumable stage checkpointing (Implementation Blueprint §11, MS5 spec C2).

A thin wrapper over the already-frozen :class:`wfcrc.utils.cache.Cache`
(MS1): the Blueprint's own docstring for that module names it as "the
foundation for the score, loss-table, and dual caches added in later
milestones" — `Checkpointer` is exactly that later milestone, reusing
`Cache`'s content-addressed, read-through storage rather than
reimplementing it, per the "compose existing components, not reimplement
them" rule. The only addition is :func:`stage_key`, a stable content hash
of a stage's exact inputs, so two runs with identical inputs share a
checkpoint and any changed input gets a fresh one (Implementation
Blueprint §9's caching strategy, applied to run stages instead of scores).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wfcrc.utils.cache import Cache, make_key

__all__ = ["Checkpointer", "stage_key"]


def stage_key(stage: str, *parts: Any) -> str:
    """Derive a stable checkpoint key for one named stage.

    Args:
        stage: Short, stable stage name (e.g. `"experiment"`, `"figures"`).
        *parts: Any number of JSON-serializable values (dicts/lists/scalars/
            numpy) that together determine this stage's identity — every
            input that could change the stage's output must be included.

    Returns:
        A stable hex string key; identical `(stage, *parts)` always yields
        the same key, and the key changes if any part changes.

    Raises:
        TypeError: If any part is not JSON-serializable.
    """
    return make_key(stage, *parts)


class Checkpointer:
    """Resumable, content-addressed checkpoint storage for one run directory.

    Attributes:
        run_dir: The run directory this checkpointer is scoped to.
        force_recompute: When `True`, every stage recomputes and overwrites
            regardless of an existing checkpoint (forwarded to the
            underlying `Cache`).
    """

    def __init__(self, run_dir: str | Path, *, force_recompute: bool = False) -> None:
        """Initialize the checkpointer under `<run_dir>/checkpoints/`.

        Args:
            run_dir: Root directory of the current run.
            force_recompute: Forwarded to the underlying `Cache`.

        Raises:
            OSError: If the checkpoint directory cannot be created.
        """
        self.run_dir = Path(run_dir)
        self.force_recompute = force_recompute
        self._cache = Cache(self.run_dir / "checkpoints", force_recompute=force_recompute)

    def exists(self, key: str) -> bool:
        """Return `True` if a checkpoint for `key` is present on disk.

        Args:
            key: A key produced by :func:`stage_key`.
        """
        return self._cache.exists(key)

    def save(self, state: Any, key: str) -> None:
        """Persist `state` under `key`, overwriting any existing entry.

        Args:
            state: A JSON-serializable value (dict/list/scalars/numpy).
            key: A key produced by :func:`stage_key`.

        Raises:
            TypeError: If `state` is not serializable.
            OSError: On filesystem failures.
        """
        self._cache.put(key, state)

    def load(self, key: str) -> Any:
        """Load the previously saved state for `key`.

        Args:
            key: A key produced by :func:`stage_key`.

        Returns:
            The previously saved value.

        Raises:
            wfcrc.exceptions.CacheError: If no checkpoint exists for `key`,
                or it is corrupt.
        """
        return self._cache.load(key)

    def get_or_compute(self, key: str, compute_fn: Any) -> Any:
        """Return the checkpointed value for `key`, computing it on a miss.

        This is the "resume from a stage boundary" primitive: called again
        with the same `key` after a completed run, it returns the existing
        checkpoint without re-invoking `compute_fn` (unless
        `self.force_recompute`). If `compute_fn` raises, nothing is
        persisted — the stage is left un-checkpointed, exactly as if it had
        never started (this is what lets a failed verify STOP-gate halt a
        run without leaving a misleading "completed" checkpoint behind).

        Args:
            key: A key produced by :func:`stage_key`.
            compute_fn: Zero-argument callable producing the value to
                checkpoint on a miss.

        Returns:
            The checkpointed (or freshly computed) value.
        """
        return self._cache.get_or_compute(key, compute_fn)
