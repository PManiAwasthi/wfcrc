"""Generic content-addressed, read-through cache.

Foundation for the score, loss-table, and dual caches added in later
milestones. Entries are immutable once written (a given key always maps to
the same value) and keys are derived from the full set of inputs that
produced the value, so a cache can never silently serve a stale result for
different inputs — a key collision would require a hash collision.

Concurrency scope: single-process only (matches the rest of MS1's
reproducibility model); no file locking is implemented.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from wfcrc.constants import CACHE_KEY_HASH_WIDTH
from wfcrc.exceptions import CacheError, SerializationError
from wfcrc.utils.io import content_hash, ensure_dir, load_array, load_json, save_array, save_json

__all__ = ["Cache", "make_key"]

_logger = logging.getLogger(__name__)


def make_key(*parts: Any) -> str:
    """Derive a stable cache key from arbitrary JSON-serializable ``parts``.

    Uses the full, untruncated SHA-256 digest
    (:data:`wfcrc.constants.CACHE_KEY_HASH_WIDTH`) rather than
    :func:`~wfcrc.utils.io.content_hash`'s shorter default width, since
    cache keys accumulate across long-running research sweeps where
    collision probability at a truncated width is no longer negligible.

    Args:
        *parts: Any number of JSON-serializable values (dict/list/scalars/
            numpy) that together determine the cached value's identity.

    Returns:
        A stable hex string key; identical ``parts`` always yield the same
        key, and the key changes if any part changes.

    Raises:
        TypeError: If any part is not JSON-serializable.
    """
    return content_hash(list(parts), width=CACHE_KEY_HASH_WIDTH)


class Cache:
    """A directory-backed, content-addressed read-through cache.

    Values that are :class:`numpy.ndarray` are stored as ``<key>.npz``;
    everything else is stored as ``<key>.json`` (via the numpy-aware
    encoder in :mod:`wfcrc.utils.io`, so nested arrays inside a dict are
    also supported).

    Attributes:
        dir: Root directory backing this cache (created if missing).
        force_recompute: When ``True``, :meth:`get_or_compute` always
            recomputes and overwrites, ignoring any existing entry.
    """

    def __init__(self, directory: str | Path, *, force_recompute: bool = False) -> None:
        """Initialize the cache, creating ``directory`` if it does not exist.

        Args:
            directory: Root directory for cache entries.
            force_recompute: When ``True``, bypass existing entries in
                :meth:`get_or_compute` (also bypasses corrupt entries).

        Raises:
            OSError: If ``directory`` cannot be created.
        """
        self.dir = ensure_dir(directory)
        self.force_recompute = force_recompute

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.dir / f"{key}.json", self.dir / f"{key}.npz"

    def exists(self, key: str) -> bool:
        """Check whether an entry for ``key`` is present on disk.

        Args:
            key: Cache key, typically produced by :func:`make_key`.

        Returns:
            ``True`` if a ``.json`` or ``.npz`` entry exists for ``key``.
        """
        json_path, npz_path = self._paths(key)
        return json_path.exists() or npz_path.exists()

    def put(self, key: str, value: Any) -> None:
        """Write (or overwrite) the entry for ``key``.

        Args:
            key: Cache key, typically produced by :func:`make_key`.
            value: A :class:`numpy.ndarray`, or any JSON-serializable value.

        Raises:
            TypeError: If ``value`` is neither an array nor JSON-serializable.
            OSError: On filesystem failures.
        """
        json_path, npz_path = self._paths(key)
        if isinstance(value, np.ndarray):
            save_array(npz_path, value)
            json_path.unlink(missing_ok=True)
        else:
            save_json(json_path, value)
            npz_path.unlink(missing_ok=True)

    def load(self, key: str) -> Any:
        """Load the entry for ``key``.

        Args:
            key: Cache key, typically produced by :func:`make_key`.

        Returns:
            The previously cached value.

        Raises:
            CacheError: If no entry exists for ``key``, or the entry on disk
                is corrupt/unreadable.
        """
        json_path, npz_path = self._paths(key)
        try:
            if npz_path.exists():
                return load_array(npz_path)
            if json_path.exists():
                return load_json(json_path)
        except SerializationError as exc:
            raise CacheError(f"corrupt cache entry for key '{key}': {exc}") from exc
        raise CacheError(f"no cache entry for key '{key}'")

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, computing and storing it on a miss.

        Args:
            key: Cache key, typically produced by :func:`make_key`.
            compute_fn: Zero-argument callable producing the value to cache
                on a miss.

        Returns:
            The cached (or freshly computed) value.

        Raises:
            CacheError: If an existing entry is corrupt and
                ``self.force_recompute`` is ``False``.
        """
        if self.force_recompute:
            _logger.info("cache force_recompute key=%s", key)
            value = compute_fn()
            self.put(key, value)
            return value

        if self.exists(key):
            _logger.info("cache hit key=%s", key)
            return self.load(key)

        _logger.info("cache miss key=%s", key)
        value = compute_fn()
        self.put(key, value)
        return value
