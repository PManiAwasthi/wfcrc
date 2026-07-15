"""Canonical content hashing and crash-safe serialization.

This is the root utility that :mod:`wfcrc.utils.cache` and :mod:`wfcrc.config`
build on. Two properties are load-bearing for the rest of the project:

1. **Hash stability**: :func:`content_hash` produces an identical digest for
   semantically identical inputs regardless of dict key insertion order,
   process, or platform (given the same object graph).
2. **Atomicity**: :func:`atomic_write` never leaves a partially written file
   visible at the destination path, even if the process is killed mid-write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from wfcrc.constants import ATOMIC_TMP_SUFFIX, DEFAULT_HASH_ALGO, DEFAULT_HASH_WIDTH, TEXT_ENCODING
from wfcrc.exceptions import SerializationError

_logger = logging.getLogger(__name__)

__all__ = [
    "atomic_write",
    "content_hash",
    "ensure_dir",
    "load_array",
    "load_json",
    "save_array",
    "save_json",
]

_NDARRAY_TAG = "__ndarray__"


def _json_default(obj: Any) -> Any:
    """Encode numpy scalars/arrays for :func:`json.dumps`.

    Args:
        obj: An object that the default JSON encoder could not serialize.

    Returns:
        A JSON-serializable representation of ``obj``.

    Raises:
        TypeError: If ``obj`` is not a supported numpy type or otherwise
            JSON-serializable. This is the standard protocol expected by
            ``json.dumps(..., default=...)``.
    """
    if isinstance(obj, np.ndarray):
        return {
            _NDARRAY_TAG: True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tolist(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_object_hook(obj: dict[str, Any]) -> Any:
    """Decode ``{_NDARRAY_TAG: True, ...}`` payloads back into numpy arrays."""
    if obj.get(_NDARRAY_TAG) is True:
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def _canonical_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` to a canonical, deterministic byte string.

    Canonicalization means: dict keys sorted, no whitespace, fixed
    (round-trip-safe) float formatting via the standard ``repr``-based
    ``json`` float encoder, and ASCII-only output — so two Python processes
    on different platforms produce byte-identical output for equal objects.

    Args:
        obj: A JSON-serializable object (dict/list/scalars/numpy).

    Returns:
        The canonical UTF-8 encoded JSON byte string.

    Raises:
        TypeError: If ``obj`` (or a nested value) is not serializable.
    """
    text = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return text.encode(TEXT_ENCODING)


def content_hash(
    obj: Any, *, algo: str = DEFAULT_HASH_ALGO, width: int = DEFAULT_HASH_WIDTH
) -> str:
    """Compute a stable content hash of ``obj``.

    The hash is invariant to dict key ordering and identical across
    processes/platforms for semantically equal objects.

    Args:
        obj: A JSON-serializable object (dict/list/scalars/numpy scalars or
            arrays, arbitrarily nested).
        algo: Hash algorithm name accepted by :func:`hashlib.new`.
        width: Number of leading hex characters to return from the full
            digest. Use the full digest length (or a larger width) if
            collision resistance at scale matters more than key brevity.

    Returns:
        A ``width``-character lowercase hex digest string.

    Raises:
        TypeError: If ``obj`` is not serializable.
    """
    digest = hashlib.new(algo, _canonical_bytes(obj)).hexdigest()
    return digest[:width]


def atomic_write(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    Writes to a temporary file in the same directory as ``path`` and then
    renames it into place, so a reader never observes a partially written
    file and a crash mid-write leaves the original file (if any) untouched.

    Args:
        path: Destination file path.
        data: Raw bytes to write.

    Raises:
        OSError: On permission errors or other filesystem failures.
    """
    target = Path(path)
    overwriting = target.exists()
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(
        suffix=ATOMIC_TMP_SUFFIX, prefix=target.name + ".", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    if overwriting:
        _logger.warning("overwriting existing file: %s", target)
    else:
        _logger.debug("wrote file: %s (%d bytes)", target, len(data))


def save_json(path: str | os.PathLike[str], obj: Any) -> None:
    """Atomically write ``obj`` to ``path`` as canonical JSON.

    Args:
        path: Destination ``.json`` file path.
        obj: A JSON-serializable object (dict/list/scalars/numpy).

    Raises:
        TypeError: If ``obj`` is not serializable.
        OSError: On filesystem failures.
    """
    atomic_write(path, _canonical_bytes(obj))


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load a JSON object previously written by :func:`save_json`.

    Args:
        path: Source ``.json`` file path.

    Returns:
        The decoded object, with numpy arrays reconstructed from their
        tagged representation.

    Raises:
        SerializationError: If the file is missing, unreadable, or not
            valid JSON.
    """
    try:
        text = Path(path).read_text(encoding=TEXT_ENCODING)
        return json.loads(text, object_hook=_json_object_hook)
    except (OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"failed to load JSON from '{path}': {exc}") from exc


def save_array(path: str | os.PathLike[str], arr: np.ndarray) -> None:
    """Atomically write a single numpy array to ``path`` as ``.npz``.

    Args:
        path: Destination file path (conventionally ``*.npz``).
        arr: The array to persist.

    Raises:
        OSError: On filesystem failures.
    """
    import io as _io

    buffer = _io.BytesIO()
    np.savez(buffer, arr=arr)
    atomic_write(path, buffer.getvalue())


def load_array(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a single numpy array previously written by :func:`save_array`.

    Args:
        path: Source ``.npz`` file path.

    Returns:
        The stored array.

    Raises:
        SerializationError: If the file is missing, unreadable, or not a
            valid archive produced by :func:`save_array`.
    """
    try:
        with np.load(path) as data:
            return np.asarray(data["arr"])
    except (OSError, ValueError, KeyError) as exc:
        raise SerializationError(f"failed to load array from '{path}': {exc}") from exc


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and any missing parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The directory path as a :class:`pathlib.Path`.

    Raises:
        OSError: If the directory cannot be created (e.g. permission denied,
            or a file already exists at that path).
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
