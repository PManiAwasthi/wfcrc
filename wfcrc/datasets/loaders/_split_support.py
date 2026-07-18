"""Shared split-manifest handling for concrete ``DatasetLoader`` families (DI-2).

Every concrete loader in this package (MSD/NIfTI, ACDC, Kvasir, CIFAR)
consumes an **externally supplied** train/calibration/test split assignment
and must *never* invent a ratio or seed itself
(`docs/DATASET_INTEGRATION_GUIDE.md` §7; `docs/DATASET_SPLIT_POLICY.md`).
The mechanical work of reading that assignment (a mapping or a JSON file),
shape-validating it, checking every id resolves to a discovered example,
and handing the three id lists to the frozen
:class:`~wfcrc.datasets.base.SplitManifest` (which enforces A1 hygiene) is
**identical** across all four families.

DI-1 first implemented this logic privately inside
:class:`~wfcrc.datasets.loaders.msd.MSDNiftiLoader`. DI-2 lifts it here,
unchanged in behavior, so the three new loader families reuse one
implementation instead of copy-pasting it four times (the DI-2 self-audit's
explicit "move common functionality into shared utilities; do not repeat
validation code" directive). ``MSDNiftiLoader`` is refactored to call these
helpers; its public interface, failure modes, and 57-test behavior are
unchanged (verified by that unchanged suite).

This module introduces **no new abstraction** over the frozen
``Dataset``/``DatasetLoader``/``SplitManifest`` architecture — it only
factors out an implementation detail those contracts already required every
loader to perform.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from wfcrc.datasets.base import SplitManifest
from wfcrc.exceptions import SerializationError

__all__ = [
    "MANIFEST_FIELD",
    "SPLIT_NAMES",
    "build_manifest",
    "read_split_manifest",
    "validate_manifest_ids",
]

#: The three split names every loader (and the frozen ``SplitManifest``) recognizes.
SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "test")

#: ``SplitManifest`` field name for each split name above.
MANIFEST_FIELD: dict[str, str] = {
    "train": "train_ids",
    "calibration": "cal_ids",
    "test": "test_ids",
}


def read_split_manifest(
    source: Mapping[str, Sequence[Hashable]] | str | Path,
) -> dict[str, list[str]]:
    """Load and shape-validate a ``split_manifest`` (mapping or JSON file path).

    Ids are coerced to ``str`` so a caller's numeric/other id types compare
    consistently against a loader's own string ids.

    Args:
        source: Either a mapping ``{"train": [...], "calibration": [...],
            "test": [...]}`` of ids, or a path to a JSON file with that
            exact shape.

    Returns:
        A dict with exactly the keys :data:`SPLIT_NAMES`, each mapping to a
        list of string ids.

    Raises:
        ValueError: If ``source`` is not the required mapping shape, a
            manifest file path does not exist, or a required split name is
            missing / an unrecognized split name is present.
        SerializationError: If a manifest *file* cannot be read or parsed.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"split manifest file not found: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                raw: Any = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SerializationError(f"could not read/parse split manifest {path}: {exc}") from exc
    else:
        raw = source
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"split_manifest must be a mapping of {list(SPLIT_NAMES)} to id lists "
            f"(or a path to a JSON file with that shape), got {type(raw)}"
        )
    missing = [name for name in SPLIT_NAMES if name not in raw]
    if missing:
        raise ValueError(f"split_manifest is missing required split(s): {missing}")
    extra = sorted(set(raw) - set(SPLIT_NAMES))
    if extra:
        raise ValueError(f"split_manifest has unrecognized split name(s): {extra}")
    return {name: [str(i) for i in raw[name]] for name in SPLIT_NAMES}


def validate_manifest_ids(
    manifest_dict: Mapping[str, Sequence[str]],
    known_ids: Sequence[Hashable],
    *,
    pool_description: str,
) -> None:
    """Ensure every ``split_manifest`` id resolves to a discovered example.

    Args:
        manifest_dict: The shape-validated manifest from
            :func:`read_split_manifest`.
        known_ids: Every id this loader discovered (its full example pool).
        pool_description: A short human-readable name for the pool, used in
            the error message (e.g. ``"Task07_Pancreas training pool"`` or
            ``"ACDC labelled (train+val) pool"``).

    Raises:
        ValueError: If any manifest id is not in ``known_ids``.
    """
    known = {str(i) for i in known_ids}
    for split_name, ids in manifest_dict.items():
        unknown = sorted(set(ids) - known)
        if unknown:
            raise ValueError(
                f"split_manifest '{split_name}' references id(s) not present in the "
                f"discovered {pool_description}: {unknown}"
            )


def build_manifest(manifest_dict: Mapping[str, Sequence[str]]) -> SplitManifest:
    """Build the frozen :class:`~wfcrc.datasets.base.SplitManifest` (A1 hygiene gate).

    Args:
        manifest_dict: The shape-validated manifest from
            :func:`read_split_manifest`.

    Returns:
        A :class:`~wfcrc.datasets.base.SplitManifest`, whose construction
        enforces train/calibration/test disjointness.

    Raises:
        SplitLeakageError: If the three id lists are not pairwise disjoint
            (raised by ``SplitManifest`` itself, unchanged).
    """
    return SplitManifest(
        train_ids=tuple(manifest_dict["train"]),
        cal_ids=tuple(manifest_dict["calibration"]),
        test_ids=tuple(manifest_dict["test"]),
    )
