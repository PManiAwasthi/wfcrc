"""``Dataset``/``DatasetLoader`` — abstract data-loading contracts (M3).

Per the Implementation Blueprint (§6, `data.DatasetLoader`) and the MS2
Implementation Specification (§C1): a `DatasetLoader` loads `(id, X, Y)`
triples for a named split and enforces train/calibration/test
disjointness (A1 hygiene). This milestone implements only the abstract
contracts plus the disjointness gate — no concrete loader for any specific
named dataset (Cityscapes, MSD, ...) is built, since none of the
Experiment Blueprint's named datasets are available in this environment;
wiring one in is a later milestone's concern once real data is on hand.

**DI-1 addition (additive, non-breaking — no change to `Dataset`/
`DatasetLoader`'s existing abstract methods):** :class:`IntegrityIssue`/
:class:`IntegrityReport` are shared, reusable value types for a concrete
`Dataset`'s own optional, concrete `verify_integrity()`-style method (first
adopted by :class:`~wfcrc.datasets.loaders.msd.MSDDataset`, per
`docs/DATASET_INTEGRATION_GUIDE.md`). They are deliberately **not** added
as a new abstract method on `Dataset` itself — doing so would force every
future concrete subclass to implement one, which is an interface
redesign this project's "additive, backwards-compatible only" discipline
does not permit without explicit authorization. Any concrete loader may
adopt this pattern by returning an `IntegrityReport` from its own
same-named method; nothing in `wfcrc.calibration`/`wfcrc.evaluation`
depends on it existing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from wfcrc.exceptions import SerializationError, SplitLeakageError

__all__ = [
    "Dataset",
    "DatasetLoader",
    "IntegrityIssue",
    "IntegrityReport",
    "SplitManifest",
    "assert_split_disjoint",
]


class Dataset(ABC):
    """Abstract, lazily-iterable `(id, X, Y)` collection for one dataset split."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        """Yield `(id, X, Y)` triples for every example in this split."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of examples in this split."""

    @abstractmethod
    def ids(self) -> Sequence[Hashable]:
        """Return the immutable, ordered sequence of example ids in this split."""

    @abstractmethod
    def labels(self, id_: Hashable) -> Any:
        """Return the label `Y` for a given example id.

        Args:
            id_: An id previously returned by :meth:`ids`.
        """

    @abstractmethod
    def meta(self) -> dict[str, Any]:
        """Return dataset provenance metadata (at least `version`, `license`)."""


class DatasetLoader(ABC):
    """Abstract loader producing a :class:`Dataset` for a named split."""

    @abstractmethod
    def load(self, split_name: str) -> Dataset:
        """Load the named split (e.g. `"train"`, `"calibration"`, `"test"`).

        Args:
            split_name: Name of the split to load.

        Returns:
            A :class:`Dataset` for that split.

        Raises:
            ValueError: If `split_name` is not a recognized split for this
                loader.
        """


def assert_split_disjoint(
    train_ids: Sequence[Hashable],
    cal_ids: Sequence[Hashable],
    test_ids: Sequence[Hashable],
) -> None:
    """Assert `train ∩ calibration ∩ test = ∅` (A1 hygiene gate).

    Args:
        train_ids: Ids assigned to the training split.
        cal_ids: Ids assigned to the calibration split.
        test_ids: Ids assigned to the test split.

    Raises:
        SplitLeakageError: If any two of the three id sequences share an id.
    """
    train_set, cal_set, test_set = set(train_ids), set(cal_ids), set(test_ids)
    overlaps = {
        "train/calibration": train_set & cal_set,
        "train/test": train_set & test_set,
        "calibration/test": cal_set & test_set,
    }
    leaking = {pair: ids for pair, ids in overlaps.items() if ids}
    if leaking:
        detail = "; ".join(f"{pair}: {sorted(map(str, ids))}" for pair, ids in leaking.items())
        raise SplitLeakageError(f"overlapping split ids detected ({detail})")


@dataclass(frozen=True)
class SplitManifest:
    """Immutable record of which ids belong to which split.

    Attributes:
        train_ids: Ids assigned to the training split.
        cal_ids: Ids assigned to the calibration split.
        test_ids: Ids assigned to the test split.

    Raises:
        SplitLeakageError: On construction, if any two splits overlap
            (enforced via :func:`assert_split_disjoint`).
    """

    train_ids: tuple[Hashable, ...]
    cal_ids: tuple[Hashable, ...]
    test_ids: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        """Coerce id sequences to tuples and enforce the A1 hygiene gate."""
        object.__setattr__(self, "train_ids", tuple(self.train_ids))
        object.__setattr__(self, "cal_ids", tuple(self.cal_ids))
        object.__setattr__(self, "test_ids", tuple(self.test_ids))
        assert_split_disjoint(self.train_ids, self.cal_ids, self.test_ids)


@dataclass(frozen=True)
class IntegrityIssue:
    """One concrete, discrete data-integrity problem found in a `Dataset`.

    Attributes:
        id_: The example id the problem was found in.
        problem: A short, human-readable description of the problem
            (e.g. "image/label shape mismatch: (3,4,5) vs (3,4,6)").
    """

    id_: Hashable
    problem: str


@dataclass(frozen=True)
class IntegrityReport:
    """Aggregate outcome of a `Dataset`'s own `verify_integrity()`-style check.

    Mirrors the shape of `wfcrc.evaluation.verifier.VerificationReport`
    deliberately (collect every issue, non-strict, with an explicit
    strict-gate method) — the same "always run every check, then let the
    caller decide how strict to be" pattern already established there,
    reused here rather than inventing a second convention.

    Attributes:
        issues: Every integrity problem found, in the order checked. Empty
            if the dataset is fully intact.
    """

    issues: tuple[IntegrityIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Return `True` iff no issue was found."""
        return len(self.issues) == 0

    def assert_ok(self) -> None:
        """Raise if any issue was found (the strict gate).

        Raises:
            SerializationError: Naming every id/problem pair, if
                :attr:`ok` is `False`. Reuses the frozen `SerializationError`
                (a data-content problem) rather than adding a new
                exception type for this additive, dataset-side check.
        """
        if not self.ok:
            detail = "; ".join(f"{issue.id_!r}: {issue.problem}" for issue in self.issues)
            raise SerializationError(f"dataset integrity check failed: {detail}")
