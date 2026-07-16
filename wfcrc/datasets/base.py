"""``Dataset``/``DatasetLoader`` — abstract data-loading contracts (M3).

Per the Implementation Blueprint (§6, `data.DatasetLoader`) and the MS2
Implementation Specification (§C1): a `DatasetLoader` loads `(id, X, Y)`
triples for a named split and enforces train/calibration/test
disjointness (A1 hygiene). This milestone implements only the abstract
contracts plus the disjointness gate — no concrete loader for any specific
named dataset (Cityscapes, MSD, ...) is built, since none of the
Experiment Blueprint's named datasets are available in this environment;
wiring one in is a later milestone's concern once real data is on hand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from wfcrc.exceptions import SplitLeakageError

__all__ = ["Dataset", "DatasetLoader", "SplitManifest", "assert_split_disjoint"]


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
