"""``LossTable`` — the minimal data contract calibration consumes (L1a).

Per the Implementation Blueprint (§8, §9), `LossTable` is the shared data
contract between the (out-of-MS2-scope) loss-table builder and
calibration: "the calibrator depends only on `LossTable` + `AmbiguityFamily`
+ scalars — never on data/model modules (dimension-independence enforced
by the interface)". This module implements only that thin, immutable
value contract (`values`, `lambda_grid`, `.column()`, `.row()`, `.shape`,
`.save()`/`.load()`) — **not** `LossTableBuilder`, which assembles a table
from a dataset, model scores, a `PredictionSetConstructor`, and a
`LossEvaluator` (Implementation Blueprint §5-6, `data.LossTableBuilder`).
Building that pipeline is `datasets`/`data.loss_table` scope, explicitly
excluded from this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wfcrc.utils.io import load_json, save_json

__all__ = ["LossTable"]


@dataclass(frozen=True)
class LossTable:
    """An `n x T` table of precomputed losses `L[i, lambda]` over a `λ`-grid.

    Attributes:
        values: `float64` array of shape `(n, T)`; `values[i, j] =
            L[i, lambda_grid[j]]`.
        lambda_grid: `float64` array of shape `(T,)`, strictly increasing.
    """

    values: NDArray[np.float64]
    lambda_grid: NDArray[np.float64]
    _lambda_index: dict[float, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate shape/grid consistency and build the `λ` lookup index.

        Raises:
            ValueError: If `values` is not 2-D, `lambda_grid` is not 1-D,
                their sizes are inconsistent, `lambda_grid` is empty, or
                `lambda_grid` is not strictly increasing.
        """
        values = np.asarray(self.values, dtype=np.float64)
        lambda_grid = np.asarray(self.lambda_grid, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"values must be 2-D (n, T), got shape {values.shape}")
        if lambda_grid.ndim != 1:
            raise ValueError(f"lambda_grid must be 1-D, got shape {lambda_grid.shape}")
        if lambda_grid.size == 0:
            raise ValueError("lambda_grid must be non-empty")
        if values.shape[1] != lambda_grid.size:
            raise ValueError(
                f"values has {values.shape[1]} columns but lambda_grid has "
                f"{lambda_grid.size} entries"
            )
        if not np.all(np.diff(lambda_grid) > 0):
            raise ValueError("lambda_grid must be strictly increasing")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "lambda_grid", lambda_grid)
        object.__setattr__(
            self, "_lambda_index", {float(lam): j for j, lam in enumerate(lambda_grid)}
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return `(n, T)`."""
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    def column(self, lam: float) -> NDArray[np.float64]:
        """Return `L[:, lambda]`, the loss column for one grid point.

        Args:
            lam: A `λ` value that must be exactly one of `self.lambda_grid`'s
                entries.

        Returns:
            The `(n,)` column of losses at `lam`.

        Raises:
            ValueError: If `lam` is not in `self.lambda_grid`.
        """
        try:
            j = self._lambda_index[float(lam)]
        except KeyError as exc:
            raise ValueError(f"lambda={lam!r} is not in this table's lambda_grid") from exc
        return self.values[:, j]

    def row(self, i: int) -> NDArray[np.float64]:
        """Return `L[i, :]`, one example's loss across the whole `λ`-grid.

        Args:
            i: Row (example) index, `0 <= i < n`.

        Returns:
            The `(T,)` row of losses for example `i`.

        Raises:
            IndexError: If `i` is out of range.
        """
        return self.values[i, :]

    def save(self, path: str | Path) -> None:
        """Atomically persist this table as JSON (via :mod:`wfcrc.utils.io`).

        Args:
            path: Destination file path.

        Raises:
            OSError: On filesystem failures.
        """
        save_json(path, {"values": self.values, "lambda_grid": self.lambda_grid})

    @classmethod
    def load(cls, path: str | Path) -> LossTable:
        """Load a table previously written by :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            The reconstructed :class:`LossTable`.

        Raises:
            wfcrc.exceptions.SerializationError: If the file is missing,
                unreadable, or not valid JSON.
            ValueError: If the loaded content fails :class:`LossTable`
                validation.
        """
        data = load_json(path)
        return cls(values=data["values"], lambda_grid=data["lambda_grid"])
