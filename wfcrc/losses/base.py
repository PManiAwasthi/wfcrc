"""``LossEvaluator`` — the abstract loss contract WF-CRC calibration depends on.

Per the Algorithm Specification (§2, §5, P-2) and the Implementation
Blueprint (§6), every loss is a function `l(set, label) ∈ (-∞, B]` that,
paired with the correct nested set family, is non-increasing in the
threshold `λ`. This module fixes only that contract; it does not implement
or depend on any particular prediction-set constructor (that pairing is the
responsibility of a later milestone's `sets`/`prediction_sets` module and
its integration tests — see the module docstring in
:mod:`wfcrc.losses.fnr`/`.fpr`/`.miscoverage` for the pairing each concrete
loss requires).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["LossEvaluator"]


class LossEvaluator(ABC):
    """Abstract base class for a bounded, monotone task loss `l(set, label)`.

    Concrete subclasses must implement :meth:`evaluate`, :meth:`upper_bound`,
    and :meth:`name`. :meth:`assert_monotone` is a shared, concrete
    contract-check usable by any subclass.
    """

    @abstractmethod
    def evaluate(self, predicted_set: NDArray[np.bool_], label: NDArray[np.bool_]) -> float:
        """Compute `l(predicted_set, label)`.

        Args:
            predicted_set: Boolean array representing `C_λ(X)` — the
                predicted set at some threshold `λ` (e.g. a per-pixel
                segmentation mask, or a per-class inclusion indicator).
                Arbitrary shape; dimension-agnostic.
            label: Boolean array representing the ground truth `Y`, the
                same shape as `predicted_set`.

        Returns:
            The scalar loss, guaranteed `≤ self.upper_bound()`.

        Raises:
            ValueError: If `predicted_set`/`label` are not boolean arrays,
                or their shapes do not match.
        """

    @abstractmethod
    def upper_bound(self) -> float:
        """Return `B`, the loss's finite upper bound.

        Returns:
            The upper bound `B` such that `evaluate(...) <= B` always holds.
        """

    @abstractmethod
    def name(self) -> str:
        """Return this loss's short registry name (e.g. ``"fnr"``).

        Returns:
            A short, stable, lowercase identifier for this loss.
        """

    @staticmethod
    def _validate_shapes(predicted_set: NDArray[np.bool_], label: NDArray[np.bool_]) -> None:
        """Validate that both arrays are boolean and identically shaped.

        Args:
            predicted_set: Candidate predicted-set array.
            label: Candidate label array.

        Raises:
            ValueError: If either array is not boolean dtype, or their
                shapes differ.
        """
        if predicted_set.dtype != np.bool_:
            raise ValueError(f"predicted_set must have dtype bool, got {predicted_set.dtype}")
        if label.dtype != np.bool_:
            raise ValueError(f"label must have dtype bool, got {label.dtype}")
        if predicted_set.shape != label.shape:
            raise ValueError(
                f"shape mismatch: predicted_set.shape={predicted_set.shape} "
                f"!= label.shape={label.shape}"
            )

    def assert_monotone(self, losses_by_lambda: Sequence[float], *, tol: float = 1e-9) -> bool:
        """Check that a `λ`-ordered sequence of loss values is non-increasing.

        This is the P-2 contract check (Algorithm Specification §5, §20):
        given `[l(C_{λ_1}(X),Y), l(C_{λ_2}(X),Y), ...]` for an increasing
        `λ`-grid `λ_1 < λ_2 < ...` and a *compatible* nested set family
        (see each concrete loss's module docstring for which λ-direction of
        set growth it requires), this must be non-increasing.

        This method does not itself construct or require a set family —
        it only checks a precomputed numeric sequence — so it has no
        dependency on any `PredictionSetConstructor`.

        Args:
            losses_by_lambda: Loss values in increasing-`λ` order.
            tol: Numerical tolerance; an increase of at most `tol` between
                consecutive entries is still treated as non-increasing
                (guards against floating-point noise, not against a real
                violation).

        Returns:
            ``True`` if the sequence is non-increasing within `tol`,
            ``False`` otherwise.

        Raises:
            ValueError: If `losses_by_lambda` is empty.
        """
        arr = np.asarray(losses_by_lambda, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("losses_by_lambda must be non-empty")
        if arr.size == 1:
            return True
        return bool(np.all(np.diff(arr) <= tol))
