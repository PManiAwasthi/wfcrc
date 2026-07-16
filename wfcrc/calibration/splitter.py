"""``Splitter`` — the seeded, disjoint A/B calibration partition.

Per the Algorithm Specification (§7 step 2, §17): "Partition `{1..n}` into
`A` (size `n_A = ceil(pi*n)`) and `B` (size `n_B = n - n_A`), disjoint" —
**the only stochastic step** in the entire single-split procedure. "A fixed
seed yields a fully deterministic, reproducible procedure."
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from wfcrc.utils.seeds import derive_seed

__all__ = ["Splitter"]

#: Component name used to derive this splitter's RNG seed
#: (:func:`wfcrc.utils.seeds.derive_seed`).
_SEED_COMPONENT = "calibration.split"


class Splitter:
    """Seeded, disjoint A/B partition of `{0, ..., n-1}`."""

    def split(self, n: int, pi: float, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Partition `{0, ..., n-1}` into disjoint, sorted index arrays `(A, B)`.

        Args:
            n: Number of calibration examples, `n >= 2`.
            pi: Fraction of examples assigned to block `A`
                (dual-estimation block), `0 < pi < 1`.
            seed: Base seed; the actual RNG seed is derived from
                `(seed, "calibration.split")` via
                :func:`wfcrc.utils.seeds.derive_seed`, so the same `seed`
                always yields the same partition.

        Returns:
            `(a_idx, b_idx)`: disjoint, sorted `int64` index arrays with
            `len(a_idx) = ceil(pi * n)` and `len(a_idx) + len(b_idx) = n`.

        Raises:
            ValueError: If `n < 2`, `pi` is outside `(0, 1)`, or the
                resulting split would leave `A` or `B` empty.
        """
        if n < 2:
            raise ValueError(f"n must be >= 2 to form a non-empty A/B split, got {n}")
        if not (0.0 < pi < 1.0):
            raise ValueError(f"pi must be in (0, 1), got {pi}")

        n_a = math.ceil(pi * n)
        if n_a < 1 or n_a >= n:
            raise ValueError(
                f"split (n={n}, pi={pi}) yields n_A={n_a}, which leaves A or B "
                f"empty; choose a pi that yields 1 <= n_A <= n-1"
            )

        derived_seed = derive_seed(_SEED_COMPONENT, seed)
        rng = np.random.default_rng(derived_seed)
        perm = rng.permutation(n)
        a_idx = np.sort(perm[:n_a]).astype(np.int64)
        b_idx = np.sort(perm[n_a:]).astype(np.int64)
        return a_idx, b_idx
