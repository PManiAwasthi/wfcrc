"""``MorphologicalSets`` — dilation-margin nested prediction sets for segmentation.

Frozen definition (`Paper 1 - MS2 IMPLEMENTATION SPEC.md`, §C3 item 5):
"Morphological: grow/shrink mask by a monotone structuring element";
`sets.type∈{threshold,morphological}`, `morphological.element`, and a
`direction` (`dilation`/`erosion`) "consistent with the loss's monotonicity"
are named as the concrete config knobs (§C3 item 7), with no further formula
given anywhere in the Research Vault.

**What is implemented (dilation direction).** `score` is taken as a boolean
seed mask `M₀` (of any dimensionality — 2-D for Cityscapes/ADE20K/COCO-style
masks, 3-D for MSD volumes); `λ` is interpreted directly as a non-negative
pixel/voxel radius via `r(λ) = ⌊λ⌋`, and `C_λ = dilate(M₀, r(λ))`, where
`dilate` is standard iterated binary dilation with a fixed structuring
element (`"cross"` = axis-aligned unit steps, i.e. 4-/6-connectivity;
`"square"` = the full `{-1,0,1}^d` neighborhood, i.e. 8-/26-connectivity).
This is textbook morphology (mirroring the FPR-loss gap-fill's provenance
disclosure, `CLAIMS_TRACEABILITY.md` §3): iterated dilation always satisfies
`dilate(M₀, r) ⊆ dilate(M₀, r+1)` because each step ORs the previous mask
into the next, and `⌊λ⌋` is non-decreasing in `λ`, so P-1 nesting holds by
construction — no new mathematics, just the one standard way to realize
"grow a mask by a monotone structuring element" from the frozen phrase.

**What is *not* implemented (erosion direction) — documented gap.** A
literal reading of "erosion" applied to the *same* seed mask `M₀` that
still produces a P-1-nested (growing) family collapses, by the standard
erosion/dilation duality (`erode(A) = complement(dilate(complement(A)))`
for a symmetric structuring element), either to plain dilation of `M₀`
(no distinct behavior) or to a family with the wrong boundary behavior
(non-empty at `λ_min`, growing towards the full array without bound,
rather than the empty/full boundary the MS2 spec's own edge cases expect)
— see the module's non-implementation raising :class:`~wfcrc.exceptions.SetConstructionError`
for the exact reasoning. Constructing a well-behaved, genuinely distinct
erosion-direction family would require additional information (e.g. a
normalizing `λ_max` or a second, independent seed mask) that the frozen
`construct(score, λ)` signature does not carry, and that no vault document
supplies a formula for. Per the "if something appears missing, stop and
document it" rule, this direction is left unimplemented rather than guessed
at; selecting it raises immediately at construction time.
"""

from __future__ import annotations

import itertools
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.exceptions import SetConstructionError
from wfcrc.prediction_sets.base import PredictionSetConstructor

__all__ = ["MorphologicalSets"]

#: Supported structuring-element shapes.
ElementShape = Literal["cross", "square"]
#: Supported growth directions; only "dilation" has a frozen-spec-consistent
#: formula (see module docstring).
Direction = Literal["dilation", "erosion"]


def _neighbor_offsets(ndim: int, element: ElementShape) -> tuple[tuple[int, ...], ...]:
    """Return the non-zero structuring-element offsets for one dilation step.

    Args:
        ndim: Number of array dimensions.
        element: ``"cross"`` for axis-aligned unit offsets (4-/6-connectivity),
            ``"square"`` for the full `{-1,0,1}^ndim` neighborhood minus the
            origin (8-/26-connectivity).

    Returns:
        A tuple of nonzero integer offset tuples, each of length `ndim`.
    """
    if element == "cross":
        offsets = []
        for axis in range(ndim):
            for delta in (-1, 1):
                offset = [0] * ndim
                offset[axis] = delta
                offsets.append(tuple(offset))
        return tuple(offsets)
    return tuple(o for o in itertools.product((-1, 0, 1), repeat=ndim) if any(o))


def _shift(arr: NDArray[np.bool_], offset: tuple[int, ...]) -> NDArray[np.bool_]:
    """Shift `arr` by `offset` (per-axis), filling vacated positions with `False`.

    `result[x] = arr[x - offset]` wherever `x - offset` is in bounds, else
    `False` — i.e. content moves by `offset`, with zero-padding (no
    wraparound, unlike :func:`numpy.roll`).

    Args:
        arr: Boolean array of any dimensionality.
        offset: Per-axis integer shift, same length as `arr.ndim`.

    Returns:
        The shifted boolean array, same shape as `arr`.
    """
    result = np.zeros_like(arr)
    src_slices = []
    dst_slices = []
    for size, off in zip(arr.shape, offset, strict=True):
        if off >= 0:
            src_slices.append(slice(0, size - off) if off > 0 else slice(0, size))
            dst_slices.append(slice(off, size))
        else:
            src_slices.append(slice(-off, size))
            dst_slices.append(slice(0, size + off))
    result[tuple(dst_slices)] = arr[tuple(src_slices)]
    return result


def _dilate(mask: NDArray[np.bool_], radius: int, element: ElementShape) -> NDArray[np.bool_]:
    """Iteratively dilate `mask` by `radius` steps with the given structuring element.

    Args:
        mask: Boolean seed mask, any dimensionality.
        radius: Non-negative number of dilation steps.
        element: Structuring-element shape (see :func:`_neighbor_offsets`).

    Returns:
        The dilated boolean mask, same shape as `mask`. `radius=0` returns
        `mask` unchanged.
    """
    offsets = _neighbor_offsets(mask.ndim, element)
    current = mask
    for _ in range(radius):
        grown = current.copy()
        for offset in offsets:
            grown |= _shift(current, offset)
        current = grown
    return current


class MorphologicalSets(PredictionSetConstructor):
    """Dilation-margin prediction sets: `C_λ = dilate(M₀, ⌊λ⌋)`.

    See the module docstring for the exact construction, its nesting proof,
    and the documented erosion-direction gap.
    """

    def __init__(
        self, *, element: ElementShape = "cross", direction: Direction = "dilation"
    ) -> None:
        """Configure the structuring element and growth direction.

        Args:
            element: ``"cross"`` (axis-aligned, default) or ``"square"``
                (full neighborhood) structuring element.
            direction: Must be ``"dilation"`` — the only direction with a
                frozen-spec-consistent formula (see module docstring).

        Raises:
            SetConstructionError: If `direction == "erosion"` (documented
                specification gap; no formula exists).
            ValueError: If `element` is not one of the supported shapes.
        """
        if element not in ("cross", "square"):
            raise ValueError(f"element must be 'cross' or 'square', got {element!r}")
        if direction != "dilation":
            raise SetConstructionError(
                "MorphologicalSets(direction='erosion') is not implemented: the "
                "frozen MS2 Implementation Spec names 'erosion' as a config "
                "value but gives no formula, and the only literal readings "
                "either collapse to plain dilation (by the standard "
                "erosion/dilation duality) or violate the expected "
                "empty/full-set λ_min/λ_max boundary behavior — see this "
                "module's docstring. Implementing it would mean inventing "
                "behavior beyond the frozen specification."
            )
        self._element: ElementShape = element
        self._direction: Direction = direction

    def construct(self, score: ArrayLike, lam: float) -> NDArray[np.bool_]:
        """Build `C_λ = dilate(M₀, ⌊λ⌋)` from a boolean seed mask `M₀ = score`.

        Args:
            score: Boolean seed mask `M₀`, any dimensionality.
            lam: The threshold `λ`, must be `≥ 0`.

        Returns:
            The dilated boolean mask, same shape as `score`.

        Raises:
            ValueError: If `score` is not a boolean array, or `lam < 0`.
        """
        mask = np.asarray(score)
        if mask.dtype != np.bool_:
            raise ValueError(f"score must have dtype bool, got {mask.dtype}")
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        radius = math.floor(lam)
        return _dilate(mask, radius, self._element)

    def name(self) -> str:
        """Return the registry name ``"morphological"``."""
        return "morphological"
