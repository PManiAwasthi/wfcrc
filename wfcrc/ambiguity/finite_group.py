"""Finite-group ambiguity family.

Frozen alternative branch (Algorithm Specification §7', "Finite-group
family `{G}`"): no dual, no A/B split — the calibrator runs standard
conformal risk control independently on each group's calibration
subsample and deploys the max over groups. This module only encodes the
group membership structure `{G}`; the per-group CRC computation itself is
`wfcrc.calibration`'s responsibility (it needs the loss table, which this
family does not hold).

Group membership uses the same representation
:mod:`wfcrc.config.schema`'s `FamilyConfig.masks` already validates —
per-group tuples of integer row indices into the calibration set — rather
than boolean masks, so a family built from a loaded `Config` requires no
format conversion.
"""

from __future__ import annotations

from collections.abc import Sequence

from wfcrc.ambiguity.base import AmbiguityFamily
from wfcrc.config.schema import FamilyType
from wfcrc.exceptions import FamilyError

__all__ = ["FiniteGroupFamily"]


class FiniteGroupFamily(AmbiguityFamily):
    """Finite-group ambiguity family `{G}`.

    Attributes:
        masks: Group membership, as a tuple of per-group tuples of
            (non-negative) integer row indices into the calibration set.
    """

    def __init__(self, masks: Sequence[Sequence[int]]) -> None:
        """Initialize the finite-group family.

        Args:
            masks: One index sequence per group; each must be non-empty.

        Raises:
            FamilyError: If `masks` is empty, any group is empty, or any
                index is negative.
        """
        if len(masks) == 0:
            raise FamilyError("finite-group family requires at least one group")
        groups: list[tuple[int, ...]] = []
        for i, group in enumerate(masks):
            if len(group) == 0:
                raise FamilyError(f"group {i} is empty; every group must be non-empty")
            indices = tuple(int(idx) for idx in group)
            if any(idx < 0 for idx in indices):
                raise FamilyError(f"group {i} contains a negative index")
            groups.append(indices)
        self.masks: tuple[tuple[int, ...], ...] = tuple(groups)

    @property
    def family_type(self) -> FamilyType:
        """Return ``"finite_group"``."""
        return "finite_group"

    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Return the group membership index tuples.

        Returns:
            One tuple of row indices per group, in construction order.
        """
        return self.masks
