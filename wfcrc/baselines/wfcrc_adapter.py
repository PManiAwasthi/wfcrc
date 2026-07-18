"""``WFCRCAdapter`` — WF-CRC itself, exposed through the common `Calibrator` interface.

Per `docs/EXPERIMENT_PROTOCOL.md` §9's own requirement ("every baseline
... behind the same interface as WF-CRC"): this module does not
reimplement anything — it is a thin, one-method adapter around the frozen
:class:`~wfcrc.calibration.calibrator.WFCRCCalibrator` (MS2), reordering
its `(loss_table, family, cfg, seed)` call into the common `Calibrator`
shape by moving `family` to construction time (exactly like every other
baseline in this package already does). This is what lets WF-CRC and every
comparator baseline sit in the same `BASELINES` registry and be driven by
the same calling code with zero `isinstance` branching.
"""

from __future__ import annotations

from wfcrc.ambiguity.base import AmbiguityFamily
from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig

__all__ = ["WFCRCAdapter"]


class WFCRCAdapter(Calibrator):
    """Adapts the frozen `WFCRCCalibrator` to the common `Calibrator` interface.

    Attributes:
        family: The ambiguity family WF-CRC calibrates against
            (`cvar`/`kl`/`finite_group`/`known_weight`).
    """

    def __init__(
        self, family: AmbiguityFamily, *, calibrator: WFCRCCalibrator | None = None
    ) -> None:
        """Initialize the adapter.

        Args:
            family: The ambiguity family to pass through to the frozen
                `WFCRCCalibrator.calibrate`, unchanged.
            calibrator: An injected `WFCRCCalibrator`; defaults to a fresh
                instance (matching that class's own default-collaborator
                pattern).
        """
        self.family = family
        self._calibrator = calibrator if calibrator is not None else WFCRCCalibrator()

    @property
    def baseline_name(self) -> str:
        """Return ``"wfcrc"`` (WF-CRC is not itself a "baseline" in the
        comparator sense, but shares the same registry/interface so the
        experiment/evaluation layer never special-cases it)."""
        return "wfcrc"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Delegate to the frozen `WFCRCCalibrator.calibrate`, unchanged.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters.
            seed: Base seed for the A/B split (dual branch only).

        Returns:
            The `CalibrationResult` `WFCRCCalibrator.calibrate` returns,
            passed through verbatim.
        """
        return self._calibrator.calibrate(loss_table, self.family, cfg, seed=seed)


BASELINES["wfcrc"] = WFCRCAdapter
