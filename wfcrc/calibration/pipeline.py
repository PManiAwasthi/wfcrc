"""``run_calibration_pipeline`` — `LossTable` → `WFCRCCalibrator` → optional `Verifier`.

This is the "complete executable calibration pipeline" entry point for a
*pre-built* `LossTable` — dataset/model loading and loss-table construction
remain out of this milestone's scope (Implementation Blueprint §5, "no
experiments/datasets/models"). :meth:`wfcrc.calibration.calibrator.WFCRCCalibrator.calibrate`
is already the frozen "single integration point" (Blueprint §6, §8), so
this module calls it directly rather than re-implementing any of its
branch logic — it only adds the thin wiring to an optional verification
step.

**Why verification is accepted as an injected `VerifierLike`, not imported.**
`wfcrc.evaluation.verifier.Verifier` depends on `wfcrc.calibration`
(`CalibrationResult`, `LossTable`, `Splitter`) to do its job (Implementation
Blueprint §2: `calibration.calibrator ──► verify`, i.e. `verify` consumes
calibration's outputs — the dependency runs *from* `evaluation` *to*
`calibration`). If this module imported `wfcrc.evaluation` back, the
package graph would cycle (`calibration → evaluation → calibration`),
violating the Blueprint's acyclic module graph (§2) — an engineering
constraint this milestone's own verification checklist tests for. Instead,
`run_calibration_pipeline` accepts anything satisfying the local
:class:`VerifierLike` structural protocol; `wfcrc.evaluation.verifier.Verifier`
already satisfies it, so a caller passes `Verifier()` in at the call site
(exactly the composition the Implementation Blueprint's future `runner`
layer performs, §12) without `wfcrc.calibration` ever importing
`wfcrc.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from wfcrc.ambiguity.base import AmbiguityFamily
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig

if TYPE_CHECKING:
    from wfcrc.evaluation.verifier import VerificationReport

__all__ = ["PipelineResult", "VerifierLike", "run_calibration_pipeline"]


class VerifierLike(Protocol):
    """Structural contract a verifier passed to :func:`run_calibration_pipeline` must satisfy.

    :class:`wfcrc.evaluation.verifier.Verifier` already satisfies this
    protocol; it is defined here (rather than imported) so
    `wfcrc.calibration` incurs no import-time dependency on
    `wfcrc.evaluation` — see the module docstring.
    """

    def check_preconditions(
        self, loss_table: LossTable, *, loss_bound: float
    ) -> VerificationReport:
        """See :meth:`wfcrc.evaluation.verifier.Verifier.check_preconditions`."""

    def check_calibration(
        self,
        result: CalibrationResult,
        loss_table: LossTable,
        family: AmbiguityFamily,
        cfg: CalibrationConfig,
        *,
        seed: int,
    ) -> VerificationReport:
        """See :meth:`wfcrc.evaluation.verifier.Verifier.check_calibration`."""


@dataclass(frozen=True)
class PipelineResult:
    """The outcome of one :func:`run_calibration_pipeline` call.

    Attributes:
        calibration: The `CalibrationResult` produced by `WFCRCCalibrator`.
        verification: The merged `VerificationReport` (preconditions +
            calibration checks), or `None` if no `verifier` was supplied.
    """

    calibration: CalibrationResult
    verification: VerificationReport | None

    def assert_ok(self) -> None:
        """Raise if verification ran and any check failed.

        Raises:
            wfcrc.exceptions.VerificationError: If `verification is not
                None` and any check failed.
        """
        if self.verification is not None:
            self.verification.assert_ok()


def run_calibration_pipeline(
    loss_table: LossTable,
    family: AmbiguityFamily,
    cfg: CalibrationConfig,
    *,
    seed: int,
    calibrator: WFCRCCalibrator | None = None,
    verifier: VerifierLike | None = None,
) -> PipelineResult:
    """Compose `LossTable → WFCRCCalibrator.calibrate` (`→` optional `Verifier`).

    Args:
        loss_table: The precomputed `L[i, lambda]` table.
        family: The ambiguity family to calibrate against.
        cfg: Calibration parameters (`alpha`, `B`, `pi`, `lambda_grid`).
        seed: Base seed for the A/B split (dual branch only).
        calibrator: Injected `WFCRCCalibrator`; defaults to a fresh one.
        verifier: An optional object satisfying `VerifierLike` (e.g.
            `wfcrc.evaluation.verifier.Verifier()`); if given, its
            `check_preconditions`/`check_calibration` reports are merged
            into `PipelineResult.verification`. If omitted, no
            verification runs and `PipelineResult.verification` is `None`.

    Returns:
        A `PipelineResult` bundling the `CalibrationResult` and (if
        `verifier` was given) the merged `VerificationReport`.

    Raises:
        ValueError: Propagated from `WFCRCCalibrator.calibrate` (e.g.
            mismatched `lambda_grid`).
        wfcrc.exceptions.FamilyError: Propagated from `WFCRCCalibrator.calibrate`.
    """
    active_calibrator = calibrator if calibrator is not None else WFCRCCalibrator()
    result = active_calibrator.calibrate(loss_table, family, cfg, seed=seed)

    if verifier is None:
        return PipelineResult(calibration=result, verification=None)

    precondition_report = verifier.check_preconditions(loss_table, loss_bound=cfg.B)
    calibration_report = verifier.check_calibration(result, loss_table, family, cfg, seed=seed)
    merged = precondition_report.merge(calibration_report)
    return PipelineResult(calibration=result, verification=merged)
