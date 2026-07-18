"""``Calibrator`` — the common external interface every comparator baseline shares with WF-CRC.

Per `docs/EXPERIMENT_PROTOCOL.md` §9 (MS9, "every baseline referenced by
E1-E12 must exist behind the same interface as WF-CRC so that experiments
become configuration changes rather than engineering work"): a single
abstract method, matching :meth:`wfcrc.calibration.calibrator.
WFCRCCalibrator.calibrate`'s own external shape exactly —

    calibrate(loss_table: LossTable, cfg: CalibrationConfig, *, seed: int) -> CalibrationResult

— minus the frozen calibrator's `family` parameter, which every
:class:`Calibrator` implementation instead takes at **construction** time
(mirroring how :class:`~wfcrc.ambiguity.cvar.CVaRFamily`/
:class:`~wfcrc.ambiguity.kl.KLFamily` already take `beta`/`rho` at
construction, not at call time). This is what lets a caller — the
evaluation/runner layer, or a future config-driven experiment resolver —
hold a list of `Calibrator` instances (one WF-CRC family, several
comparator baselines) and call every one of them identically:

    for calibrator in calibrators:
        result = calibrator.calibrate(loss_table, cfg, seed=seed)

with no `isinstance`/baseline-specific branch anywhere in that loop. Every
concrete `Calibrator` returns the same frozen, unmodified
:class:`~wfcrc.calibration.calibrator.CalibrationResult` (MS2), so every
downstream consumer — :mod:`wfcrc.evaluation.metrics`,
:func:`wfcrc.evaluation.experiment.run_experiment`,
:class:`wfcrc.runner.runner.ExperimentRunner` — already works unchanged;
no frozen MS1-MS8 file is modified by this module or any baseline built on
it.

**Where a baseline's "true" decision rule cannot be summarized by one
scalar `lambda_hat`** (e.g. :class:`~wfcrc.baselines.group_conditional.
GroupConditionalCRC`'s per-group thresholds), the full rule is recorded in
`CalibrationResult.diagnostics` (already a free-form field, per its own
frozen docstring: "branch-specific extra detail") and `lambda_hat` carries
a documented, conservative summary value — the same disclosed-adaptation
pattern already used elsewhere in this project (e.g. `HippocampusScoreProvider`'s
narrow `# type: ignore` against a frozen contract, MS7) rather than a
change to the frozen `CalibrationResult` shape.

**Registry.** :data:`BASELINES` maps a short, stable name to a
`Calibrator` subclass, mirroring the `FAMILIES`/`DATASETS`/`MODELS`
name-keyed registry pattern already established in
:mod:`wfcrc.ambiguity`/`wfcrc.datasets.registry`/`wfcrc.models.registry` —
a plain module-level `dict`, no dynamic registration API, no metaclass
magic, populated additively by each baseline's own module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig

__all__ = ["BASELINES", "Calibrator"]


class Calibrator(ABC):
    """Common external interface shared by WF-CRC and every comparator baseline.

    A concrete `Calibrator` takes its own hyperparameters (family
    instance, group masks, KL radius, fixed dual parameter, ...) as
    constructor arguments; the only method every caller needs is
    :meth:`calibrate`, whose signature is identical across every
    implementation — this is the whole point of the interface (no
    baseline-specific branch in the runner/evaluation layer).
    """

    @property
    @abstractmethod
    def baseline_name(self) -> str:
        """Return this calibrator's stable, registry-keyed name.

        Returns:
            A short, stable string identifier (e.g. `"wfcrc"`,
            `"vanilla_crc"`, `"lac"`) — matches the key this class is
            registered under in :data:`BASELINES`, and is recorded
            downstream (e.g. as the `family` column in an aggregated
            results table, `docs/RESULTS_SCHEMA.md` §2.2) so a reported
            number is always traceable back to which calibrator produced
            it.
        """

    @abstractmethod
    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Run this calibrator's procedure and return a `CalibrationResult`.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters (`alpha`, `B`, `pi`, `lambda_grid`).
                Not every field is meaningful to every baseline (e.g. `pi`
                is unused by a baseline with no A/B split); an unused field
                is simply ignored, never rejected, so the same `cfg` object
                can drive every registered calibrator uniformly.
            seed: Base seed for any stochastic step this calibrator
                performs (e.g. an A/B split, a K-fold partition); ignored
                by fully deterministic baselines, exactly like the frozen
                `WFCRCCalibrator.calibrate`'s own `seed` parameter is
                ignored by its finite-group/known-weight branches.

        Returns:
            The resulting `CalibrationResult` — the same frozen type
            `WFCRCCalibrator.calibrate` returns, so every downstream
            metrics/evaluation/runner consumer works unchanged.
        """


#: Name -> concrete `Calibrator` subclass, populated additively by each
#: baseline module's own import (mirrors `wfcrc.ambiguity.FAMILIES`).
BASELINES: dict[str, type[Calibrator]] = {}
