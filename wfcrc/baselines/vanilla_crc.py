"""``VanillaCRC`` — the marginal-risk conformal risk control baseline.

Per `docs/EXPERIMENT_PROTOCOL.md` §3/`docs/MODEL_POLICY.md` §1.2 ("Vanilla
CRC / split conformal (LAC): the marginal-risk/coverage reference"), this
is Angelopoulos, Bates, Candès, Jordan & Lei's **Conformal Risk Control**
(arXiv:2208.02814, 2022): the threshold rule

    lambda_hat = inf{ lambda in Lambda : (n/(n+1)) * R_hat(lambda) + B/(n+1) <= alpha }

with no worst-case-over-family adjustment at all — the calibration set is
used once, in full, with the plain empirical mean `R_hat(lambda) =
mean_i L[i, lambda]`.

**Zero duplicated formula code.** This exact bound is already implemented
and frozen, verbatim, inside
:meth:`wfcrc.calibration.calibrator.WFCRCCalibrator._calibrate_known_weight` —
the known-weight branch's `g(lambda) = (n/(n+1)) * [sum_i w_i L[i,lambda] /
sum_i w_i] + B/(n+1)` collapses to *exactly* the plain-CRC formula above
the moment every weight equals `1` (so `sum_i w_i L[i,lambda] / sum_i w_i`
reduces algebraically to the unweighted mean `R_hat(lambda)`, and
`E_P[w] = 1` — the known-weight family's own A4 precondition — holds
trivially for a constant-`1` weight vector). `VanillaCRC` is therefore
implemented as a **provably identical** call into the already-frozen,
already-tested `WFCRCCalibrator` + `KnownWeightFamily(weights=ones(n))`,
not a second implementation of the same bound — avoiding the exact kind of
duplicated-formula drift risk `docs/RESULTS_SCHEMA.md` §7 warns about for
aggregation code.
"""

from __future__ import annotations

from wfcrc.ambiguity.known_weight import KnownWeightFamily
from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig

__all__ = ["VanillaCRC"]


class VanillaCRC(Calibrator):
    """Angelopoulos et al. (2022) Conformal Risk Control, via uniform known-weight CRC."""

    def __init__(self, *, calibrator: WFCRCCalibrator | None = None) -> None:
        """Initialize the baseline.

        Args:
            calibrator: An injected `WFCRCCalibrator`; defaults to a fresh
                instance.
        """
        self._calibrator = calibrator if calibrator is not None else WFCRCCalibrator()

    @property
    def baseline_name(self) -> str:
        """Return ``"vanilla_crc"``."""
        return "vanilla_crc"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Run plain CRC by delegating to the frozen known-weight branch with unit weights.

        Args:
            loss_table: The precomputed calibration `L[i, lambda]` table.
            cfg: Calibration parameters (`pi` is unused — there is no A/B
                split in plain CRC).
            seed: Unused (plain CRC is fully deterministic given
                `loss_table`/`cfg`); accepted only to satisfy the common
                `Calibrator` interface.

        Returns:
            A `CalibrationResult` identical in shape to the known-weight
            branch's own output (`r_hat_b` holds the plain empirical
            risk `R_hat(lambda_hat)`; `diagnostics` carries `n`/`weight_sum`
            exactly as `WFCRCCalibrator._calibrate_known_weight` already
            documents).
        """
        n = loss_table.shape[0]
        uniform_weights = KnownWeightFamily(weights=[1.0] * n)
        return self._calibrator.calibrate(loss_table, uniform_weights, cfg, seed=seed)


BASELINES["vanilla_crc"] = VanillaCRC
