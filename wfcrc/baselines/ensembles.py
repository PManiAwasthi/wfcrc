"""MC-dropout and deep-ensemble score aggregation.

Per `docs/EXPERIMENT_PROTOCOL.md`/`docs/MODEL_POLICY.md` §1.2 ("deep
ensembles; MC-dropout: standard, non-guaranteed UQ"): both methods produce
`K` **stochastic scores per example** (`K` dropout-enabled forward passes,
or `K` independently trained ensemble members) rather than the single
deterministic score every other `wfcrc.baselines`/`wfcrc.datasets`
`ScoreProvider` produces. Aggregating that `(K, n, ...)` stack into one
effective `(n, ...)` score (so a `LossTableBuilder` can consume it exactly
like any other `ScoreProvider` output) is a **score-production-time**
operation — upstream of the `LossTable`/`Calibrator` boundary, the same
architectural note `wfcrc.baselines.scaling`'s module docstring already
makes for temperature/selective scaling. This module is therefore the same
shape as that one: plain, `LossTable`-independent NumPy aggregation
utilities, testable against synthetic `(K, n, ...)` score stacks with no
real model or multi-pass inference (real integration is deferred until a
concrete `ScoreProvider` capable of `K`-pass stochastic inference exists),
plus one thin downstream `Calibrator` wrapper reusing
`wfcrc.baselines.lac.SplitConformalLAC`'s order-statistic rule over
whatever `LossTable` was built from the aggregated score.

**Deep ensembles and MC-dropout share identical aggregation arithmetic**
(mean and variance across the `K` axis) but are kept as two distinctly
named functions, not one, because the Experiment Blueprint lists them as
two separate comparator baselines (different `K`-samples source — trained
ensemble members vs. dropout-enabled forward passes of one network) whose
identity must remain distinguishable in a results table
(`docs/RESULTS_SCHEMA.md` §2.2's `family` column), even though the
arithmetic they wrap is the same.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import BaselineError

__all__ = [
    "EnsembleAggregatedLAC",
    "aggregate_deep_ensemble_scores",
    "aggregate_mc_dropout_scores",
]


def _aggregate_stack(
    score_stack: NDArray[np.float64], *, source: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared mean/variance reduction over a `(K, ...)` stochastic-score stack.

    Args:
        score_stack: `(K, ...)` array; axis 0 indexes the `K` stochastic
            samples (dropout passes or ensemble members), the remaining
            axes match one example's score shape (`[K]` classification,
            `[H,W,K]`/`[K,H,W]` segmentation, per `ScoreProvider`'s frozen
            shape contract).
        source: A short label used only in the raised error message
            (`"MC-dropout"` / `"deep ensemble"`), so a caller sees which
            of the two near-identical public functions rejected its input.

    Returns:
        `(mean, variance)`, each with `score_stack`'s shape minus axis 0.

    Raises:
        BaselineError: If `score_stack` has fewer than 2 samples along
            axis 0 (a variance needs at least 2 observations) or is empty
            along any other axis.
    """
    arr = np.asarray(score_stack, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[0] < 2:
        raise BaselineError(
            f"{source} aggregation requires at least 2 stochastic samples along axis 0, "
            f"got shape {arr.shape}"
        )
    if arr.size == 0:
        raise BaselineError(f"{source} score_stack must be non-empty")
    mean = np.mean(arr, axis=0)
    variance = np.var(arr, axis=0, ddof=1)
    return mean, variance


def aggregate_mc_dropout_scores(
    score_stack: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate `K` dropout-enabled stochastic forward passes into one effective score.

    Standard MC-dropout aggregation (Gal & Ghahramani 2016): the
    predictive mean across `K` dropout-enabled forward passes is used as
    the effective score fed downstream; the predictive variance is
    returned alongside as a diagnostic (not itself consumed by any frozen
    `PredictionSetConstructor`, which expects a single score array).

    Args:
        score_stack: `(K, ...)` stack of per-pass scores for one batch of
            examples, `K >= 2`.

    Returns:
        `(mean_score, variance)`.

    Raises:
        BaselineError: If `score_stack` has fewer than 2 samples along
            axis 0, or is empty.
    """
    return _aggregate_stack(score_stack, source="MC-dropout")


def aggregate_deep_ensemble_scores(
    score_stack: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate `K` independently trained ensemble members' scores into one effective score.

    Standard deep-ensemble aggregation (Lakshminarayanan, Pritzel &
    Blundell 2017): the same mean/variance reduction as
    :func:`aggregate_mc_dropout_scores`, over `K` independently trained
    models' outputs rather than `K` stochastic passes of one model — kept
    as a separate function so the two comparator baselines remain
    distinguishable in a results table even though their arithmetic is
    identical (see this module's own docstring).

    Args:
        score_stack: `(K, ...)` stack of per-member scores for one batch
            of examples, `K >= 2`.

    Returns:
        `(mean_score, variance)`.

    Raises:
        BaselineError: If `score_stack` has fewer than 2 samples along
            axis 0, or is empty.
    """
    return _aggregate_stack(score_stack, source="deep ensemble")


class EnsembleAggregatedLAC(Calibrator):
    """LAC's order-statistic calibration, downstream of an aggregated ensemble/MC-dropout score.

    A thin, disclosed wrapper: identical to
    `wfcrc.baselines.lac.SplitConformalLAC`, renamed and re-registered so
    an aggregated results table can distinguish "LAC over a raw score"
    from "LAC over a mean-aggregated ensemble/MC-dropout score." This
    class performs **no** aggregation itself — the caller must have
    already built `loss_table` from a mean score produced via
    `aggregate_mc_dropout_scores`/`aggregate_deep_ensemble_scores` upstream.
    """

    def __init__(self, *, lac: SplitConformalLAC | None = None) -> None:
        """Initialize the wrapper.

        Args:
            lac: An injected `SplitConformalLAC`; defaults to a fresh
                instance.
        """
        self._lac = lac if lac is not None else SplitConformalLAC()

    @property
    def baseline_name(self) -> str:
        """Return ``"ensemble_aggregated_lac"``."""
        return "ensemble_aggregated_lac"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Delegate to `SplitConformalLAC.calibrate`, unchanged."""
        return self._lac.calibrate(loss_table, cfg, seed=seed)


BASELINES["ensemble_aggregated_lac"] = EnsembleAggregatedLAC
