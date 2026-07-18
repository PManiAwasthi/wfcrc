"""Temperature scaling (Guo et al. 2017) and selective scaling (Geifman & El-Yaniv 2017).

Per `docs/EXPERIMENT_PROTOCOL.md`/`docs/MODEL_POLICY.md` §1.2 ("temperature
/ selective scaling: standard, non-guaranteed UQ pre-empting 'missing
baseline'"): both methods are **score-level** recalibration procedures —
they rescale/threshold a model's raw confidence output *before* any
`wfcrc.calibration.loss_table.LossTable` is built from it, not calibration
procedures over an already-built `LossTable` the way every other module in
`wfcrc.baselines` is. This is a genuine architecture boundary, not a
scoping shortcut: `wfcrc.calibration.calibrator.WFCRCCalibrator.calibrate`
and every `wfcrc.baselines.base.Calibrator` implementation depend only on
`LossTable` + scalars (L1a dimension-independence, frozen since MS2) —
neither method below can be expressed as a pure function of an
already-built `LossTable`, since temperature scaling needs the *raw
logits* (not the post-hoc, per-`lambda` loss values a `LossTable` stores)
and selective scaling needs a *per-example confidence score paired with
its own loss*, i.e. the same conditional-mean-over-a-shrinking-subset
structure no other frozen `g(lambda) = mean over all n` criterion in this
project uses.

**Consequently:** :func:`fit_temperature`/:func:`apply_temperature` and
:func:`fit_selective_threshold`/:func:`apply_selective_threshold` are
plain, `LossTable`-independent NumPy utilities (testable against synthetic
logit/confidence arrays with no model or dataset, exactly like
`wfcrc.datasets.preprocessing`'s functions were in MS6.2) — real
integration (recalibrating an actual model's real logits) is deferred
until a concrete `ScoreProvider` exists, the same "generic piece now, real
data later" sequencing this whole project already follows. Temperature
scaling's *downstream* decision (once its recalibrated scores have built a
`LossTable`) is still expressible as an ordinary `Calibrator` —
`TemperatureScaledLAC` below is a thin, disclosed wrapper: the same
`SplitConformalLAC` order-statistic rule, applied to whatever `LossTable`
was built from already-temperature-scaled scores. Selective scaling has no
equivalent downstream wrapper (its output *is* the abstention decision;
there is no LAC-style set-construction step after it), so none is provided
— see the MS9 final report for this explicit scope boundary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from wfcrc.baselines.base import BASELINES, Calibrator
from wfcrc.baselines.lac import SplitConformalLAC
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import BaselineError

__all__ = [
    "TemperatureScaledLAC",
    "apply_selective_threshold",
    "apply_temperature",
    "fit_selective_threshold",
    "fit_temperature",
]

#: Golden-section search bounds/tolerance for temperature fitting.
_T_MIN = 1e-2
_T_MAX = 1e2
_GOLDEN_TOL = 1e-8
_GOLDEN_MAX_ITER = 200
_INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0


def _log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable row-wise log-softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _golden_section_minimize(h: Callable[[float], float], lo: float, hi: float) -> float:
    """Golden-section search for the minimizer of a unimodal `h` on `[lo, hi]`."""
    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    h_c, h_d = h(c), h(d)
    for _ in range(_GOLDEN_MAX_ITER):
        if (b - a) < _GOLDEN_TOL:
            break
        if h_c < h_d:
            b, d, h_d = d, c, h_c
            c = b - _INV_PHI * (b - a)
            h_c = h(c)
        else:
            a, c, h_c = c, d, h_d
            d = a + _INV_PHI * (b - a)
            h_d = h(d)
    # The loop always reaches `break` well before `_GOLDEN_MAX_ITER`
    # iterations: the bracket width shrinks by the same golden-ratio factor
    # every iteration regardless of `h`'s behavior (both branches above
    # shrink `(b - a)` identically), so for any finite `lo < hi` the width
    # is below `_GOLDEN_TOL` long before 200 iterations (0.618^200 is many
    # orders of magnitude below any reachable tolerance) -- not a reachable
    # fallback, the same disclosed-unreachable-defensive-code pattern as
    # `wfcrc/evaluation/metrics.py`'s `assert var_w > 0.0` in `paired_wilcoxon`.
    return (a + b) / 2.0


def fit_temperature(
    logits: NDArray[np.float64],
    labels: NDArray[np.intp],
    *,
    t_min: float = _T_MIN,
    t_max: float = _T_MAX,
) -> float:
    """Fit a scalar temperature `T` minimizing NLL on calibration logits (Guo et al. 2017).

    Args:
        logits: `(n, K)` pre-softmax class logits.
        labels: `(n,)` integer true-class indices, `0 <= labels[i] < K`.
        t_min: Lower search bound on `T`, must satisfy `0 < t_min < t_max`.
        t_max: Upper search bound on `T`.

    Returns:
        The fitted temperature `T > 0` minimizing
        `NLL(T) = -mean(log_softmax(logits/T)[i, labels[i]])`, via a
        dependency-free golden-section search (no scipy, matching this
        project's own "no new numerical library unless genuinely required"
        policy already used for the KL family's dual solver).

    Raises:
        BaselineError: If `logits` is not 2-D, `labels` has the wrong
            length or an out-of-range class index, or `t_min >= t_max`.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.intp)
    if logits.ndim != 2:
        raise BaselineError(f"logits must be 2-D (n, K), got shape {logits.shape}")
    n, k = logits.shape
    if labels.shape != (n,):
        raise BaselineError(f"labels must have shape ({n},), got {labels.shape}")
    if np.any((labels < 0) | (labels >= k)):
        raise BaselineError(
            f"labels must be in [0, {k}); got range [{labels.min()}, {labels.max()}]"
        )
    if not (0.0 < t_min < t_max):
        raise BaselineError(f"require 0 < t_min < t_max, got t_min={t_min}, t_max={t_max}")

    rows = np.arange(n)

    def nll(temperature: float) -> float:
        log_probs = _log_softmax(logits / temperature)
        return float(-np.mean(log_probs[rows, labels]))

    return _golden_section_minimize(nll, t_min, t_max)


def apply_temperature(logits: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """Apply a fitted temperature, returning calibrated class probabilities.

    Args:
        logits: `(n, K)` pre-softmax class logits (or a single `(K,)` row).
        temperature: A temperature `T > 0`, e.g. from :func:`fit_temperature`.

    Returns:
        `softmax(logits / T)`, same shape as `logits`.

    Raises:
        BaselineError: If `temperature <= 0`.
    """
    if not (temperature > 0.0):
        raise BaselineError(f"temperature must be > 0, got {temperature}")
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def fit_selective_threshold(
    confidence: NDArray[np.float64], losses: NDArray[np.float64], target_risk: float
) -> float:
    """Largest-coverage confidence threshold with empirical selective risk `<= target_risk`.

    Standard selective-classification thresholding (Geifman & El-Yaniv
    2017, "Selective Classification for Deep Neural Networks", §3):
    among examples with `confidence >= tau`, find the smallest `tau`
    (i.e. largest coverage) whose empirical mean loss on that selected
    subset is `<= target_risk`. **Disclosed simplification:** this is the
    paper's own basic empirical selective-risk criterion, not the paper's
    additional finite-sample statistical guarantee bound (their `SGR`
    algorithm's binomial-tail correction) — no exact formula for that
    stronger bound is transcribed anywhere in this project's frozen vault,
    so implementing it here would mean inventing a specific numeric
    correction rather than following a verified one (the same "disclose,
    do not invent" precedent already used for `one_sided_risk_test`/
    `paired_wilcoxon`/`holm_correct`, `wfcrc/evaluation/metrics.py`).

    Args:
        confidence: `(n,)` per-example confidence scores (higher = more
            confident; e.g. max softmax probability).
        losses: `(n,)` per-example loss/error values, same length as
            `confidence`.
        target_risk: The target selective risk level.

    Returns:
        The threshold `tau`; examples with `confidence >= tau` are
        "selected" (not abstained). If no non-empty selected subset
        achieves `target_risk`, returns `+inf` (select nothing —
        the selective-classification analogue of the empty-selection
        fallback `wfcrc.calibration.threshold_search.ThresholdSearch`
        uses elsewhere in this project).

    Raises:
        BaselineError: If `confidence`/`losses` differ in length or are
            empty.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if confidence.shape != losses.shape:
        raise BaselineError(
            f"confidence and losses must have the same shape, got "
            f"{confidence.shape} vs {losses.shape}"
        )
    if confidence.size == 0:
        raise BaselineError("confidence/losses must be non-empty")

    order = np.argsort(-confidence, kind="mergesort")
    sorted_conf = confidence[order]
    sorted_loss = losses[order]
    cum_mean_loss = np.cumsum(sorted_loss) / np.arange(1, confidence.size + 1)

    feasible = np.nonzero(cum_mean_loss <= target_risk)[0]
    if feasible.size == 0:
        return float("inf")
    # Largest coverage (most inclusive prefix) among feasible prefixes.
    best_idx = int(feasible[-1])
    return float(sorted_conf[best_idx])


def apply_selective_threshold(confidence: NDArray[np.float64], tau: float) -> NDArray[np.bool_]:
    """Return the selection mask `{i : confidence[i] >= tau}`.

    Args:
        confidence: `(n,)` per-example confidence scores.
        tau: A threshold, e.g. from :func:`fit_selective_threshold`.

    Returns:
        A boolean `(n,)` mask, `True` where the example is selected
        (not abstained).
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    return confidence >= tau


class TemperatureScaledLAC(Calibrator):
    """LAC's order-statistic calibration, applied downstream of a temperature-scaled score.

    A thin, disclosed wrapper: identical to
    `wfcrc.baselines.lac.SplitConformalLAC`, renamed and re-registered so
    an aggregated results table (`docs/RESULTS_SCHEMA.md` §2.2) can
    distinguish "LAC over raw scores" from "LAC over temperature-scaled
    scores" by `family`/`baseline_name` alone. This class performs **no**
    temperature fitting itself — the caller must have already built
    `loss_table` from scores produced via :func:`apply_temperature`
    upstream (see this module's own docstring for why that step cannot
    happen inside a `Calibrator`).
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
        """Return ``"temperature_scaled_lac"``."""
        return "temperature_scaled_lac"

    def calibrate(
        self, loss_table: LossTable, cfg: CalibrationConfig, *, seed: int
    ) -> CalibrationResult:
        """Delegate to `SplitConformalLAC.calibrate`, unchanged."""
        return self._lac.calibrate(loss_table, cfg, seed=seed)


BASELINES["temperature_scaled_lac"] = TemperatureScaledLAC
