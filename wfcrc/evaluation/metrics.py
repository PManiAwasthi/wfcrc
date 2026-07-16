"""Realized-risk and statistical utilities (Implementation Blueprint §6, MS4 spec §C2, M13).

Free functions matching the frozen specification's own literal, unprefixed
API text (no `MetricSuite.` prefix appears anywhere in the Implementation
Blueprint's or MS4 Implementation Spec's API lines) — the same precedent
already recorded for `assert_monotone`/`calibrate` in `CLAIMS_TRACEABILITY.md`
§2 (prefer the Blueprint's simpler form when it differs from a fuller
class-based framing elsewhere).

**Provenance disclosure (matching the FPR-loss gap-fill precedent,
`CLAIMS_TRACEABILITY.md` §3):** the Experiment Blueprint (§12) and MS4 spec
name `one_sided_risk_test`, `paired_wilcoxon`, and `holm_correct` as
required statistical utilities but give no exact formula for any of them —
only the textual descriptions reproduced in each function's docstring
below. Each is implemented as the standard, textbook realization of the
named procedure (one-sample one-sided z-test under the normal
approximation; Wilcoxon signed-rank test with the standard tie-corrected
normal approximation; the exact Holm-Bonferroni step-down algorithm), with
no scipy dependency (`math.erf` supplies the exact normal CDF). This is
disclosed here, not silently assumed.

**Scope note:** `realized_worst_case_risk` is defined only for dual
families (`cvar`/`kl`). The finite-group and known-weight branches' own
notion of "worst case" is already covered by other functions in this
module — `per_group_risk` maxed over groups for finite-group, and the
weighted mean itself (`realized_marginal_risk` is the unweighted analogue)
for known-weight — so no separate dispatch branch is invented for them
here; passing a non-dual family raises `FamilyError`, mirroring
`WFCRCCalibrator.calibrate`'s own dispatch guard.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.calibration.calibrator import CalibrationResult
from wfcrc.calibration.loss_table import LossTable
from wfcrc.exceptions import FamilyError
from wfcrc.losses.miscoverage import MiscoverageLoss
from wfcrc.prediction_sets.base import PredictionSetConstructor
from wfcrc.utils.seeds import derive_seed

__all__ = [
    "CI",
    "TestResult",
    "bootstrap_ci",
    "coverage",
    "duality_gap",
    "effective_sizes",
    "holm_correct",
    "mean_set_size",
    "one_sided_risk_test",
    "paired_wilcoxon",
    "per_group_risk",
    "realized_marginal_risk",
    "realized_worst_case_risk",
]


@dataclass(frozen=True)
class CI:
    """A two-sided confidence interval.

    Attributes:
        lo: Lower bound.
        hi: Upper bound.
        level: Nominal confidence level (e.g. `0.95`).
    """

    lo: float
    hi: float
    level: float


@dataclass(frozen=True)
class TestResult:
    """The outcome of a statistical hypothesis test.

    Attributes:
        statistic: The test statistic.
        p_value: The test's p-value.
    """

    statistic: float
    p_value: float
    #: Suppresses pytest's name-based test-class collection (this class is
    #: named `TestResult` per the frozen spec's own API text, not a test).
    __test__ = False


#: Below this sample standard deviation, `risks` is treated as
#: floating-point-noise-only zero variance (e.g. `std([0.2, 0.2, 0.2],
#: ddof=1))` is `~3e-17`, not exactly `0.0`) — the z-statistic's
#: denominator would otherwise blow up on rounding noise alone.
_NEAR_ZERO_STD_TOL = 1e-9


def _normal_cdf(z: float) -> float:
    """Standard normal CDF, via the exact error function (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def realized_worst_case_risk(
    result: CalibrationResult, test_loss_table: LossTable, family: AmbiguityFamily
) -> float:
    """Realized worst-case risk of the deployed `C_λ̂` on a held-out test set.

    Re-derives the family's own worst-case functional (Algorithm Spec §7,
    A5: `sup_Q E_Q[L] = min_theta [c(theta) + E_P[t(L;theta)]]`) on the
    test loss column at `lambda_hat`, estimating a *fresh* dual parameter
    from the test data itself — the calibration-time `theta_hat` was fit on
    calibration block A and has no relationship to the test distribution.

    **Descriptive-statistic caveat:** `theta` is re-estimated on the same
    `test_loss_table` this function then measures, so the result carries
    the ordinary optimism of any plug-in/empirical-minimum estimator (the
    argmin is chosen to fit the very sample it is evaluated on, which
    tends to understate the true population worst-case risk somewhat).
    This is *not* the same failure mode Math Spec §12 item 3 warns
    against (a same-data dual feeding a conformal *threshold-selection*
    rule, which breaks a finite-sample validity guarantee) — no threshold
    is selected here, `lambda_hat` is already fixed, and this function
    makes no validity claim of its own. It is purely a descriptive
    measurement, and like most such plug-in estimates it should be read
    as mildly optimistic rather than as an unbiased estimate of the true
    worst-case risk.

    Args:
        result: The calibration outcome whose `lambda_hat` is deployed.
        test_loss_table: Held-out test-set loss table (same `lambda_grid`
            as the calibration table `result` was produced from).
        family: The ambiguity family to evaluate the worst case under; must
            be a `DualAmbiguityFamily` (`cvar`/`kl`) — see the module
            docstring's scope note.

    Returns:
        `c(theta) + mean(t(L_test; theta))`, the realized worst-case risk.

    Raises:
        FamilyError: If `family` is not a `DualAmbiguityFamily`.
        ValueError: If `result.lambda_hat` is not in
            `test_loss_table.lambda_grid`.
    """
    if not isinstance(family, DualAmbiguityFamily):
        raise FamilyError(
            "realized_worst_case_risk is only defined for dual families (cvar/kl); "
            f"got family_type={family.family_type!r}"
        )
    test_losses = test_loss_table.column(result.lambda_hat)
    theta = family.estimate_dual(test_losses)
    return float(np.mean(family.transform(test_losses, theta)))


def realized_marginal_risk(result: CalibrationResult, test_loss_table: LossTable) -> float:
    """Realized marginal (unweighted) risk of the deployed `C_λ̂` on a test set.

    Args:
        result: The calibration outcome whose `lambda_hat` is deployed.
        test_loss_table: Held-out test-set loss table.

    Returns:
        `mean(L_test[:, lambda_hat])`.
    """
    return float(np.mean(test_loss_table.column(result.lambda_hat)))


def per_group_risk(
    result: CalibrationResult,
    test_loss_table: LossTable,
    groups: Sequence[Sequence[int]],
) -> dict[int, float]:
    """Realized per-group mean risk of the deployed `C_λ̂` on a test set.

    Args:
        result: The calibration outcome whose `lambda_hat` is deployed.
        test_loss_table: Held-out test-set loss table.
        groups: One row-index sequence per group, indexing into
            `test_loss_table`.

    Returns:
        `{group_index: mean(L_test[group_indices, lambda_hat])}`, keyed by
        `groups`' position.

    Raises:
        ValueError: If `groups` is empty, or any group is empty.
    """
    if len(groups) == 0:
        raise ValueError("groups must be non-empty")
    col = test_loss_table.column(result.lambda_hat)
    output: dict[int, float] = {}
    for g, indices in enumerate(groups):
        if len(indices) == 0:
            raise ValueError(f"group {g} is empty")
        output[g] = float(np.mean(col[np.asarray(indices, dtype=np.intp)]))
    return output


def mean_set_size(
    constructor: PredictionSetConstructor, scores: Sequence[ArrayLike], lam: float
) -> float:
    """Mean cardinality of `C_λ̂(x)` over a test set of scores.

    Args:
        constructor: Builds `C_λ(score)`.
        scores: Per-example scores to build sets from.
        lam: The deployed threshold `λ̂`.

    Returns:
        `mean(|C_λ̂(score_i)|)` over `scores`.

    Raises:
        ValueError: If `scores` is empty.
    """
    if len(scores) == 0:
        raise ValueError("scores must be non-empty")
    sizes = [float(np.sum(constructor.construct(score, lam))) for score in scores]
    return float(np.mean(sizes))


def coverage(
    constructor: PredictionSetConstructor,
    scores: Sequence[ArrayLike],
    labels: Sequence[ArrayLike],
    lam: float,
) -> float:
    """Empirical coverage `P(Y ⊆ C_λ̂(X))` over a test set.

    Reuses the frozen :class:`~wfcrc.losses.miscoverage.MiscoverageLoss`
    (`l = 1{Y∉C}`): `coverage = 1 - mean(miscoverage indicator)`.

    Args:
        constructor: Builds `C_λ(score)`.
        scores: Per-example scores to build sets from.
        labels: Per-example ground-truth labels, same length as `scores`.
        lam: The deployed threshold `λ̂`.

    Returns:
        The fraction of examples with full label coverage, in `[0, 1]`.

    Raises:
        ValueError: If `scores`/`labels` are empty or differ in length.
    """
    if len(scores) == 0:
        raise ValueError("scores must be non-empty")
    if len(scores) != len(labels):
        raise ValueError(f"scores has {len(scores)} entries but labels has {len(labels)}")
    miscoverage_loss = MiscoverageLoss()
    indicators = [
        miscoverage_loss.evaluate(
            constructor.construct(score, lam), np.asarray(label, dtype=np.bool_)
        )
        for score, label in zip(scores, labels, strict=True)
    ]
    return 1.0 - float(np.mean(indicators))


def effective_sizes(
    result: CalibrationResult, *, weights: ArrayLike | None = None
) -> dict[str, float]:
    """Effective sample sizes for a calibration result.

    `n_A`/`n_B` (dual branch) and per-group `n_G` (finite-group branch) are
    read directly from `result`; `n_eff = (Σw)² / Σw²` (Kish's effective
    sample size, MS4 Implementation Spec §C2 item 5) is computed from
    `weights` when given (known-weight branch).

    Args:
        result: The calibration outcome to summarize.
        weights: Known weights (known-weight branch only); `None` if not
            applicable.

    Returns:
        A dict with whichever of `"n_a"`, `"n_b"`, `"n_g_<i>"`, `"n_eff"`
        apply to `result` (and `weights`, if given).
    """
    output: dict[str, float] = {}
    if result.n_a is not None:
        output["n_a"] = float(result.n_a)
    if result.n_b is not None:
        output["n_b"] = float(result.n_b)
    per_group = result.diagnostics.get("per_group")
    if per_group is not None:
        for item in per_group:
            output[f"n_g_{item['group']}"] = float(item["n_g"])
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        output["n_eff"] = float(np.sum(w) ** 2 / np.sum(w**2))
    return output


def duality_gap(surrogate_risk: float, realized_risk: float) -> float:
    """Duality-gap proxy: surrogate (calibration-time) risk minus realized (test-time) risk.

    Args:
        surrogate_risk: The calibration-time surrogate (e.g. `g(lambda_hat)`
            or `result.r_hat_b`).
        realized_risk: The realized (test-time) risk (e.g. from
            `realized_worst_case_risk`).

    Returns:
        `surrogate_risk - realized_risk`.
    """
    return surrogate_risk - realized_risk


def bootstrap_ci(
    values: Sequence[float], level: float = 0.95, *, n_resamples: int = 2000, seed: int
) -> CI:
    """Nonparametric percentile bootstrap confidence interval on `mean(values)`.

    Args:
        values: Observed values (e.g. per-resample realized risks).
        level: Nominal two-sided confidence level, in `(0, 1)`.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the deterministic resampling RNG (derived via
            :func:`wfcrc.utils.seeds.derive_seed`, never a bare global RNG).

    Returns:
        A `CI` with the `(1-level)/2` and `1-(1-level)/2` percentiles of
        the bootstrap distribution of `mean(values)`.

    Raises:
        ValueError: If `values` is empty, `level` is outside `(0, 1)`, or
            `n_resamples < 1`.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    derived_seed = derive_seed("evaluation.metrics.bootstrap_ci", seed)
    rng = np.random.default_rng(derived_seed)
    n = arr.size
    resample_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = np.mean(arr[idx])

    alpha = 1.0 - level
    lo = float(np.percentile(resample_means, 100 * alpha / 2))
    hi = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return CI(lo=lo, hi=hi, level=level)


def one_sided_risk_test(risks: Sequence[float], alpha: float) -> TestResult:
    """One-sided test of `H0: E[realized risk] <= alpha` (Experiment Blueprint §12).

    Standard one-sample, one-sided z-test under the normal approximation:
    `z = (mean(risks) - alpha) / (std(risks) / sqrt(n))`, upper-tail
    p-value `1 - Phi(z)`. The Experiment Blueprint names the test only as
    "one-sided test of H0: E[realized risk] <= alpha", with no formula;
    this is the standard textbook realization (no scipy) — see the module
    docstring's provenance disclosure. A large p-value means the data is
    consistent with validity (no evidence the true mean risk exceeds
    `alpha`); a small p-value flags a possible violation.

    Args:
        risks: Per-resample realized risks, `n >= 2`.
        alpha: The target risk level `alpha`.

    Returns:
        `TestResult(statistic=z, p_value=1 - Phi(z))`.

    Raises:
        ValueError: If `risks` has fewer than 2 entries, or has zero
            variance (the z-statistic is undefined).
    """
    arr = np.asarray(risks, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("risks must have at least 2 entries")
    std = float(np.std(arr, ddof=1))
    if std < _NEAR_ZERO_STD_TOL:
        raise ValueError("risks has (near-)zero variance; the z-statistic is undefined")
    n = arr.size
    z = (float(np.mean(arr)) - alpha) / (std / math.sqrt(n))
    p_value = 1.0 - _normal_cdf(z)
    return TestResult(statistic=z, p_value=p_value)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assign 1-based ranks to `values`, averaging ranks within ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty_like(values, dtype=np.float64)
    n = values.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Paired Wilcoxon signed-rank test (Experiment Blueprint §12), normal approximation.

    Standard textbook algorithm (no scipy) — see the module docstring's
    provenance disclosure: differences `d_i = a_i - b_i` with `d_i = 0`
    discarded; ranks assigned to `|d_i|` (average ranks for ties);
    `W+ = sum of ranks where d_i > 0`; normal-approximation z-score using
    the standard mean/variance of `W+` under `H0`, with the standard tie
    correction to the variance.

    Args:
        a: First paired sample.
        b: Second paired sample, same length as `a`.

    Returns:
        `TestResult(statistic=z, p_value=two-sided p-value)`.

    Raises:
        ValueError: If `a`/`b` differ in shape, or no nonzero difference
            remains after discarding ties.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"a and b must have the same shape, got {a_arr.shape} vs {b_arr.shape}")
    diff = a_arr - b_arr
    nonzero = diff[diff != 0.0]
    n = nonzero.size
    if n == 0:
        raise ValueError("no nonzero differences; the Wilcoxon statistic is undefined")

    abs_diff = np.abs(nonzero)
    ranks = _average_ranks(abs_diff)
    signed_ranks = ranks * np.sign(nonzero)
    w_plus = float(np.sum(signed_ranks[signed_ranks > 0]))

    mean_w = n * (n + 1) / 4.0
    _, tie_counts = np.unique(abs_diff, return_counts=True)
    tie_correction = float(np.sum(tie_counts.astype(np.float64) ** 3 - tie_counts))
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    # var_w > 0 always: the untied term n(n+1)(2n+1)/24 ~ n^3/12 dominates
    # the maximum possible tie correction (n^3-n)/48 ~ n^3/48 for every
    # n >= 1 (ratio 4:1), even in the worst case of a single tie group
    # spanning all n values — so this is not a reachable failure mode,
    # unlike the two genuine ValueError cases above.
    assert var_w > 0.0

    z = (w_plus - mean_w) / math.sqrt(var_w)
    p_value = min(2.0 * (1.0 - _normal_cdf(abs(z))), 1.0)
    return TestResult(statistic=z, p_value=p_value)


def holm_correct(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down multiple-comparison correction.

    Standard, exact algorithm: sort ascending, multiply the `k`-th smallest
    p-value (1-indexed rank `i`) by `(m - i + 1)`, enforce running-maximum
    monotonicity, clip to `1.0`, then restore the original order.

    Args:
        pvals: Uncorrected p-values.

    Returns:
        Corrected p-values, same order as `pvals`.

    Raises:
        ValueError: If `pvals` is empty, or any value is outside `[0, 1]`.
    """
    arr = np.asarray(pvals, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("pvals must be non-empty")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("pvals must each be in [0, 1]")

    m = arr.size
    order = np.argsort(arr, kind="mergesort")
    sorted_p = arr[order]
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for k in range(m):
        candidate = (m - k) * sorted_p[k]
        running_max = max(running_max, candidate)
        adjusted[k] = min(running_max, 1.0)

    result = np.empty(m, dtype=np.float64)
    result[order] = adjusted
    return result.tolist()
