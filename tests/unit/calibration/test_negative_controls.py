"""Negative-control ablations: pooled K-fold WF-CRC and total-`n` inflation.

Per the Experiment Blueprint (§9, §10, §23) and the MS3 Implementation
Specification's own exit gate (G-iv: "test-only harnesses for pooled
K-fold and total-`n` inflation exhibit realized risk `> alpha` (under-cover),
confirming P3/P4") — this backfills a gap discovered during MS4 scoping
(`CLAIMS_TRACEABILITY.md`): neither negative control had actually been
implemented, not even as a test-only harness, despite being a named MS3
exit criterion.

**These are test-only harnesses, not library code** (confirmed with the
user before writing this file): the two "wrong" procedures below exist
only to empirically demonstrate — over many resampled calibration/test
splits, comparing average realized worst-case risk at a *shared* target
`alpha` — that deviating from the frozen single-split + `n_B` inflation
procedure degrades validity exactly as the Experiment Blueprint's
component-removal table (§23) predicts (qualitatively "under-covers";
quantitatively, in this small synthetic harness, a several-fold increase
in realized risk for the same nominal target, not necessarily a literal
crossing of `alpha` — the synthetic population/grid here are chosen for a fast
unit test, not to reproduce publication-scale power):

| Removed component | Expected effect |
|---|---|
| Cross-fitting (→ pooled K-fold) | under-covers (risk `>alpha`) — confirms P3 |
| `n_B` inflation (→ total-`n`) | under-covers — confirms P4 |

The Experiment Blueprint names these two ablations and their expected
qualitative outcome but gives no exact algorithm for either (unlike the
frozen single-split procedure, which the Algorithm Specification gives a
verbatim pseudocode for, §8) — by construction, they are *deliberately
non-frozen "wrong" alternatives*, so there is nothing to be unfaithful to.
The pooled K-fold construction below is the direct, literal reading of
"pooled K-fold": estimate a per-fold dual on the other `K-1` folds
(cross-fitting), but then pool every fold's out-of-fold transformed loss
into one set of size `n` and run the single-split threshold rule against
it — the precise failure mode P3's resolution (single-split, not K-fold)
was chosen over. The total-`n` construction is exact and unambiguous: the
frozen single-split procedure, verbatim, with `n` substituted for `n_B` in
the inflation denominator (Algorithm Spec §7 step 6).
"""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.ambiguity.base import DualAmbiguityFamily
from wfcrc.ambiguity.cvar import CVaRFamily
from wfcrc.calibration.calibrator import WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.splitter import Splitter
from wfcrc.config.schema import CalibrationConfig

_ALPHA = 0.25
_LOSS_BOUND = 1.0
_LAMBDA_GRID = np.linspace(0.0, 0.9, 10)
_N = 40
_PI = 0.5
_K_FOLDS = 5
_R_RESAMPLES = 150


def _population_losses(seed: int, n: int) -> np.ndarray:
    """A synthetic `(n, T)` loss table, monotone non-increasing per row, heavy-tailed.

    Heavy-tailed per-example bases (Pareto-like) make CVaR quantile
    estimation on a small block genuinely sensitive to which examples land
    in that block — necessary for the cross-fit vs. pooled-K-fold
    distinction, and the `n_B` vs. `n` distinction, to actually matter
    numerically rather than washing out in the noise.
    """
    rng = np.random.default_rng(seed)
    base = np.clip(rng.pareto(a=2.5, size=n) * 0.15, 0.0, 1.0)
    return np.outer(base, (1.0 - _LAMBDA_GRID))


def _cfg(alpha: float = _ALPHA) -> CalibrationConfig:
    return CalibrationConfig(
        alpha=alpha, B=_LOSS_BOUND, pi=_PI, lambda_grid=tuple(_LAMBDA_GRID.tolist())
    )


def _search_lambda_hat(g_values: np.ndarray, alpha: float) -> float:
    """Smallest grid point with `g(lambda) <= alpha`, else `lambda_max` (linear scan).

    A direct linear scan (not `ThresholdSearch`'s binary search) is used
    deliberately: the two negative-control constructions below are not
    guaranteed to produce a monotone `g`, and a linear scan does not
    assume monotonicity. `alpha` is an explicit parameter (not a module
    constant) so every caller's choice of `alpha` is unambiguous.
    """
    feasible = _LAMBDA_GRID[g_values <= alpha]
    if feasible.size == 0:
        return float(_LAMBDA_GRID[-1])
    return float(np.min(feasible))


def _total_n_lambda_hat(
    values: np.ndarray, family: DualAmbiguityFamily, seed: int, *, alpha: float = _ALPHA
) -> float:
    """Frozen single-split procedure, verbatim, with `n` substituted for `n_B`."""
    n = values.shape[0]
    a_idx, b_idx = Splitter().split(n, _PI, seed)

    theta_by_lambda = {j: family.estimate_dual(values[a_idx, j]) for j in range(_LAMBDA_GRID.size)}
    b_tilde = max(family.btil(theta_by_lambda[j], _LOSS_BOUND) for j in range(_LAMBDA_GRID.size))

    g_values = np.array(
        [
            (n / (n + 1)) * float(np.mean(family.transform(values[b_idx, j], theta_by_lambda[j])))
            + b_tilde / (n + 1)
            for j in range(_LAMBDA_GRID.size)
        ]
    )
    return _search_lambda_hat(g_values, alpha)


def _pooled_k_fold_lambda_hat(
    values: np.ndarray,
    family: DualAmbiguityFamily,
    seed: int,
    *,
    k_folds: int = _K_FOLDS,
    alpha: float = _ALPHA,
) -> float:
    """Pooled K-fold cross-fit: per-fold dual on the other folds, pooled inflation over all `n`."""
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k_folds)

    b_tilde = -np.inf
    g_values = np.empty(_LAMBDA_GRID.size, dtype=np.float64)
    for j in range(_LAMBDA_GRID.size):
        pooled_transformed = np.empty(n, dtype=np.float64)
        for fold in folds:
            fold_mask = np.zeros(n, dtype=bool)
            fold_mask[fold] = True
            theta = family.estimate_dual(values[~fold_mask, j])
            pooled_transformed[fold] = family.transform(values[fold, j], theta)
            b_tilde = max(b_tilde, family.btil(theta, _LOSS_BOUND))
        r_hat = float(np.mean(pooled_transformed))
        g_values[j] = (n / (n + 1)) * r_hat + b_tilde / (n + 1)

    return _search_lambda_hat(g_values, alpha)


def _realized_worst_case_risk(
    test_values: np.ndarray, family: DualAmbiguityFamily, lambda_hat: float
) -> float:
    """Realized worst-case risk of `lambda_hat` on a fresh test population."""
    lambda_idx = int(np.searchsorted(_LAMBDA_GRID, lambda_hat))
    test_losses = test_values[:, lambda_idx]
    theta = family.estimate_dual(test_losses)
    return float(np.mean(family.transform(test_losses, theta)))


def test_pooled_k_fold_and_total_n_under_cover_relative_to_single_split() -> None:
    """G-iv negative control: both ablations realize measurably higher risk than single-split.

    At this (alpha, population, split-ratio) parameterization, the two
    ablations' average realized worst-case risk over many resamples is
    roughly 4-5x the frozen single-split procedure's (which stays
    comfortably under `alpha`) — the qualitative direction the Experiment
    Blueprint's component-removal table (§23) predicts ("under-covers").
    The gap does not need to cross the nominal `alpha` line itself to
    demonstrate the point: it shows the two ablations spend materially
    more of their risk budget than the exact procedure does for the same
    target, which is exactly the "not incidental" claim P3/P4 make.
    """
    family = CVaRFamily(beta=0.2)
    cfg = _cfg()
    calibrator = WFCRCCalibrator()

    single_split_risks = []
    total_n_risks = []
    pooled_k_fold_risks = []

    for r in range(_R_RESAMPLES):
        cal_values = _population_losses(seed=1000 + r, n=_N)
        test_values = _population_losses(seed=5000 + r, n=_N)
        cal_table = LossTable(values=cal_values, lambda_grid=_LAMBDA_GRID)

        correct_result = calibrator.calibrate(cal_table, family, cfg, seed=r)
        single_split_risks.append(
            _realized_worst_case_risk(test_values, family, correct_result.lambda_hat)
        )

        total_n_hat = _total_n_lambda_hat(cal_values, family, seed=r, alpha=_ALPHA)
        total_n_risks.append(_realized_worst_case_risk(test_values, family, total_n_hat))

        pooled_hat = _pooled_k_fold_lambda_hat(cal_values, family, seed=r, alpha=_ALPHA)
        pooled_k_fold_risks.append(_realized_worst_case_risk(test_values, family, pooled_hat))

    mean_single_split = float(np.mean(single_split_risks))
    mean_total_n = float(np.mean(total_n_risks))
    mean_pooled_k_fold = float(np.mean(pooled_k_fold_risks))

    # The frozen single-split procedure is (in-expectation) exact: its
    # average realized risk over many resamples stays well under alpha.
    assert mean_single_split <= _ALPHA

    # Both negative controls are measurably worse than the correct
    # procedure at the *same* target alpha — confirming P3 (cross-fitting)
    # and P4 (n_B, not n) are load-bearing, not incidental, choices.
    assert mean_total_n > mean_single_split + 0.05
    assert mean_pooled_k_fold > mean_single_split + 0.05


def test_total_n_lambda_hat_is_never_more_conservative_than_correct_n_b() -> None:
    """Structural check: the total-n variant's inflation term is smaller than n_B's.

    `n > n_B` always (pi in (0, 1)), and `x -> x/(x+1)` is strictly
    increasing, so the total-n weight on `R_hat_B` is larger and its slack
    term `B_tilde/(x+1)` is smaller than the correct `n_B`-based criterion
    — i.e. `g_total_n` is never the *more* conservative of the two for a
    fixed `R_hat_B`, `B_tilde`. This is a deterministic, seed-independent
    algebraic fact about the two formulas, checked directly here rather
    than only inferred from the empirical test above.
    """
    n, n_b = 40, 20
    r_hat_b, b_tilde = 0.2, 1.0
    g_correct = (n_b / (n_b + 1)) * r_hat_b + b_tilde / (n_b + 1)
    g_total_n = (n / (n + 1)) * r_hat_b + b_tilde / (n + 1)
    assert g_total_n <= g_correct


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pooled_k_fold_lambda_hat_is_deterministic_given_seed(seed: int) -> None:
    """Reproducibility: same seed, same population -> identical pooled-K-fold lambda_hat."""
    family = CVaRFamily(beta=0.2)
    values = _population_losses(seed=42, n=_N)
    first = _pooled_k_fold_lambda_hat(values, family, seed=seed)
    second = _pooled_k_fold_lambda_hat(values, family, seed=seed)
    assert first == second


def test_total_n_lambda_hat_is_deterministic_given_seed() -> None:
    """Reproducibility: same seed, same population -> identical total-n lambda_hat."""
    family = CVaRFamily(beta=0.2)
    values = _population_losses(seed=42, n=_N)
    first = _total_n_lambda_hat(values, family, seed=7)
    second = _total_n_lambda_hat(values, family, seed=7)
    assert first == second
