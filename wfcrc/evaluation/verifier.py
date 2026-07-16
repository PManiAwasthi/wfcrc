"""``Verifier`` — centralizes the frozen preconditions + AS §20 checklist (M12).

Per the Implementation Blueprint (§6, `verify.Verifier`) and the MS4
Implementation Specification (C1): this is "the single go/no-go gate...
Centralizes (does not duplicate) the MS3 inline gates so there is one
source of truth." Every check is deterministic and numeric — no dataset,
model, or statistical-significance machinery, matching this milestone's
explicit scope (the Algorithm Specification §20 checklist's final item, an
"empirical validity smoke test... illustrative, not a proof", is
experimental-statistical validation and is out of this milestone's scope by
the task brief; it is exercised as a synthetic-data test in
:mod:`wfcrc.calibration`'s own test suite, per `CLAIMS_TRACEABILITY.md`).

Checks implemented (Algorithm Specification §20, items 1-6):

- :meth:`Verifier.check_preconditions` — P-2 monotonicity and boundedness,
  checked directly on a :class:`~wfcrc.calibration.loss_table.LossTable`
  (item 1). P-1 nestedness is *not* re-derived here: the loss table no
  longer carries the raw per-example scores a `PredictionSetConstructor`
  needs (L1a dimension-independence — Implementation Blueprint §8), and
  nestedness is already gated at the `prediction_sets` layer via
  :meth:`wfcrc.prediction_sets.base.PredictionSetConstructor.assert_nested`
  (MS2 Implementation Spec, C3 item 12) — this follows the Implementation
  Blueprint's own simpler `check_preconditions(loss_table)->Report`
  signature (§6) rather than the MS4 spec's richer
  `(loss_table, constructor, loss)`, the same kind of Blueprint-over-detail
  deviation already recorded for `assert_monotone`/`calibrate` in
  `CLAIMS_TRACEABILITY.md` §2.
- :meth:`Verifier.check_calibration` — re-derives, from the same frozen
  primitives the calibrator itself uses (:class:`~wfcrc.calibration.splitter.Splitter`,
  the `AmbiguityFamily`, a fresh :class:`~wfcrc.calibration.calibrator.WFCRCCalibrator`
  run), the remaining checklist items: `A∩B=∅` and correct block sizes
  (item 2, dual branch), `B̃<∞` and `L̃≤B̃` (item 3, dual branch), `g`
  monotone and `λ̂` minimal (item 4), the correct inflation denominator per
  branch (item 5), and reproducibility under the fixed seed (item 6).
  Checks that do not apply to the family's branch are reported as passed
  with a "not applicable" detail (MS4 spec C1: "missing inputs → skip item
  with a WARN detail").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wfcrc.ambiguity.base import AmbiguityFamily, DualAmbiguityFamily
from wfcrc.calibration.calibrator import CalibrationResult, WFCRCCalibrator
from wfcrc.calibration.loss_table import LossTable
from wfcrc.calibration.splitter import Splitter
from wfcrc.config.schema import CalibrationConfig
from wfcrc.exceptions import VerificationError

__all__ = ["CheckResult", "VerificationReport", "Verifier"]


@dataclass(frozen=True)
class CheckResult:
    """One discrete verification item's outcome.

    Attributes:
        name: Short, stable identifier for this check.
        passed: Whether the check passed (checks reported as "not
            applicable" to a given branch are recorded as `passed=True`).
        detail: Human-readable explanation of the outcome.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    """The aggregate outcome of a set of :class:`CheckResult` items.

    Attributes:
        items: Every check that was run, in execution order.
    """

    items: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Return `True` iff every item in :attr:`items` passed."""
        return all(item.passed for item in self.items)

    def assert_ok(self) -> None:
        """Raise if any item failed (the strict gate).

        Raises:
            VerificationError: Naming every failing check, if
                :attr:`passed` is `False`.
        """
        failing = [item.name for item in self.items if not item.passed]
        if failing:
            raise VerificationError(f"verification failed: {', '.join(failing)}")

    def merge(self, other: VerificationReport) -> VerificationReport:
        """Combine this report with another, concatenating their items.

        Args:
            other: Another report (e.g. from a different `check_*` call).

        Returns:
            A new report whose `items` is `self.items + other.items`.
        """
        return VerificationReport(items=self.items + other.items)


class Verifier:
    """Runs the deterministic AS §20 checks against calibration artifacts."""

    def __init__(self, *, tol: float = 1e-9) -> None:
        """Initialize the verifier.

        Args:
            tol: Numerical tolerance shared by every numeric comparison
                below (guards against floating-point noise, not against a
                real violation — mirrors
                :meth:`wfcrc.losses.base.LossEvaluator.assert_monotone`'s
                own `tol` convention).
        """
        self.tol = tol

    def check_preconditions(
        self, loss_table: LossTable, *, loss_bound: float
    ) -> VerificationReport:
        """Check P-2 (monotonicity, boundedness) directly on a loss table.

        Args:
            loss_table: The `L[i, lambda]` table to check.
            loss_bound: `B`, the loss's declared upper bound.

        Returns:
            A report with two items: ``"p2_monotone_nonincreasing"`` and
            ``"p2_bounded"``.
        """
        diffs = np.diff(loss_table.values, axis=1)
        monotone_ok = bool(np.all(diffs <= self.tol))
        bounded_ok = bool(np.all(loss_table.values <= loss_bound + self.tol))
        items = (
            CheckResult(
                name="p2_monotone_nonincreasing",
                passed=monotone_ok,
                detail=(
                    "every row of L[i,lambda] is non-increasing in lambda"
                    if monotone_ok
                    else "found a row of L[i,lambda] that increases beyond tol"
                ),
            ),
            CheckResult(
                name="p2_bounded",
                passed=bounded_ok,
                detail=(
                    f"L[i,lambda] <= B={loss_bound} for every entry"
                    if bounded_ok
                    else f"found an entry of L[i,lambda] > B={loss_bound}"
                ),
            ),
        )
        return VerificationReport(items=items)

    def check_calibration(
        self,
        result: CalibrationResult,
        loss_table: LossTable,
        family: AmbiguityFamily,
        cfg: CalibrationConfig,
        *,
        seed: int,
    ) -> VerificationReport:
        """Check AS §20 items 2-6 against a `CalibrationResult`.

        Args:
            result: The calibration outcome to verify.
            loss_table: The `L[i, lambda]` table `result` was computed from.
            family: The ambiguity family `result` was computed with.
            cfg: The calibration config `result` was computed with.
            seed: The seed `result` was computed with.

        Returns:
            A report covering reproducibility (every branch) plus the
            branch-specific items (dual: split/`B̃`/`g`-monotonicity/
            `λ̂`-minimality; finite-group: per-group threshold consistency;
            known-weight: full-`n` usage); non-applicable items are
            reported as passed with a "not applicable" detail.

        Raises:
            wfcrc.exceptions.FamilyError: Propagated from
                :meth:`wfcrc.calibration.calibrator.WFCRCCalibrator.calibrate`
                (invoked internally by the reproducibility check, which
                runs first) if `family.family_type` claims a branch
                (`"cvar"`/`"kl"`/`"finite_group"`/`"known_weight"`) but
                `family` does not actually implement that branch's
                required interface.
        """
        # `_check_reproducibility` (above) unconditionally calls
        # `WFCRCCalibrator.calibrate(loss_table, family, cfg, seed=seed)`, which
        # already performs this exact family/branch validation (and raises
        # `FamilyError` first if it fails) — so by this point `family` is
        # guaranteed well-formed for its declared `family_type`. The
        # `assert`s below are for `mypy`'s type narrowing only, not a second
        # runtime guard (MS4 spec C1: centralize checks, don't duplicate them).
        reproducibility_item = self._check_reproducibility(result, loss_table, family, cfg, seed)
        items: list[CheckResult] = [reproducibility_item]

        if family.family_type in ("cvar", "kl"):
            assert isinstance(family, DualAmbiguityFamily)
            items.extend(self._check_dual(result, loss_table, family, cfg, seed))
        else:
            items.append(self._not_applicable("dual_branch_checks", family.family_type))

        if family.family_type == "finite_group":
            groups_fn = getattr(family, "groups", None)
            assert groups_fn is not None
            items.extend(self._check_finite_group(result, loss_table, groups_fn(), cfg))
        else:
            items.append(self._not_applicable("finite_group_checks", family.family_type))

        if family.family_type == "known_weight":
            weights_fn = getattr(family, "weights", None)
            assert weights_fn is not None
            items.extend(self._check_known_weight(result, loss_table, weights_fn(), cfg))
        else:
            items.append(self._not_applicable("known_weight_checks", family.family_type))

        return VerificationReport(items=tuple(items))

    @staticmethod
    def _not_applicable(name: str, family_type: str) -> CheckResult:
        """Build a passed, "not applicable" item for a branch that doesn't need `name`."""
        return CheckResult(
            name=name,
            passed=True,
            detail=f"not applicable to family_type={family_type!r}",
        )

    def _check_reproducibility(
        self,
        result: CalibrationResult,
        loss_table: LossTable,
        family: AmbiguityFamily,
        cfg: CalibrationConfig,
        seed: int,
    ) -> CheckResult:
        """AS §20 item 6: fixed seed => identical `λ̂` (and `empty_flag`)."""
        rerun = WFCRCCalibrator().calibrate(loss_table, family, cfg, seed=seed)
        lambda_matches = abs(rerun.lambda_hat - result.lambda_hat) <= self.tol
        flag_matches = rerun.empty_flag == result.empty_flag
        passed = lambda_matches and flag_matches
        return CheckResult(
            name="reproducibility",
            passed=passed,
            detail=(
                "recomputing under the same seed reproduces lambda_hat and empty_flag"
                if passed
                else (
                    f"recompute gave lambda_hat={rerun.lambda_hat}, "
                    f"empty_flag={rerun.empty_flag}, expected "
                    f"lambda_hat={result.lambda_hat}, empty_flag={result.empty_flag}"
                )
            ),
        )

    def _check_dual(
        self,
        result: CalibrationResult,
        loss_table: LossTable,
        family: DualAmbiguityFamily,
        cfg: CalibrationConfig,
        seed: int,
    ) -> tuple[CheckResult, ...]:
        """AS §20 items 2-5 for the dual (`cvar`/`kl`) branch."""
        n = loss_table.shape[0]
        a_idx, b_idx = Splitter().split(n, cfg.pi, seed)
        n_b = len(b_idx)
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])

        disjoint = len(set(a_idx.tolist()) & set(b_idx.tolist())) == 0
        sizes_match = result.n_a == len(a_idx) and result.n_b == n_b
        split_ok = disjoint and sizes_match
        split_item = CheckResult(
            name="dual_split_disjoint_and_sizes_match",
            passed=split_ok,
            detail=(
                f"A/B disjoint with n_A={len(a_idx)}, n_B={n_b} matching the result"
                if split_ok
                else (
                    f"disjoint={disjoint}, result.n_a={result.n_a} vs {len(a_idx)}, "
                    f"result.n_b={result.n_b} vs {n_b}"
                )
            ),
        )

        theta_by_lambda: dict[float, Any] = {
            float(lam): family.estimate_dual(loss_table.column(float(lam))[a_idx])
            for lam in lambda_grid
        }
        b_tilde = max(family.btil(theta_by_lambda[float(lam)], cfg.B) for lam in lambda_grid)
        b_tilde_finite = np.isfinite(b_tilde)
        expected_b_tilde = result.b_tilde if result.b_tilde is not None else float("nan")
        b_tilde_matches = abs(b_tilde - expected_b_tilde) <= self.tol
        b_tilde_ok = bool(b_tilde_finite and b_tilde_matches)
        b_tilde_item = CheckResult(
            name="dual_b_tilde_finite_and_matches",
            passed=b_tilde_ok,
            detail=(
                f"B_tilde={b_tilde} is finite and matches result.b_tilde"
                if b_tilde_ok
                else f"recomputed B_tilde={b_tilde}, result.b_tilde={result.b_tilde}"
            ),
        )

        transform_ok = True
        for lam in lambda_grid:
            theta = theta_by_lambda[float(lam)]
            l_tilde = family.transform(loss_table.column(float(lam))[b_idx], theta)
            if not np.all(l_tilde <= b_tilde + self.tol):
                transform_ok = False
                break
        transform_item = CheckResult(
            name="dual_transform_bounded_by_b_tilde",
            passed=transform_ok,
            detail=(
                "L_tilde[i,lambda] <= B_tilde for every i in B, lambda"
                if transform_ok
                else "found L_tilde[i,lambda] > B_tilde"
            ),
        )

        def g(lam: float) -> float:
            theta = theta_by_lambda[float(lam)]
            l_tilde = family.transform(loss_table.column(lam)[b_idx], theta)
            r_hat = float(np.mean(l_tilde))
            return (n_b / (n_b + 1)) * r_hat + b_tilde / (n_b + 1)

        g_values = np.array([g(float(lam)) for lam in lambda_grid], dtype=np.float64)
        g_monotone = bool(np.all(np.diff(g_values) <= self.tol))
        g_item = CheckResult(
            name="dual_g_monotone_nonincreasing",
            passed=g_monotone,
            detail=(
                "g(lambda) is non-increasing across the grid"
                if g_monotone
                else "found g(lambda) increasing beyond tol"
            ),
        )

        empty_expected = g_values[-1] > cfg.alpha
        if empty_expected:
            expected_lambda_hat = lambda_max
        else:
            feasible = lambda_grid[g_values <= cfg.alpha]
            expected_lambda_hat = float(np.min(feasible))
        lambda_hat_ok = (
            abs(expected_lambda_hat - result.lambda_hat) <= self.tol
            and empty_expected == result.empty_flag
        )
        lambda_hat_item = CheckResult(
            name="dual_lambda_hat_is_argmin_or_empty_default",
            passed=lambda_hat_ok,
            detail=(
                "lambda_hat is the smallest grid point with g(lambda) <= alpha "
                "(or lambda_max if none)"
                if lambda_hat_ok
                else (
                    f"expected lambda_hat={expected_lambda_hat}, empty={empty_expected}; "
                    f"got lambda_hat={result.lambda_hat}, empty={result.empty_flag}"
                )
            ),
        )

        return (split_item, b_tilde_item, transform_item, g_item, lambda_hat_item)

    def _check_finite_group(
        self,
        result: CalibrationResult,
        loss_table: LossTable,
        groups: tuple[tuple[int, ...], ...],
        cfg: CalibrationConfig,
    ) -> tuple[CheckResult, ...]:
        """AS §20 items 4-5 for the finite-group branch."""
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])

        lambda_hats = []
        n_gs = []
        for indices_seq in groups:
            indices = np.asarray(indices_seq, dtype=np.intp)
            n_g = indices.size
            n_gs.append(n_g)

            def g_group(lam: float, indices: np.ndarray = indices, n_g: int = n_g) -> float:
                r_hat = float(np.mean(loss_table.column(lam)[indices]))
                return (n_g / (n_g + 1)) * r_hat + cfg.B / (n_g + 1)

            if g_group(lambda_max) > cfg.alpha:
                lambda_hats.append(lambda_max)
            else:
                feasible = [float(lam) for lam in lambda_grid if g_group(float(lam)) <= cfg.alpha]
                lambda_hats.append(min(feasible))

        expected_lambda_hat = max(lambda_hats)
        lambda_hat_ok = abs(expected_lambda_hat - result.lambda_hat) <= self.tol
        lambda_hat_item = CheckResult(
            name="finite_group_lambda_hat_matches_max_over_groups",
            passed=lambda_hat_ok,
            detail=(
                "lambda_hat equals max_G lambda_hat_G"
                if lambda_hat_ok
                else f"expected max_G lambda_hat_G={expected_lambda_hat}, got {result.lambda_hat}"
            ),
        )

        per_group = result.diagnostics.get("per_group", [])
        n_g_ok = len(per_group) == len(n_gs) and all(
            int(item["n_g"]) == n_g for item, n_g in zip(per_group, n_gs, strict=True)
        )
        n_g_item = CheckResult(
            name="finite_group_uses_n_g",
            passed=n_g_ok,
            detail=(
                "each group's inflation used its own n_G"
                if n_g_ok
                else f"result diagnostics per_group n_g did not match {n_gs}"
            ),
        )

        return (lambda_hat_item, n_g_item)

    def _check_known_weight(
        self,
        result: CalibrationResult,
        loss_table: LossTable,
        weights: np.ndarray,
        cfg: CalibrationConfig,
    ) -> tuple[CheckResult, ...]:
        """AS §20 item 5 for the known-weight branch: full `n`, weighted mean."""
        n = loss_table.shape[0]
        lambda_grid = loss_table.lambda_grid
        lambda_max = float(lambda_grid[-1])
        weight_sum = float(np.sum(weights))

        def g(lam: float) -> float:
            r_hat = float(np.sum(weights * loss_table.column(lam)) / weight_sum)
            return (n / (n + 1)) * r_hat + cfg.B / (n + 1)

        if g(lambda_max) > cfg.alpha:
            expected_lambda_hat = lambda_max
        else:
            feasible = [float(lam) for lam in lambda_grid if g(float(lam)) <= cfg.alpha]
            expected_lambda_hat = min(feasible)
        lambda_hat_ok = abs(expected_lambda_hat - result.lambda_hat) <= self.tol

        used_full_n = int(result.diagnostics.get("n", -1)) == n
        passed = lambda_hat_ok and used_full_n
        item = CheckResult(
            name="known_weight_uses_full_n",
            passed=passed,
            detail=(
                "weighted CRC used the full calibration set and matches lambda_hat"
                if passed
                else (
                    f"expected lambda_hat={expected_lambda_hat} (used_full_n={used_full_n}), "
                    f"got lambda_hat={result.lambda_hat}, "
                    f"diagnostics.n={result.diagnostics.get('n')}"
                )
            ),
        )
        return (item,)
