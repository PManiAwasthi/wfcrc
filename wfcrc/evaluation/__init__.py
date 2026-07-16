"""Deterministic verification + realized-risk metrics (Algorithm Spec §20, MS4 spec).

Public API: :class:`~wfcrc.evaluation.verifier.Verifier` and its
:class:`~wfcrc.evaluation.verifier.CheckResult` /
:class:`~wfcrc.evaluation.verifier.VerificationReport` result types
(Implementation Blueprint §6, `verify.Verifier`); the
:mod:`wfcrc.evaluation.metrics` free functions (`MetricSuite`, M13) plus
their :class:`~wfcrc.evaluation.metrics.CI` /
:class:`~wfcrc.evaluation.metrics.TestResult` result types; and
:func:`~wfcrc.evaluation.experiment.run_experiment` /
:class:`~wfcrc.evaluation.experiment.ExperimentReport` (the reduced,
dataset-free experiment-execution entry point). Visualization
(:mod:`wfcrc.visualization`, Implementation Blueprint's `viz.Plotter`) and
the full sweep/checkpointing runner (:class:`wfcrc.runner.ExperimentRunner`)
are implemented in their own packages (MS5, complete), not this one — this
package's own scope stops at calibrate+verify+metrics; see each module's
docstring for the exact boundary.
"""

from __future__ import annotations

from wfcrc.evaluation.experiment import ExperimentReport, run_experiment
from wfcrc.evaluation.metrics import (
    CI,
    TestResult,
    bootstrap_ci,
    coverage,
    duality_gap,
    effective_sizes,
    holm_correct,
    mean_set_size,
    one_sided_risk_test,
    paired_wilcoxon,
    per_group_risk,
    realized_marginal_risk,
    realized_worst_case_risk,
)
from wfcrc.evaluation.verifier import CheckResult, VerificationReport, Verifier

__all__ = [
    "CI",
    "CheckResult",
    "ExperimentReport",
    "TestResult",
    "VerificationReport",
    "Verifier",
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
    "run_experiment",
]
