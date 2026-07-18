# Experiment Protocol — WFCRC Research Program (Paper 1)

> **Status:** Frozen for first use (MS8), **not yet reviewed by a domain
> expert / PI** — same disclosure posture as `DATASET_SPLIT_POLICY.md` §0.
> **Version:** 1.0. **Date:** 2026-07-18. **Scope:** experiments E1–E12 of
> the frozen `Paper 1 - EXPERIMENT BLUEPRINT.md`, executed against the
> frozen Phase-A dataset suite (`Paper 1 - DATASET SELECTION AUDIT.md`)
> through the `wfcrc` repository as it stands at MS7 freeze.
>
> **What this document is not.** It is not a new experiment design — every
> hypothesis, experiment, dataset assignment, baseline, statistical test,
> and metric named below is transcribed from the already-frozen
> `Paper 1 - EXPERIMENT BLUEPRINT.md` and `Paper 1 - EXPERIMENT EXECUTION
> TRACKER.md` (Research Vault), cross-checked against the frozen
> `wfcrc.evaluation.metrics` implementation for exact formulas. Where the
> vault specifies a procedure only in prose with no exact formula (e.g. the
> statistical tests), this document names the disclosed, textbook
> realization already implemented and frozen in `wfcrc/evaluation/
metrics.py` — the same "disclose, do not silently invent" precedent used
> throughout MS1–MS7 (`CLAIMS_TRACEABILITY.md` §8). No numeric dataset
> split ratio is restated here — see §3's own note and `DATASET_SPLIT_
> POLICY.md`'s review (Task 4 of the MS8 brief) for why.

---

## 1 · Research Objective

### 1.1 Scientific hypotheses (Experiment Blueprint §1, verbatim)

| ID | Hypothesis |
|---|---|
| **H1 Validity** | WF-CRC controls worst-case-over-family risk `≤ α`, finite-sample, for every supported family (CVaR, KL, finite-group, known-weight). |
| **H2 Conditional** | The conditional instantiation (`finite_group`) controls per-group/region risk where global CRC exceeds `α`. |
| **H3 Robust** | Under shift inside the ambiguity ball, WF-CRC keeps risk `≤ α` where vanilla CRC drifts above. |
| **H4 Unification** | One `λ̂` over an f-divergence ball delivers robust **and** (nested) conditional control; it coincides/diverges as predicted by the nesting condition (L4). |
| **H5 Efficiency** | WF-CRC pays a bounded conservativeness (set-size) price — larger than marginal CRC, smaller than worst-case-ball CP baselines. |
| **H6 Architecture** | Single-split with `n_B` inflation is valid; pooled K-fold under-covers; fixed-η is valid but more conservative (empirical confirmation of frozen Proof Obligations P3/P4). |
| **H7 Generality** | Validity holds across ambiguity families and beyond segmentation (classification). |

### 1.2 Hypothesis → experiment mapping

| Hypothesis | Validated by |
|---|---|
| H1 | **E1** (primary), E8, E9 (secondary confirmation across families/hyperparameters) |
| H2 | **E2**, E5 (nesting boundary) |
| H3 | **E3** (synthetic shift), **E4** (real shift) |
| H4 | **E5** |
| H5 | **E6** |
| H6 | **E7** (the load-bearing architecture defense) |
| H7 | **E8**, and E1's cross-modality datasets (Cityscapes + MSD + CIFAR) |

### 1.3 Objective

Demonstrate, with statistical rigor sufficient for NeurIPS/CVPR/MICCAI/IEEE-TMI
(Experiment Blueprint §2, §sufficiency-determination): (i) exact worst-case
validity, (ii) the conditional guarantee, (iii) the robust guarantee, (iv) the
unification of (ii)+(iii), (v) the efficiency price, (vi) that the frozen
architectural decisions (single-split, `n_B` inflation) are empirically
necessary and not an arbitrary implementation choice, and (vii) generality
across ambiguity families and modalities.

---

## 2 · Experimental Overview

### 2.1 Workflow

```
Dataset                  (wfcrc.datasets — DatasetLoader.load(split) -> Dataset)
   |
   v
Model / ScoreProvider     (wfcrc.models.scores.* -- checkpoint inference -> ScoreArray,
   |                       cached read-through by (model_fingerprint, id_))
   v
LossTableBuilder          (wfcrc.datasets.loss_table_builder — frozen, MS4:
   |                       Dataset + ScoreProvider + PredictionSetConstructor +
   |                       LossEvaluator + lambda_grid -> LossTable)
   v
Calibration                (wfcrc.calibration — Splitter divides the calibration
   |                        LossTable into A/B by pi; WFCRCCalibrator.calibrate
   |                        (loss_table, family, cfg, seed=...) -> CalibrationResult
   |                        {lambda_hat, n_a, n_b, b_tilde, r_hat_b, diagnostics})
   v
Prediction Sets             (wfcrc.prediction_sets — PredictionSetConstructor.
   |                         construct(score, lambda_hat) -> boolean set/mask,
   |                         same constructor used at calibration and test time)
   v
Evaluation                  (wfcrc.evaluation — Verifier (STOP-gate) then
                             wfcrc.evaluation.metrics / run_experiment on the
                             held-out test LossTable -> realized risk, per-group
                             risk, set size, coverage, duality gap, effective
                             sizes; wfcrc.runner.ExperimentRunner orchestrates
                             calibrate -> verify -> metrics -> figure -> manifest)
```

Every arrow above is an already-frozen module boundary (`MS6_ARCHITECTURE_SPEC.md`
§0, §2); this protocol adds no new stage and reorders nothing. The **train**
split (base-model training/selection) is consumed only by the Model/
ScoreProvider stage, per `docs/MODEL_POLICY.md` §2 (inference-only) — it never
reappears downstream of the Dataset stage.

### 2.2 Per-run identity

Every experiment run is uniquely identified by `Manifest.config_hash`
(`Config.hash()`, a content hash over `data`/`model`/`sets`/`loss`/`family`/
`calibration`/`runner`/`seed`) plus `family_type`/`family_params`/`seed`
recorded separately in the manifest (`CLAIMS_TRACEABILITY.md` §10 item 1) —
two runs differing only in seed or family parameters are always
distinguishable. See `docs/RESULTS_SCHEMA.md` §3 for the full manifest schema.

---

## 3 · Experiment Matrix (E1–E12)

**Note on dataset split ratios.** This matrix names *which* dataset(s) and
*which* WFCRC role (train/calibration/test) each experiment consumes. It
deliberately does **not** restate the numeric split ratio (e.g. "60/20/20")
for any dataset — that number has exactly one authoritative home,
`docs/DATASET_SPLIT_POLICY.md` §3, and is cited from there. See this
document's §4 note and the MS8 review of that policy (final report, Task 4)
for the justification: a ratio copied into two documents drifts the moment
one is revised and not the other, silently reopening the exact leakage risk
`DATASET_SPLIT_POLICY.md` §1 exists to prevent.

**Note on R (resample count).** `R = 100` (Experiment Blueprint §6, §29) is
the number of independent calibration/test resamples drawn for distributional
reporting (mean ± 95% CI) in every experiment below unless stated otherwise.
A "resample" re-draws the calibration/test partition *within* a dataset's
already-frozen WFCRC pool (`DATASET_SPLIT_POLICY.md` §3) under a fresh
derived seed — it never redraws the train/calibration/test dataset-level
boundary itself (that boundary, and the base model trained against it, is
fixed for the whole experimental campaign per `docs/MODEL_POLICY.md`).

| # | Objective | Hypothesis | Datasets | Model / ScoreProvider | Score-Prov. role | Calibration family | Loss | λ-grid | α values | Eval. metrics | Expected outputs | Tables | Figures | Statistical comparison |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | Marginal worst-case validity | H1 | Cityscapes, MSD-Hippocampus, CIFAR-10 | `cityscapes_segmenter`, `hippocampus_segmenter`, `cifar_classifier` | scores only (no shift) | cvar, kl | miscoverage (cls), FNR (seg) | resolution per §4 | {0.05, 0.10, 0.20} | realized worst-case risk, realized marginal risk, set size, coverage, `n_a`/`n_b`/`n_eff` | realized-risk table + 95% CI, per-α/family | T1 | F1 | one-sided `H0: E[risk] ≤ α` (§5), across R=100 |
| **E2** | Conditional per-region/organ risk | H2 | Cityscapes (class/region groups), MSD-Pancreas (organ groups) | `cityscapes_segmenter`, `pancreas_segmenter` | scores only | finite_group (+ cvar/kl baseline comparator) | FNR | resolution per §4 | 0.10 | per-group realized risk, max-over-group risk, set size | per-group risk table, WF-CRC vs vanilla CRC vs AA-CRC vs sem-CRC | T2 | F2 | paired Wilcoxon signed-rank + Holm across baselines, per group |
| **E3** | Synthetic shift robustness | H3 | Cityscapes-C (severities 1–5) | `cityscapes_segmenter` (unchanged, reused) | scores only, test-side corruption only (never applied to calibration) | cvar, kl | FNR | resolution per §4 | 0.10 | realized risk vs severity | risk-vs-severity curve, WF-CRC vs vanilla CRC vs Cauchois–Duchi/LP vs weighted CRC | T3 | F3 | one-sided `≤ α` per severity + monotone-trend test |
| **E4** | Real shift robustness (external validity of E3) | H3 | Cityscapes→ACDC, CIFAR-10→CIFAR-10.1 | `cityscapes_segmenter` (reused), `cifar_classifier` (reused) | scores only; calibration = source domain, test = target domain | cvar, kl | FNR (seg), miscoverage (cls) | resolution per §4 | 0.10 | realized risk on target domain | realized risk on real shift vs baselines | T3 | F3 (real-shift panel) | one-sided `≤ α` |
| **E5** | Unification / coincide-diverge nesting | H4 | Cityscapes, MSD-Pancreas | `cityscapes_segmenter`, `pancreas_segmenter` | scores only | kl (f-div ball) with varying group mass `P(G)` | FNR | resolution per §4 | 0.10 | simultaneous per-group + worst-case-ball risk; divergence vs `P(G)` | divergence-vs-mass curve; boundary matches nesting threshold `g(ρ)` (Math Spec L4) | T1 | F5 | paired comparison + threshold detection vs `P(G)` |
| **E6** | Efficiency / conservativeness | H5 | Cityscapes, CIFAR-10, MSD-Hippocampus | as E1 | scores only | cvar/kl vs marginal-CRC/worst-case-ball-CP comparators | FNR / miscoverage | resolution per §4 | matched realized validity (not a fixed α — see §4) | mean set size / interval length at matched realized risk; duality-gap proxy | set-size table at matched validity | T1, T7 | F4 | paired Wilcoxon + Holm |
| **E7** | Architecture ablation (single-split / K-fold / n_B / fixed-η) | H6 — **key defense** | Cityscapes, MSD-Hippocampus, CIFAR-10 | as E1 | scores only | cvar/kl, variants: single-split(n_B) / pooled-K-fold / total-n inflation / fixed-η | FNR / miscoverage | resolution per §4 | 0.10 | realized worst-case risk per variant | single-split ≤ α; K-fold & total-n **under-cover** (> α); fixed-η ≤ α but larger sets | T4 | — | one-sided `≤ α` per variant |
| **E8** | Family ablation | H7 | Cityscapes, CIFAR-10 | as E1 | scores only | cvar(β) vs kl(ρ) vs known_weight | FNR / miscoverage | resolution per §4 | 0.10 | validity + set size per family | all families valid; efficiency differs by family/radius | T1 | F1 | one-sided `≤ α` |
| **E9** | Hyperparameter sensitivity | H1 (stability) | Cityscapes, MSD-Hippocampus | as E1 | scores only | cvar/kl, swept `π ∈ {0.2, 0.3, 0.5}`, `ρ`/`β` range, λ-grid resolution | FNR | swept per §4 | 0.10 | validity + set size across settings | validity invariant across sweep; efficiency varies smoothly | T5 | — | report ranges + 95% CI (no single hypothesis test — a sensitivity sweep) |
| **E10** | Complexity / runtime / memory | — (supports H1/H6 cost claims) | Cityscapes, CIFAR-10 (n, T scaling) | as E1 | scores only (cache-warm-up excluded from timing) | cvar, kl | FNR / miscoverage | swept grid resolution `T` | 0.10 | wall-clock, peak memory vs `n`, `T` | matches predicted `O(T·n_A log n_A)` (CVaR) / `O(T·n_A·I)` (KL) + `O(T·n_B)`; calibration ≪ base-model cost | T6 | F7 | fit to predicted complexity (regression, not a significance test) |
| **E11** | Calibration quality | — (supports H5 efficiency framing) | Cityscapes, MSD-Hippocampus | as E1 | scores only | cvar/kl vs temperature/selective-scaling baselines | FNR | resolution per §4 | swept (curve) | ECE, realized-vs-target risk curve | conformal tracks target from above; better per-group calibration than scaling | — | F8 | curves + 95% CI (descriptive, no single test) |
| **E12** | Failure-case & qualitative analysis | — (honesty/interpretability, all H) | Cityscapes, MSD-Pancreas, Kvasir-SEG | as E1/E2, `kvasir_segmenter` | scores only | finite_group, cvar/kl (whichever produced the case under study) | FNR | resolution per §4 | 0.10 (illustrative) | per-pixel risk heatmaps, empty-selection rate, small-`n_G` cases, out-of-ball shift cases | documented conservative/failure regimes; interpretable risk maps; FP-controlled polyp masks | — | F6 | qualitative (no statistical test — see §5.5) |

**Execution order** (Experiment Execution Tracker, unchanged):
`PRE score caches → E1 → E7 → E2, E6, E8 → E3, E5 → E4, E9, E10, E11, E12.`
**Critical path** (gates submission): `PRE(Cityscapes, MSD, CIFAR) → E1 → E7 →
E2 → E3 → E6`.

---

## 4 · Hyperparameter Policy

Every fixed parameter below is an input to calibration, never tuned on
calibration or test data (Algorithm Specification §"config", verbatim: "All
are inputs; none is tuned on the calibration data").

| Parameter | Value / policy | Source | Notes |
|---|---|---|---|
| `α` (target risk level) | `{0.05, 0.10, 0.20}` (E1, E8); `0.10` fixed (E2–E7, E9–E12 unless swept) | Experiment Blueprint §18, Execution Tracker per-E rows | `0.10` is the default working point; `{0.05, 0.20}` bracket it in E1/E8 to show validity is not an artifact of one α. |
| `π` (calibration A/B split fraction) | `0.3` default; swept `{0.2, 0.3, 0.5}` in E9 only | Experiment Blueprint §18 | Orthogonal to the dataset-level split (`DATASET_SPLIT_POLICY.md` §2) — this is the WFCRC-internal `Splitter` fraction, already frozen engineering (MS2). |
| `R` (resample count) | `100` | Experiment Blueprint §6, §29 | See §3 note. |
| CVaR `β` | Per-experiment, swept in E8/E9; not a single global value | Algorithm Specification §"family spec" | No single frozen β exists across all experiments — each experiment's own row in the Execution Tracker treats it as an experiment-local knob (E8's whole purpose is to vary it). Recorded per-run in `Manifest.family_params`. |
| KL `ρ` (radius) | Per-experiment, swept in E3 (shift-ball radius vs corruption severity), E8/E9 | Algorithm Specification §"family spec" | Same status as `β` — an experiment-local knob, not a single global constant. E3/E4 additionally require `ρ` to be set to (or estimated to bound) the actual shift magnitude; the exact estimation procedure for "ρ set to the ball" is not specified numerically anywhere in the frozen vault and is an **open item** (§8 below). |
| `B` (loss upper bound) | `1.0` | `wfcrc.config.schema.CalibrationConfig.B`; all three frozen losses (FNR, FPR, miscoverage) are bounded in `[0, 1]` | Not a tunable — a structural property of the loss functions used. |
| λ-grid | Strictly increasing, resolution swept explicitly in E9; no single global grid resolution is frozen for E1–E8/E10–E12 | `CalibrationConfig.lambda_grid`, Algorithm Specification §"config" | This is the one hyperparameter the frozen vault leaves genuinely open at the numeric level (grid density/range is dataset- and score-range-dependent). **Open item** (§8): a per-dataset default grid (range + point count) must be fixed and recorded in each dataset's `configs/experiment_E*.yaml` before E1 executes, and that grid must not change between an experiment's calibration and any later re-analysis of the same cached logits (E6–E11). |
| Verifier configuration | Default `Verifier()` (Algorithm Spec §20 deterministic checklist) — no experiment overrides its checks | `wfcrc.evaluation.verifier.Verifier` | The STOP-gate (`ExperimentRunner.run`) always runs before any metric is exposed; a failing gate is a **calibrator-gate escalation** (E1 row), never silently downgraded to a warning. |
| Random seeds | One global per-campaign base seed; every component seed (dataset resample, calibration split, bootstrap CI) derived via `wfcrc.utils.seeds.derive_seed(name, base_seed)` | `PROJECT_CONTEXT.md` §9, `docs/reproducibility.md` §2 | No bare `numpy.random.*` call anywhere in the pipeline (project-wide lint policy). The base seed itself is not yet fixed for the E1–E12 campaign — **open item** (§8): must be chosen and recorded (e.g. in a campaign-level config) before E1 executes, then held fixed for the campaign's primary run (R=100 resamples use derived, not identical, sub-seeds). |
| Device selection | CPU (`device="cpu"`), per the already-installed `torch==2.13.0+cpu` build | `requirements/lock.txt`, MS7 record | GPU is **not** currently provisioned in this environment; the Dataset Selection Audit's GPU-day estimates (§6 of that audit) are planning estimates for whichever environment eventually trains/runs the heavier base models (MSD-Pancreas). If a GPU-enabled environment is used for E1–E12 execution, `docs/MODEL_POLICY.md` §7 governs the device-selection and determinism requirements that must hold regardless of device. |
| Deterministic settings | Fixed seeds; no bare global RNG; content-addressed caching (`wfcrc.utils.cache.Cache`) so identical inputs never silently recompute or drift; deterministic figure rendering (fixed `savefig` metadata, fixed SVG hashsalt) | `docs/reproducibility.md` | Same guarantees already exercised by `scripts/reproduce.py`'s golden-file check (MS5), extended to real-data runs via `Manifest`/`ResultBundle`. |

---

## 5 · Statistical Analysis Plan

Defined in advance, before any E1–E12 result exists — none of the choices
below may be revised after observing results without a recorded, dated
protocol deviation (§8 mirrors this constraint for methodology documents
generally).

### 5.1 Confidence intervals

**Nonparametric percentile bootstrap**, `wfcrc.evaluation.metrics.
bootstrap_ci`: for a sequence of `R` per-resample values (risk, set size,
runtime), draw `n_resamples = 2000` bootstrap resamples of the mean, report
the `(1-level)/2` / `1-(1-level)/2` percentiles at `level = 0.95`. Applied
to every reported mean in every table/figure. Deterministic given a derived
seed (`derive_seed("evaluation.metrics.bootstrap_ci", seed)`).

### 5.2 Hypothesis tests

- **Validity (`H0: E[realized risk] ≤ α`):** `one_sided_risk_test` — a
  one-sample, one-sided z-test under the normal approximation across the
  `R` resamples: `z = (mean(risks) - α) / (std(risks)/√n)`, upper-tail
  p-value `1 - Φ(z)`. Also report the raw fraction of resamples with
  realized risk `≤ α` (Experiment Blueprint §12's own descriptive
  complement to the test). **Disclosed gap-fill:** the Blueprint names
  this test only as "one-sided test of `H0: E[risk] ≤ α`" with no formula;
  the z-test above is the standard textbook realization, already frozen in
  `wfcrc/evaluation/metrics.py` (module docstring's own provenance
  disclosure) — not invented for this protocol.
- **Method comparisons (set size, per-group risk):** `paired_wilcoxon` — a
  paired Wilcoxon signed-rank test with the standard tie-corrected normal
  approximation (differences `d_i = a_i - b_i`, zero differences discarded,
  average-rank ties, `z` from `W+`'s standard mean/variance under `H0`,
  two-sided p-value). Same disclosed-gap-fill status as above.
- **Trend under severity (E3):** a monotone-trend test on realized risk vs.
  corruption severity. **Open item** (§8): the frozen vault names "monotone-
  trend test" (Experiment Blueprint §12) without specifying which one;
  `wfcrc.evaluation.metrics` does not currently implement one. This
  protocol recommends a standard, dependency-free choice consistent with
  the rest of this module's own "no scipy" policy — a **Mann-Kendall trend
  test** (rank-based, no distributional assumption, natural pairing with
  the already-adopted Wilcoxon/rank family) — but this is a **methodological
  decision requiring explicit sign-off before E3 executes**, not a silent
  default; see the MS8 final report's gap list.

### 5.3 Multiple-comparison correction

`holm_correct` — exact Holm-Bonferroni step-down procedure (sort ascending,
multiply the `i`-th smallest p-value by `(m-i+1)`, enforce running-maximum
monotonicity, clip to `1.0`) — applied across every baseline compared
against WF-CRC within one experiment (e.g. E2's {vanilla CRC, Gibbs, AA-CRC,
sem-CRC}), and across every group within a per-group comparison (E2, E5).
Corrections are computed **per experiment**, not pooled across E1–E12 —
each experiment's own family of comparisons is Holm-corrected independently,
matching the Blueprint's own per-experiment "Stat test" column (§EXPERIMENT
CATALOG) rather than a single campaign-wide correction.

### 5.4 Effect size measures

**Open item, disclosed rather than silently filled** (§8): the frozen
Experiment Blueprint (§12–13) specifies significance tests and confidence
intervals but names no effect-size measure. This protocol recommends the
standard companion to a Wilcoxon signed-rank test — the **matched-pairs
rank-biserial correlation** `r = Z / √N` — reported alongside every
`paired_wilcoxon` result (E2, E6), so that "statistically significant" and
"practically meaningful" are never conflated in the reported tables. This
recommendation requires explicit sign-off before it is treated as frozen
methodology, per the same discipline as §5.2's trend-test item.

### 5.5 Aggregation methods

- **Primary aggregation:** mean ± bootstrap 95% CI across `R = 100`
  resamples, reported in every table (T1–T7) and shown as CI bands in every
  figure with a resample dimension (F1–F5, F7, F8).
- **Per-group aggregation (E2, E5):** `per_group_risk` computes the mean
  realized risk within each group's row-index set on the test `LossTable`;
  the **max over groups** is the reported "conditional validity" quantity
  (Experiment Blueprint §11).
- **Qualitative aggregation (E12):** no numeric aggregation — failure
  cases are catalogued individually (rare-class/thin-structure regions,
  empty-selection cases `λ̂ = λ_max`, small-`n_G` groups, large duality-gap
  cases, out-of-ball shift), per Experiment Blueprint §24. This is a
  deliberate, disclosed departure from §5.1–5.4's quantitative machinery,
  not an oversight — E12's own purpose is documenting boundaries, not
  testing a hypothesis (Blueprint §24's "Test: qualitative").

### 5.6 Post-hoc discipline

No statistical method listed above may be swapped, added, or have its
parameters (`n_resamples`, `level`, correction family) changed after any
E1–E12 result has been observed. A genuine need to revise this plan (e.g.
resolving §5.2's trend-test open item) is a dated, recorded protocol
amendment — the same discipline `DATASET_SPLIT_POLICY.md` §7 already
requires for split-ratio revisions.

---

## 6 · Evaluation Metrics

All formulas below are transcribed exactly from the frozen, tested
`wfcrc.evaluation.metrics` module (no metric is redefined by this
document — it documents the already-implemented functions for the
methodology record).

### 6.1 Primary metrics

- **Realized worst-case risk** (`realized_worst_case_risk`, dual families
  only): re-derives the family's own worst-case functional on the test
  loss column at `λ̂`, re-estimating a fresh dual parameter `θ` from the
  test data: `c(θ) + mean(t(L_test[:, λ̂]; θ))`. The primary validity
  metric for H1/H3/H4/H6/H7 (E1, E3, E4, E5, E7, E8).
- **Realized marginal risk** (`realized_marginal_risk`): `mean(L_test[:,
  λ̂])` — the unweighted analogue, used as the "vanilla CRC" comparator
  throughout.
- **Per-group realized risk / max-over-group risk** (`per_group_risk`):
  `{g: mean(L_test[group_g_indices, λ̂])}`, maxed over `g` for the
  conditional-validity claim (H2, E2, E5).

### 6.2 Secondary metrics

- **Coverage** (`coverage`): `1 - mean(miscoverage indicator)` =
  `P(Y ⊆ C_λ̂(X))`, empirical.
- **Mean prediction-set size** (`mean_set_size`): `mean(|C_λ̂(score_i)|)`
  — the efficiency metric for H5/E6.

### 6.3 Robustness metrics

- **Realized risk vs. corruption severity** (E3) / **realized risk on
  target domain** (E4): `realized_worst_case_risk`/`realized_marginal_risk`
  evaluated per severity level or per target dataset, with the
  monotone-trend test (§5.2) applied across severities.
- **Duality-gap proxy** (`duality_gap`): `surrogate_risk - realized_risk`
  (calibration-time `g(λ̂)`/`r_hat_b` minus test-time realized risk) — a
  diagnostic for how conservative a given run was, used in E6/E7/E12.

### 6.4 Efficiency metrics

- **Effective sample sizes** (`effective_sizes`): `n_a`, `n_b` (dual
  branch, read from `CalibrationResult`), `n_g_<i>` (per-group, finite-group
  branch), and Kish's effective sample size `n_eff = (Σw)²/Σw²` (known-weight
  branch, when `weights` is given).
- **Wall-clock / peak memory** (E10): measured via the runner's own timing
  instrumentation, cache-warm-up excluded (Execution Tracker's own E10
  "Bugs" warning: "timing includes cache warm-up (must exclude)").
- **Reliability / ECE** (E11): **open item** (§8) — `wfcrc.evaluation.
  metrics` does not currently implement an ECE (Expected Calibration
  Error) function; the Experiment Blueprint (§21) requires one for E11 but
  specifies no binning scheme. This protocol flags the binning scheme
  (equal-width vs. equal-mass bins, bin count) as a methodological decision
  requiring sign-off before E11 executes — not silently defaulted.

---

## 7 · Reproducibility

| Item | Value | Source |
|---|---|---|
| Python | `3.12` (`>=3.12` in `pyproject.toml`; CI pins `3.12` exactly) | `pyproject.toml`, `.github/workflows/ci.yml` |
| Core numeric stack | `numpy==2.5.1`, `scipy==1.18.0` (evaluation/metrics itself is scipy-free by policy; scipy is a transitive/dev dependency only) | `requirements/lock.txt` |
| Visualization | `matplotlib==3.11.0` | `requirements/lock.txt` |
| DL runtime | `torch==2.13.0+cpu` (CPU-only build, confined to `wfcrc/models/` per MS6 Q1) | `requirements/lock.txt`, `MS6_ARCHITECTURE_SPEC.md` §8.1 |
| Image/volume I/O | `pillow==12.3.0`, `nibabel==5.4.2`, `scikit-image==0.24.0`, `imagecorruptions==1.1.2` | `requirements/lock.txt` |
| Hardware (current environment) | CPU-only (no GPU provisioned); the Dataset Selection Audit's GPU-day estimates (§6) are planning figures for base-model *training*, which happens **outside** this repository (`docs/MODEL_POLICY.md` §2) | MS7 record, Dataset Selection Audit §6 |
| Operating system | Development/lockfile generation: Windows (`docs/reproducibility.md` §5, "lockfile was generated on the primary development platform (Windows, Python 3.12)"); CI: `ubuntu-latest` | `docs/reproducibility.md`, `.github/workflows/ci.yml` |
| CUDA | Not applicable — CPU-only `torch` build installed via the PyTorch CPU index; no CUDA toolkit is part of this project's environment today | `requirements/lock.txt` install-command note |
| Dependency lock | `requirements/lock.txt`, the fully pinned closure for `pip install -e ".[dev,docs]"`; `make install-locked` reproduces it exactly. CI uses loose ranges instead (`docs/reproducibility.md` §5) — a deliberate split, not a drift risk, since CI's job is catching upstream breakage, not reproducing a past result. | `docs/reproducibility.md` |
| Random seed policy | One base seed per campaign; every component seed derived via `derive_seed(name, base_seed)`; no bare global RNG. The campaign base seed itself is an **open item** (§8, §4) — must be fixed and recorded before E1 executes. | `PROJECT_CONTEXT.md` §9 |
| Checkpoint versioning | `checkpoint_fingerprint` — a content-hash of the checkpoint **file bytes** (`wfcrc.utils.io.content_hash`), used as the cache key for `ScoreProvider` results. **Disclosed caveat** (already recorded at MS7 freeze): `torch.save` is not guaranteed byte-identical across independent calls for identical tensor values, so re-saving an unchanged checkpoint can change its fingerprint — this matches the fingerprint's actual purpose (cache-key identity of *this specific file*), not "the weights" in the abstract, and must not be read as a stronger claim than that. | `PROJECT_CONTEXT.md` §7 (MS7 section), `wfcrc/models/checkpoint.py` |
| Provenance capture | Every `Manifest` records `config_hash`, `seed`, `family_type`/`family_params`, `git_commit` (`get_git_commit`), and an environment fingerprint (`get_environment_fingerprint`) — sufficient to fully explain a result: parameters, randomness, and code/environment. | `docs/reproducibility.md` "Provenance capture" |
| Golden-file regression | `scripts/reproduce.py` / `make reproduce` — a fixed-seed synthetic reference experiment, diffed against `tests/fixtures/reproduce_golden.json` within `1e-9` absolute tolerance; run as a fourth CI job. Real-data E1–E12 runs are a separate, additional reproducibility surface — this golden file does not itself validate any real-dataset run. | `docs/reproducibility.md`, `PROJECT_CONTEXT.md` §7 (MS5 section) |

---

## 8 · Threats to Validity

Transcribed and specialized from Experiment Blueprint §30, plus items
specific to this repository's current state.

### 8.1 Internal validity

- **Calibration leakage.** Mitigated by the frozen A1 hygiene gate
  (`assert_split_disjoint`/`SplitLeakageError`) plus `docs/DATASET_SPLIT_
  POLICY.md`'s scientific partitioning policy (§2 of that document) and
  `docs/MODEL_POLICY.md`'s checkpoint-provenance requirement (R4/R-CKPT1).
- **Monotonicity violations.** Mitigated by the frozen `Verifier` STOP-gate
  (Algorithm Spec §20), which halts a run before any metric is exposed if
  the deterministic checklist fails.
- **Dual non-convergence.** Mitigated by the frozen, disclosed fixed-η
  fallback (`KLFamily.estimate_dual`, `CLAIMS_TRACEABILITY.md` item 1) —
  itself the object of E7's ablation, not merely a silent safety net.
- **Open numeric gaps (this document, §4/§5.2/§5.4/§6.4)** — λ-grid
  resolution, campaign base seed, the E3 trend test, the E6 effect-size
  measure, and E11's ECE binning scheme are not yet numerically frozen.
  Until each is resolved and recorded, no experiment consuming that
  specific choice may be executed at full scale (see §9 Self-Audit gaps).

### 8.2 External validity

- **Dataset representativeness.** The Phase-A suite (seven datasets) is a
  deliberately minimal, scientifically-defensible subset of a larger
  candidate pool (Dataset Selection Audit §3); Phase-B datasets (ADE20K,
  ImageNet(-V2), MSD-BrainTumour/Prostate) strengthen but are not required
  for any claim in this protocol.
- **Shift realism.** E3 (synthetic) is explicitly paired with E4 (real
  shift) so no claim rests on synthetic corruption alone.
- **Radius mis-specification.** `ρ`'s relationship to the *actual* shift
  magnitude in E3/E4 is reported as a sensitivity (E9), not assumed exact;
  see §4's open item on `ρ` estimation for shift experiments.

### 8.3 Construct validity

- **Worst-case risk as the right target** — the framework's own central
  claim (H1); not independently re-litigated here.
- **Ground-truth label quality.** Flagged, not resolved, per the
  Experiment Blueprint's own scope: ontological label-space uncertainty is
  explicitly **Paper 2/3** territory (`PROJECT_CONTEXT.md` §2), out of
  scope for Paper 1's E1–E12.

### 8.4 Statistical validity

- **Monte Carlo error.** Bounded and reported via `R=100` and the 95%
  bootstrap CI (§5.1) — never a point estimate alone.
- **Multiple comparisons.** Holm-corrected per experiment (§5.3).
- **Post-hoc test/metric selection.** Explicitly disallowed (§5.6); every
  open item in §5.2/§5.4/§6.4 must be resolved *before*, not after, the
  relevant experiment's results are observed.

### 8.5 Reproducibility

- **The single stochastic dataset-generation and calibration split.**
  Mitigated by fixed, recorded seeds and the fact that WF-CRC's finite-sample
  validity guarantee holds for *any* fixed split (Algorithm Spec P3/P4) —
  reproducibility does not depend on hitting one "lucky" split.
- **Environment drift.** Mitigated by the pinned lockfile and the
  golden-file regression check (§7); real-data runs additionally record a
  full `Manifest` provenance block per run.

---

## Connections

`docs/DATASET_SPLIT_POLICY.md` · `docs/MODEL_POLICY.md` ·
`docs/RESULTS_SCHEMA.md` · `MS6_ARCHITECTURE_SPEC.md` ·
`CLAIMS_TRACEABILITY.md` · `PROJECT_CONTEXT.md` · Research Vault:
`Paper 1 - EXPERIMENT BLUEPRINT.md`, `Paper 1 - EXPERIMENT EXECUTION
TRACKER.md`, `Paper 1 - DATASET SELECTION AUDIT.md`, `Paper 1 - ALGORITHM
SPECIFICATION.md`
