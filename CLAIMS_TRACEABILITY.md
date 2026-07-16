# Claims Traceability (repository-side)

The Research Vault's `Paper 1 - CLAIMS TRACEABILITY MATRIX.md` is frozen
and read-only — this file does not modify it. It exists to record
**implementation-level** facts (deviations, gap-fills, fallback
activations) that a future reader of the vault's matrix would want to
find, without touching the frozen document itself. Cross-references below
use the vault matrix's own claim IDs (`CL-#`) and gate IDs.

## 1 · Fixed-η fallback (Algorithm Spec §15, F-4)

**What:** `wfcrc.ambiguity.kl.KLFamily.estimate_dual` detects when the 1-D
dual-estimation search returns exactly the `eta_min` boundary — which, by
convexity, occurs if and only if the profiled objective `h` is
non-decreasing on all of `[eta_min, infinity)` (a degenerate/near-constant
`losses_a_col`, including the `n_A=1` singleton case). When detected, the
data-driven `eta` is discarded and replaced by `fallback_eta`, a fixed,
data-independent constructor parameter (default `1.0`); `mu` is still
recomputed via the same closed-form profile step at that fixed `eta`.

**Why this preserves validity, not just "doesn't crash":** weak duality
(Math Spec §5) holds for *any* `eta > 0` paired with its closed-form-optimal
`mu` — fixing `eta` trades tightness for robustness, never validity. This
is not a new mode: **CL-8**'s own statement already names it — "single-split
**(or fixed-η)** with `n_B` inflation is exact" — and Math Spec §6 states
"Fixed-η (no estimation) and single-split are special cases" of the
decoupling scheme. The implementation activates this pre-existing frozen
special case automatically at the documented failure boundary, rather than
requiring the caller to select it manually.

**Before this change:** the boundary case raised `FamilyError` (F-3,
"unbounded transform"), which is a valid but overly conservative reading —
it discarded a case the frozen spec explicitly provides a fallback for
(§15: "Dual non-convergence ⇒ use a fixed data-independent η"). `FamilyError`
is still raised if the fallback's own transform is *also* non-finite (a
genuine, unfixable F-3 case — see `test_kl_fallback_does_not_prevent_genuine_f3_rejection`).

**Code:** `wfcrc/ambiguity/kl.py` (`KLFamily.__init__`'s `fallback_eta`
parameter, `KLFamily.estimate_dual`'s boundary check).
**Tests:** `tests/unit/ambiguity/test_kl.py` (`test_fallback_activation_is_exact_boundary_equality`,
`test_fallback_behaviour_is_deterministic`, `test_near_constant_calibration_block_triggers_fallback`,
`test_fallback_is_not_triggered_for_a_genuine_interior_optimum`,
`test_fallback_preserves_weak_duality_pointwise_domination`, `test_fallback_eta_must_be_positive`),
`tests/unit/calibration/test_calibrator.py` (`test_kl_degenerate_a_block_succeeds_via_fixed_eta_fallback`,
`test_kl_singleton_a_block_succeeds_via_fixed_eta_fallback`,
`test_kl_singleton_a_block_fallback_eta_is_configurable`,
`test_kl_fallback_does_not_prevent_genuine_f3_rejection`).

## 2 · Intentional API signature differences from frozen text

Both deviations exist because `sets`/`prediction_sets`
(`PredictionSetConstructor`) is out of MS2 scope (a later milestone); both
were necessary to implement `losses`/`calibration` standalone without it.

- **`LossEvaluator.assert_monotone`** — MS2 Implementation Spec §C4 gives
  `assert_monotone(constructor, score, label, λ_grid)`, which requires a
  `PredictionSetConstructor`. The Implementation Blueprint §6 gives the
  simpler `assert_monotone(grid)`. The implementation follows the
  Blueprint: `assert_monotone(losses_by_lambda: Sequence[float]) -> bool`
  checks a precomputed λ-ordered loss sequence directly, with no
  dependency on any set constructor.
- **`WFCRCCalibrator.calibrate`** — Implementation Blueprint §6 gives
  `calibrate(loss_table, family, α, cfg) -> CalibrationResult`. The
  implementation gives `calibrate(loss_table, family, cfg, *, seed) ->
  CalibrationResult`: `α` is folded into `cfg`
  (`wfcrc.config.schema.CalibrationConfig` already carries `alpha`, `B`,
  `pi`, `lambda_grid` as one frozen, MS1-validated object — passing it
  once avoids a redundant separate `α` argument), and `seed` is pulled out
  as an explicit keyword argument (Algorithm Spec §8's own
  `wf_crc_single_split` pseudocode signature takes `seed` as a distinct
  parameter alongside `alpha`/`pi`/`B`, which this mirrors).

Neither deviation changes any claim's mathematical content; both are
recorded here so a reader reconciling code against **CL-2**, **CL-8**, or
**CL-10** does not mistake them for unexplained drift.

## 3 · FPR loss: documented gap-fill (not a frozen formula)

**CL-10** ("generality across families & modalities") and the Algorithm
Spec's `LossEvaluator` catalog name `FPRLoss` as a required concrete loss,
but **no document in the Research Vault gives FPR a closed-form
definition** — only the informal phrase "FPR/accepted-FP proportion"
(`Paper 1 - MS2 IMPLEMENTATION SPEC.md`, §C4), unlike FNR and Miscoverage,
which have exact frozen formulas in `05 Loss Functions/`.

`wfcrc/losses/fpr.py` implements the standard textbook False Positive Rate,
`ℓ = |C_λ(X) \ Y| / |Y^c|`, structurally mirroring the frozen FNR formula
with `Y` and its complement swapped, and states this provenance explicitly
in its module docstring. This is disclosed here so it is traceable at the
project level, not only discoverable by reading the source file.

## 4 · MS3 scope resolution (confirmed with the user before implementation)

The MS3 task prompt asked for `wfcrc/prediction_sets/{base,classification,
segmentation,multilabel}.py` and `wfcrc/calibration/{pipeline,verifier,
validator,interfaces}.py` verbatim. Cross-checking these paths against the
frozen Implementation Blueprint, the MS2/MS3/MS4 Implementation
Specifications, and `PROJECT_CONTEXT.md`'s own §7 numbering caveat surfaced
three genuine mismatches, each resolved by asking the user rather than
guessing (per the "if something appears missing, stop and document it"
rule):

1. **`multilabel.py`:** no vault document gives a multilabel prediction-set
   constructor a formula — the MS2 Implementation Spec (§C3) names only
   `ThresholdSets` (LAC) and `MorphologicalSets`; "multi-label pixels" is
   listed once as a `ThresholdSets` *edge case*, never as its own
   constructor. **Resolution:** `prediction_sets/` contains `base.py`,
   `classification.py` (`ThresholdSets`), and `segmentation.py`
   (`MorphologicalSets`) only; no `multilabel.py`.
2. **`Verifier`'s location:** the Implementation Blueprint places `verify`
   as a top-level sibling of `calibration` (§2's dependency graph:
   `calibration.calibrator ──► verify`, i.e. `verify` *consumes*
   calibration's outputs), and `PROJECT_CONTEXT.md` §6 already reserves
   the empty `wfcrc/evaluation/` directory for exactly this (vault's own
   MS4, M12). **Resolution:** `Verifier` lives in
   `wfcrc/evaluation/verifier.py`, not `wfcrc/calibration/verifier.py`.
3. **`pipeline.py`/`validator.py`/`interfaces.py`:** no vault document
   names a `validator` module (the MS4 spec explicitly warns the
   `Verifier` should "centralize, not duplicate" checks — a second
   validator would violate that), and no vault document defines
   `interfaces.py` as a physical file (the frozen ABCs already live at
   each subpackage root, matching the existing `losses/base.py`/
   `ambiguity/base.py` pattern). The closest analog to "pipeline",
   `runner.ExperimentRunner`, is specified to depend on datasets/models/
   viz — explicitly out of this milestone's scope. **Resolution:** built
   only a minimal `calibration/pipeline.py` (`run_calibration_pipeline`)
   composing `LossTable → WFCRCCalibrator.calibrate` with an *optional*,
   structurally-typed (`VerifierLike` `Protocol`, not a runtime import of
   `wfcrc.evaluation`) verification step — this keeps the module
   dependency graph acyclic while still giving the "complete executable
   pipeline" real value beyond calling `calibrate()` directly. No
   `validator.py`, no `interfaces.py`.

These three points were put to the user as an `AskUserQuestion` before any
code was written; all three resolutions above were the options the user
selected.

## 5 · `MorphologicalSets`: documented erosion-direction gap

**What:** the MS2 Implementation Spec (§C3 item 5/7) names `direction`
(`dilation`/`erosion`) as a `MorphologicalSets` config knob "consistent
with the loss's monotonicity," but gives no formula for either direction —
only "grow/shrink mask by a monotone structuring element."
`wfcrc.prediction_sets.segmentation.MorphologicalSets` implements
`direction="dilation"` in full (`C_λ = dilate(M₀, ⌊λ⌋)`, standard iterated
binary dilation — nested by construction since each step ORs the previous
mask in and `⌊λ⌋` is non-decreasing).

**Why erosion is not implemented, not just "harder":** a literal reading of
"erosion" applied to the same seed mask that still produces a P-1-nested
(growing) family collapses, by the standard erosion/dilation duality
(`erode(A) = complement(dilate(complement(A)))` for a symmetric structuring
element), to one of two outcomes — either identical behavior to plain
dilation (no distinct construction at all), or a family with the wrong
boundary behavior (non-empty at `λ_min`, unbounded growth, rather than the
empty/full boundary the MS2 spec's own edge cases name explicitly). A
genuinely distinct, well-behaved erosion-direction construction would need
information the frozen `construct(score, λ)` signature does not carry
(e.g. a normalizing `λ_max`) and no vault document supplies a formula for
it. `MorphologicalSets(direction="erosion")` therefore raises
`SetConstructionError` at construction time with this reasoning, rather
than guessing.

**Code:** `wfcrc/prediction_sets/segmentation.py` (module docstring,
`MorphologicalSets.__init__`). **Tests:**
`tests/unit/prediction_sets/test_segmentation.py::test_init_rejects_erosion_direction_as_a_documented_gap`.

## 6 · `Verifier`: malformed-family checks centralized, not duplicated

`Verifier.check_calibration` runs `_check_reproducibility` first, which
unconditionally calls `WFCRCCalibrator.calibrate(...)` — the same call
that already validates a family actually implements the interface its
declared `family_type` promises (raising `FamilyError` if not, per
`calibrator.py`'s own dispatch guard). `check_calibration`'s own dispatch
therefore uses plain `assert`s (for `mypy`'s type narrowing only) rather
than a second `raise FamilyError(...)`, per MS4 spec's own instruction to
centralize checks rather than duplicate them. A malformed family passed to
`check_calibration` still raises `FamilyError` — from the
`WFCRCCalibrator.calibrate` call inside `_check_reproducibility` — so
behavior is unchanged; only the (dead, unreachable-by-construction)
duplicate raise was removed. **Tests:**
`tests/unit/evaluation/test_verifier.py::test_check_calibration_raises_family_error_for_malformed_{dual,finite_group,known_weight}_family`.

## 7 · MS4 scope resolution (confirmed with the user before implementation)

The MS4 task prompt asked for `datasets/`, `metrics/`, "experiment
execution", "result aggregation", "statistical utilities", "experiment
configuration", "benchmark orchestration", and "reporting utilities" — a
mix spanning the vault's own MS2 tail (M3/M4/M7), MS4 (M13 metrics), and
MS5 (M14 viz, M15 runner). Four genuine scope conflicts were resolved by
asking the user before any code was written (per the "if something appears
missing, stop and document" rule):

1. **`datasets/`:** the Experiment Blueprint (§3) names real datasets
   (Cityscapes, ADE20K, MS-COCO, MSD, Kvasir-SEG/CVC-ClinicDB, CIFAR/
   ImageNet, etc.) with real pretrained-model score providers, none of
   which are present in this environment. **Resolution:** ABC contracts
   only (`Dataset`, `DatasetLoader`, `ScoreProvider`) — no concrete loader
   or provider for any specific dataset/model. The one exception is
   `LossTableBuilder` (M7): its assembly logic ("iterate examples × λ,
   call sets→loss") is fully mechanical and uses only already-frozen
   concrete pieces (`PredictionSetConstructor`, `LossEvaluator`), so it is
   implemented concretely and tested against small synthetic `Dataset`/
   `ScoreProvider` test doubles — flagged explicitly as the one departure
   from a literal "ABCs only" reading.
2. **"Experiment execution"/"benchmark orchestration":** the vault's
   `runner.ExperimentRunner` (M15) is specified to orchestrate
   `load config → build/load loss tables → calibrate → verify → metrics →
   plot → write manifest` — i.e. it depends on the same missing dataset/
   model stage *and* on `viz` (M14), excluded from this milestone.
   **Resolution:** a reduced pipeline extension,
   `wfcrc.evaluation.experiment.run_experiment`, composing the already-
   frozen `run_calibration_pipeline` (MS3) with the new `metrics` module
   into a structured `ExperimentReport`, given an already-built calibration
   `LossTable` and test `LossTable`. No dataset loading, no plotting, no
   sweeps/checkpointing/resume — those remain the full MS5 runner's scope.
3. **Negative-control ablations** (pooled K-fold WF-CRC, total-`n`
   inflation WF-CRC — Experiment Blueprint §9/§23, also the vault's own
   MS3 exit gate G-iv, discovered during scoping to have never actually
   been implemented, not even as a test-only harness). **Resolution:**
   test-only harnesses (`tests/unit/calibration/test_negative_controls.py`),
   not library code — matching the MS3 spec's own "test-only harnesses"
   phrasing literally.
4. **"Reporting utilities":** given `viz.Plotter` (M14) is out of scope.
   **Resolution:** structured data only — `ExperimentReport.to_dict()`
   (JSON/dict), not the M14 viz module's figure/CSV data sidecars.

These four points were put to the user as an `AskUserQuestion` before any
code was written; all four resolutions above were the options the user
selected.

## 8 · `metrics`: statistical-test provenance disclosure (not a frozen formula)

**CL-10**-style generality and the Experiment Blueprint (§12) name
`one_sided_risk_test`, `paired_wilcoxon`, and `holm_correct` as required
statistical utilities, but **no document in the Research Vault gives any
of the three an exact formula** — only the textual descriptions
("one-sided test of `H0: E[realized risk] ≤ α`"; "paired Wilcoxon
signed-rank... with Holm correction"). Unlike the frozen single-split
procedure (Algorithm Spec §8's verbatim pseudocode), these are standard,
generic statistics, not part of the paper's own mathematics.

`wfcrc/evaluation/metrics.py` implements each as the standard textbook
realization, disclosed explicitly in the module docstring and here (same
pattern as the FPR loss gap-fill, §3 above): a one-sample one-sided z-test
under the normal approximation; the Wilcoxon signed-rank test with the
standard tie-corrected normal approximation; the exact Holm-Bonferroni
step-down algorithm. All three avoid scipy (`math.erf` supplies the exact
normal CDF), consistent with the project's dependency-light policy.

**Code:** `wfcrc/evaluation/metrics.py`. **Tests:**
`tests/unit/evaluation/test_metrics.py`.

## 9 · Two bugs caught during MS4 self-review (fixed before freeze)

Recorded here because both are the kind of drift a future reader
reconciling code against tests might otherwise mistake for intentional
behavior:

1. **`one_sided_risk_test`'s zero-variance guard used exact float
   equality** (`std == 0.0`) instead of a tolerance. `np.std([0.2, 0.2,
   0.2], ddof=1)` is `~3.4e-17`, not exactly `0.0` — the guard silently
   let a degenerate (effectively constant) `risks` array through, dividing
   by a near-zero standard deviation and producing a meaningless huge
   `z`-statistic instead of raising. Fixed by comparing against
   `_NEAR_ZERO_STD_TOL = 1e-9` instead of `0.0`. Caught by
   `test_one_sided_risk_test_rejects_zero_variance` failing during the
   test-writing pass, before freeze.
2. **A test-tuning bug in the negative-control harness** (not a bug in
   shipped `wfcrc/` code): `_search_lambda_hat`/`_total_n_lambda_hat`/
   `_pooled_k_fold_lambda_hat` originally read a module-level `_ALPHA`
   constant directly instead of taking `alpha` as an explicit parameter —
   a "no hidden globals" violation. While exploring parameterizations for
   `test_pooled_k_fold_and_total_n_under_cover_relative_to_single_split`,
   this silently decoupled the negative-control procedures' `alpha` from
   the correct procedure's `cfg.alpha` in an ad hoc debug script, produced
   a misleading result, and was caught by re-deriving the comparison with
   `alpha` properly threaded through before the test was finalized. Fixed
   by making `alpha` an explicit parameter throughout.

## 10 · MS4 independent audit findings, fixed before freeze

A post-implementation independent audit (adversarial review against the
Framework/Math/Algorithm Specs, Implementation/Experiment Blueprints, and
the repository itself) found four issues, all corrected:

1. **`ExperimentReport.config_hash` omitted `seed` and `family` entirely**
   (HIGH). `run_experiment(..., seed: int, ...)` never stored `seed` on the
   report, and `config_hash` covered only `{alpha, B, pi, lambda_grid}` —
   not the ambiguity family's type or parameters. Since `seed` is the only
   stochastic quantity in the entire procedure (Algorithm Spec §17) and the
   report's own docstring cites Implementation Blueprint §17's
   "seeds"-in-every-manifest requirement, this meant the artifact could not
   actually serve the reproducibility role it claimed. **Fixed:**
   `ExperimentReport` gained `seed: int`, `family_type: str`,
   `family_params: dict[str, Any]` fields (via a new, generic, non-invasive
   `_family_params(family)` helper using `vars(family)` — no change to any
   frozen `wfcrc.ambiguity` class); `config_hash` now covers all of `cfg`,
   `seed`, `family_type`, and `family_params`. Two experiments differing in
   seed, family type, or family parameters now get different hashes.
   `to_dict()` exposes the three new fields directly. Additive change: no
   existing field removed or renamed, `run_experiment`'s parameter list
   unchanged.
2. **Bug in the pooled-K-fold negative-control harness's `B̃` accumulation**
   (MEDIUM, test-only). `_pooled_k_fold_lambda_hat` updated
   `b_tilde = max(b_tilde, ...)` *inside* the same loop that immediately
   consumed it to build `g_values[j]`, so early λ-grid points used a
   partially-accumulated (too-small) bound rather than the full two-pass
   global max its sibling `_total_n_lambda_hat` already computed correctly.
   Confirmed to flip the selected `lambda_hat` in ~13% of resamples in
   isolation, but the effect on the test's own R=150-resample aggregate
   comparison was negligible (0.137 → 0.132 mean realized risk; the
   qualitative conclusion, and the "roughly 4-5x" figure quoted in the
   module docstring, both survive). **Fixed:** rewritten as an explicit
   two-pass computation (collect every fold's theta first, take the global
   max, then build every `g_values[j]`), matching `_total_n_lambda_hat`'s
   existing pattern. Test-only; no public API changed.
3. **Misattributed formula citation** (MEDIUM). `effective_sizes`'s
   docstring cited Kish's `n_eff = (Σw)²/Σw²` to "Algorithm Spec §20" —
   that section (the §20 verification checklist) does not contain this
   formula; it is from the vault's own MS4 Implementation Spec §C2 item 5.
   **Fixed:** citation corrected; the formula itself was never wrong and is
   unchanged.
4. **Undisclosed same-sample optimism in `realized_worst_case_risk`**
   (MEDIUM-LOW). The function re-estimates the family's dual parameter on
   the same test sample it then measures — the ordinary, generic optimism
   of any plug-in/empirical-minimum estimator, not the specific same-data
   *threshold-selection* failure mode Math Spec §12 item 3 warns against
   (no threshold is selected here; the function makes no validity claim).
   Still undisclosed anywhere, and this is the metric the Experiment
   Blueprint's E1 hypothesis is evaluated against. **Fixed:** a
   "Descriptive-statistic caveat" paragraph added to the function's
   docstring explaining the distinction and the expected direction of the
   bias. Documentation only; no implementation change.

Three additional low-severity/observational items from the audit
(`bootstrap_ci` not routing through `wfcrc.utils.numerics.quantile`; the
new `_NEAR_ZERO_STD_TOL` tolerance constant living in `metrics.py` rather
than `wfcrc/constants.py`; four independent
`isinstance(family, DualAmbiguityFamily)` dispatch sites) were explicitly
deferred to future cleanup per the user's own scoping of this pass, and
remain open, not open *problems* — just optional polish.

## 11 · MS5 scope resolution (confirmed with the user before implementation)

The MS5 task prompt asked for the full Implementation Blueprint reading of
`runner.ExperimentRunner.run(config)->ResultBundle`, orchestrating "load
config -> build/load loss tables (cal+test) -> calibrate -> verify ->
metrics -> plot -> write manifest." Cross-checking this against the actual
repository surfaced one genuine scope conflict, resolved by asking the user
before any code was written (per the "if something appears missing, stop
and document" rule):

**Loss-table stage.** Building loss tables from `config.data`/`.model`/
`.sets`/`.loss` requires resolving those name strings to concrete
`DatasetLoader`/`ScoreProvider`/`PredictionSetConstructor`/`LossEvaluator`
instances. No such dataset/model registry exists anywhere in this
repository (`wfcrc.datasets` is ABC-contracts-only per the MS4 scope
decision — no real dataset/model is available in this environment) — the
same blocking condition MS4 hit with `run_experiment`. **Resolution
(user-selected, of three options presented):** `ExperimentRunner.run` takes
already-built `cal_loss_table`/`test_loss_table` `LossTable` objects
directly, exactly like `run_experiment` already does. Only `config.family`
is resolved to a concrete `AmbiguityFamily`, via the already-frozen
`wfcrc.ambiguity.FAMILIES` registry (MS2, §2 above already documents that
this registry exists) — no new dataset/sets/losses registry was built. The
other two options considered and not selected: building new name->class
registries for `sets`/`losses`/`family` now (still would have needed
dataset/model injection regardless, and touches three already-frozen
packages); or implementing only checkpointing/sweep/resume primitives
without presenting `run(config)` as covering the loss-table stage at all
(functionally identical to the selected option, differing only in framing).

This is documented here because a future reader reconciling
`ExperimentRunner.run`'s signature against the Implementation Blueprint's
literal `run(config)->ResultBundle` text should not mistake the missing
`config.data`/`.model` resolution for an oversight.

## 12 · MS5 implementation-level disclosures (not open issues)

1. **The `g`-curve is the only figure a single `ExperimentRunner.run` call
   produces.** Most of the Experiment Blueprint's F1-F8 figures (§26)
   aggregate *many* calibration runs (risk vs `alpha`, vs severity, vs
   group) and have no canonical single-run form — they are downstream of a
   sweep's collected metrics, not `ExperimentRunner`'s concern.
   `wfcrc.visualization.plots.plot_g_curve` is the one figure genuinely tied
   to a single run; `ExperimentRunner.run` recomputes `g(lambda)` across the
   whole grid for it (`wfcrc.runner.runner._dual_g_curve`), mirroring
   exactly the frozen dual-branch computation `WFCRCCalibrator`/`Verifier`
   already perform (Algorithm Spec §7 steps 3-6, same family API) —
   read-only, `lambda_hat` itself is never re-selected. Finite-group/
   known-weight families have no `g`-curve concept in the frozen spec and
   produce no figures. **Code:** `wfcrc/runner/runner.py`
   (`_dual_g_curve`, `ExperimentRunner._render_figures`). **Tests:**
   `tests/unit/runner/test_runner.py::TestDualGCurve`,
   `TestRunNonDualFamilies`.
2. **The verify STOP-gate is enforced by discarding, not skipping,
   metrics on failure.** `run_experiment` (MS4, frozen) always computes
   metrics and attaches a `VerificationReport`; it does not itself gate
   metric exposure on verification passing (nothing in its own scope
   requires that — that responsibility belongs to the runner, a higher
   layer, per the Implementation Blueprint's own module dependency graph).
   `ExperimentRunner.run` reuses `run_experiment` wholesale (composing
   frozen code, not reimplementing calibrate+verify+metrics), then checks
   `report.verification.passed` *before* checkpointing or returning
   anything, raising `VerificationError` if it failed. Metrics are computed
   once, internally, as an unavoidable consequence of that reuse, but a
   failed gate means they are never checkpointed, never written to the
   manifest, and never returned to the caller — "no downstream metrics" in
   effect (MS5 spec C2 item 8), even though they were computed once and
   discarded. **Code:** `wfcrc/runner/runner.py`
   (`ExperimentRunner.run`'s `_compute_experiment` closure). **Tests:**
   `tests/unit/runner/test_runner.py::TestVerifyStopGate`.
3. **Checkpoint stage boundaries are reduced to `"experiment"` and
   `"figures"`**, not the full "scores, loss table, dual, calibration,
   metrics" list the MS5 spec names (C2 item 5) — because "scores"/"loss
   table"/"dual" boundaries do not exist in this milestone's scope (loss
   tables are injected, not built; the dual estimate is internal to
   `WFCRCCalibrator.calibrate`, not separately exposed), and reusing
   `run_experiment` wholesale bundles calibration+verification+metrics into
   one checkpointable unit rather than three separate ones. This is the
   same kind of disclosed scope reduction already applied to
   `run_experiment` itself (§4 item 2 above); resumability is still genuine
   for both boundaries that actually exist (see
   `tests/unit/runner/test_runner.py::TestResume`, including a
   figures-only-remaining resume case).
4. **`ResultBundle.verification` is `None` on a checkpoint-hit resume**,
   not a reconstructed live object. `VerificationReport` (MS4, frozen) has
   no `from_dict`/deserialization method, and this milestone does not add
   one to a frozen MS4 class. `Manifest.verification_passed` (this
   milestone's own, JSON-safe field) carries the same boolean gate outcome
   on both a fresh run and a resumed one — and is guaranteed `True` for any
   manifest that was actually written, since a failing gate halts `run()`
   before either is produced (see item 2 above). **Code:**
   `wfcrc/runner/runner.py` (`ExperimentRunner.run`'s `live_verification`
   holder). **Tests:**
   `tests/unit/runner/test_runner.py::TestResume::test_resume_skips_recomputation_of_experiment_stage`.
5. **Sweep cells always use a *derived* seed, never a swept raw seed
   directly** (`derive_seed(f"runner.sweep.cell.{index}", seed)`), per the
   MS5 spec's own acceptance criterion "sweep isolation (distinct dirs +
   derived seeds)" (C2 item 10). This means two cells that name the same
   raw seed in `SweepConfig.seeds` (e.g. to explore how sensitive results
   are to seed choice at several `alpha`s) still get distinct, independent
   splits — disclosed since a reader might otherwise expect the raw seed
   value to be used verbatim. **Tests:**
   `tests/unit/runner/test_runner.py::TestRunSweep::test_seeds_are_derived_and_distinct_even_when_raw_seed_repeats`.
6. **`RunnerError`** (new, purely additive `WFCRCError` subclass, matching
   every previous milestone's own pattern of adding one new exception for
   its own new failure domain — `PreconditionError`/`FamilyError` in MS2,
   `SetConstructionError`/`VerificationError` in MS3,
   `SplitLeakageError` in MS4): covers `ExperimentRunner`-specific failures
   with no more specific existing exception — `resume()` on a directory
   with no resumable run, and an invalid swept `alpha` caught defensively
   before it would otherwise reach `wfcrc.config`/`wfcrc.calibration` (see
   item 7 below).
7. **`run_sweep` validates `alpha` itself before constructing a cell's
   `CalibrationConfig`.** `wfcrc.config.schema.CalibrationConfig` is a
   plain dataclass with no `__post_init__` validation (range validation
   lives in `wfcrc.config.loader`'s parsing functions, not the dataclass
   itself, and `dataclasses.replace` bypasses the loader entirely) — so an
   out-of-range swept `alpha` would otherwise silently produce a malformed
   config rather than a clear error. `run_sweep` checks `0 < alpha < B`
   itself before building the cell (raising `RunnerError`, caught by the
   sweep's own "record + continue" handling, same as any other cell
   failure) — a self-contained defensive addition, not a modification of
   the frozen `CalibrationConfig`/`loader.py`.
8. **`wfcrc.visualization`'s figure-input shapes are a disclosed,
   reasonable realization, not a frozen data schema.** The MS5
   Implementation Spec names `FigureSpec`/`FigureFile` as data structures
   (C1 item 6) but gives no schema for what data each of the F1-F8 plotting
   functions actually takes — only the Experiment Blueprint's prose
   description of each figure's content (§26). Each `plot_*` function's
   exact parameters (e.g. `plot_risk_vs_alpha(alphas, risks_by_family,
   out_path, *, cis_by_family=None, ...)`) were chosen to match that prose
   directly and are disclosed here, the same pattern already established
   for the FPR loss and statistical-test gap-fills (§3, §8 above) — no
   frozen mathematical content is affected (these are rendering functions,
   not part of the calibration procedure).
9. **`matplotlib>=3.8,<4`** is a new dependency (`pyproject.toml`),
   justified by the same "no new dependencies unless the frozen spec
   genuinely requires a capability numpy/stdlib can't provide" policy
   already used to justify avoiding scipy in MS2's KL solver: the MS5 spec
   explicitly requires vector figure files (`.pdf`/`.svg`) with axes,
   legends, and CI bands (C1 items 2-4), which stdlib/numpy cannot produce.
   Figure output is made byte-deterministic across processes (fixed
   `savefig` metadata; a fixed `rcParams["svg.hashsalt"]`) — see
   `wfcrc/visualization/base.py`'s module docstring and
   `tests/unit/visualization/test_base.py::TestRenderFigure::test_byte_deterministic_across_calls`.
10. **`make reproduce`'s "reference experiment" is synthetic**
    (`scripts/reproduce.py`), for the same reason `ExperimentRunner.run`
    itself takes injected `LossTable`s: no real dataset/model exists in
    this environment. A small, fixed-seed synthetic calibration/test
    `LossTable` pair (same recipe every invocation, including a *fixed*
    `Config.runner.cache_dir` string so `Manifest.config_hash` itself is
    bit-for-bit reproducible, not just the numeric outputs) is calibrated
    via `ExperimentRunner`, and its manifest is diffed against
    `tests/fixtures/reproduce_golden.json` within a `1e-9` absolute
    tolerance. `python scripts/reproduce.py --write-golden` regenerates the
    golden file after a deliberate change to the reference experiment
    itself. **Tests:** `tests/unit/scripts/test_reproduce.py`.

## Connections

Vault (frozen, not modified by this file): `Paper 1 - CLAIMS TRACEABILITY
MATRIX.md`, `Paper 1 - ALGORITHM SPECIFICATION.md` (§7, §15, §20),
`Paper 1 - Mathematical Specification (WF-CRC).md` (§5, §6), `Paper 1 -
IMPLEMENTATION BLUEPRINT.md` (§6, §9, §11, §12, §17), `Paper 1 - MS2/MS3/
MS4/MS5 IMPLEMENTATION SPEC.md`, `Paper 1 - EXPERIMENT BLUEPRINT.md` (§3,
§9, §12, §23, §26). Repository: `CHANGELOG.md`, `PROJECT_CONTEXT.md` §7.
