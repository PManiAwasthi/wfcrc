# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows the milestone sequence defined by the frozen
Implementation Blueprint (MS1–MS5) rather than conventional semantic
versioning while pre-1.0.

## [0.4.0] — 2026-07-16 — MS4: datasets contracts, metrics & experiment execution

### Added

- `wfcrc.datasets`: `Dataset`/`DatasetLoader` abstract contracts (M3),
  `ScoreProvider` abstract contract (M4), and a concrete
  `LossTableBuilder` (M7) that assembles a `LossTable` from them using the
  already-frozen `PredictionSetConstructor`/`LossEvaluator` — plus the A1
  hygiene gate (`assert_split_disjoint`, `SplitManifest`, raising the new
  `SplitLeakageError`). No concrete dataset/model/score-provider for any
  specific named dataset is implemented (none of the Experiment
  Blueprint's real datasets are available in this environment); tested
  exclusively against small synthetic test doubles.
- `wfcrc.evaluation.metrics` (`MetricSuite`, M13): `realized_worst_case_risk`
  (dual families only), `realized_marginal_risk`, `per_group_risk`,
  `mean_set_size`, `coverage` (reuses frozen `MiscoverageLoss`),
  `effective_sizes` (Kish's `n_eff`), `duality_gap`, `bootstrap_ci`,
  `one_sided_risk_test`, `paired_wilcoxon`, `holm_correct` — plus `CI` /
  `TestResult` result types. The last three are the standard, dependency-free
  (no scipy) textbook realizations of statistical procedures the Experiment
  Blueprint names but does not give an exact formula for — disclosed the
  same way the FPR loss's gap-fill was disclosed (see
  `CLAIMS_TRACEABILITY.md` §7).
- `wfcrc.evaluation.experiment`: `run_experiment`/`ExperimentReport` — a
  reduced, dataset-free "experiment execution" entry point composing the
  already-frozen `run_calibration_pipeline` with `metrics` into a
  structured, JSON-serializable report (calibration + verification +
  metrics + a reproducibility config hash). Not the Implementation
  Blueprint's full `runner.ExperimentRunner` (MS5): no dataset/model
  loading, no plotting, no sweeps/checkpointing/resume.
- Test-only negative-control harnesses (`tests/unit/calibration/test_negative_controls.py`):
  pooled K-fold WF-CRC and total-`n` inflation WF-CRC, backfilling the
  vault's own MS3 exit gate (G-iv) that was never actually implemented —
  demonstrates empirically that both ablations realize measurably higher
  risk than the frozen single-split procedure at the same target `alpha`,
  confirming P3 (cross-fitting) and P4 (`n_B`, not `n`) are load-bearing
  choices. Not exposed as library API, per explicit scope decision.
- `wfcrc.exceptions`: added `SplitLeakageError` (dataset split hygiene).
- 81 new unit tests across `datasets`/`evaluation.metrics`/
  `evaluation.experiment`/the negative-control harnesses; repository total
  512 tests, 100% line and branch coverage maintained across all 43
  source modules.

### Documentation

- `CLAIMS_TRACEABILITY.md` records this milestone's scope resolutions
  (confirmed with the user before implementation): `datasets/` is
  ABC-contracts-only (no real dataset/model); "experiment execution" is
  the reduced `run_experiment` pipeline extension, not the full MS5
  runner; the negative-control ablations are test-only, not library code;
  "reporting utilities" means structured JSON/dict output, not the M14
  viz module's figure/CSV sidecars. Also records the standard-textbook
  provenance of `one_sided_risk_test`/`paired_wilcoxon`/`holm_correct`,
  and two implementation-level bugs caught during self-review before
  freeze (a floating-point zero-variance guard using exact equality
  instead of a tolerance; a test-tuning bug from reading a module
  constant instead of a threaded parameter).

### Not implemented (explicitly out of MS4 scope)

- No concrete `Dataset`/`ScoreProvider` for any real, named dataset
  (Cityscapes, MSD, etc.) — building one requires an actual dataset/model
  checkpoint, neither present in this environment.
- No `wfcrc.evaluation.metrics`-adjacent visualization: `viz.Plotter`
  (Implementation Blueprint's M14, figures F1–F8) is out of scope.
- No full `runner.ExperimentRunner` (M15): no sweeps, no checkpointing/
  resume, no `make reproduce` golden-diff harness, no dataset/model
  loading inside the pipeline itself.
- The pooled-K-fold/total-`n` negative controls are test-only, not a
  reusable `wfcrc` library API.

## [0.3.0] — 2026-07-16 — MS3: prediction sets & calibration pipeline

### Added

- `wfcrc.prediction_sets`: `PredictionSetConstructor` contract (`construct`,
  `name`, shared `assert_nested` contract check) + the two frozen concrete
  constructors named by the MS2 Implementation Spec (§C3) —
  `ThresholdSets` (LAC: `C_λ = {k : score_k ≥ 1−λ}`) and `MorphologicalSets`
  (dilation-margin `C_λ = dilate(M₀, ⌊λ⌋)`, dimension-agnostic, dependency-free
  iterated dilation) — plus a `SETS` registry. `MorphologicalSets`'
  `direction="erosion"` raises `SetConstructionError` immediately: no
  formula for it exists anywhere in the vault (see Not implemented, below).
- `wfcrc.evaluation`: `Verifier` (Implementation Blueprint §6, MS4
  Implementation Spec's M12) — `check_preconditions` (Algorithm Spec §20
  item 1: monotonicity + boundedness, checked directly on a `LossTable`)
  and `check_calibration` (§20 items 2-6: split disjointness/sizes, `B̃`
  finiteness and `L̃≤B̃`, `g` monotonicity and `λ̂`-minimality, correct
  inflation denominator per branch, reproducibility under a fixed seed) —
  plus `CheckResult`/`VerificationReport` (with `.merge()` and the strict
  `.assert_ok()` gate).
- `wfcrc.calibration.pipeline`: `run_calibration_pipeline` /
  `PipelineResult` — thin `LossTable → WFCRCCalibrator.calibrate` (→
  optional `Verifier`) orchestration for a pre-built loss table. Accepts
  any verifier satisfying the local `VerifierLike` structural protocol
  (not a runtime import of `wfcrc.evaluation`) so `wfcrc.calibration`'s
  dependency footprint stays unchanged and the module graph stays acyclic
  (`wfcrc.evaluation` depends on `wfcrc.calibration`, not the reverse).
- `wfcrc.exceptions`: added `SetConstructionError` (prediction-set
  construction failures with no frozen formula) and `VerificationError`
  (raised by `VerificationReport.assert_ok()`).
- 78 new unit tests across `prediction_sets`/`evaluation`/`calibration.pipeline`;
  repository total 431 tests, 100% line and branch coverage maintained
  across all 37 source modules.

### Documentation

- `CLAIMS_TRACEABILITY.md` records this milestone's scope resolutions
  (confirmed with the user before implementation, per `PROJECT_CONTEXT.md`
  §7's numbering caveat): `prediction_sets` filenames follow the vault's
  own concrete class split rather than a classification/segmentation/
  multilabel split; `Verifier` lives in `wfcrc.evaluation` (matching the
  vault's own `verify`-is-a-sibling-of-`calibration` module boundary and
  this repository's pre-existing empty-directory scaffold) rather than
  nested under `wfcrc.calibration`; no `validator.py`/`interfaces.py` were
  added (the `Verifier` already centralizes checks, and the ABCs already
  live at each subpackage root); and the documented erosion-direction gap
  in `MorphologicalSets`.

### Not implemented (explicitly out of MS3 scope)

- No `multilabel.py` / multilabel-specific prediction-set constructor: no
  vault document gives one a formula (the MS2 Implementation Spec names
  only `ThresholdSets`/`MorphologicalSets`; "multi-label pixels" appears
  only as a `ThresholdSets` edge case, not a constructor).
- `MorphologicalSets(direction="erosion")`: no concrete formula exists in
  the vault beyond the phrase "grow/shrink mask by a monotone structuring
  element"; the only literal readings either collapse to plain dilation
  (by the standard erosion/dilation duality) or violate the expected
  empty/full-set `λ_min`/`λ_max` boundary behavior. Raises
  `SetConstructionError` at construction time rather than guessing.
- `wfcrc.evaluation.metrics` (`MetricSuite`, Implementation Blueprint's M13):
  bootstrap confidence intervals, significance tests (Wilcoxon/Holm),
  realized-risk statistics — this is experimental/statistical validation,
  explicitly out of this milestone's scope. Likewise, Algorithm
  Specification §20's final checklist item (the "empirical validity smoke
  test... illustrative, not a proof") is intentionally not one of
  `Verifier`'s checks, for the same reason; it is already exercised as a
  synthetic-data test in `wfcrc.calibration`'s own MS2 test suite.
- No `datasets`, `models`, or experiment-runner code; `run_calibration_pipeline`
  operates only on an already-built `LossTable`.

## [0.2.0] — 2026-07-15 — MS2: core algorithm

### Added

- `wfcrc.losses`: `LossEvaluator` contract + `FNRLoss`, `FPRLoss`,
  `MiscoverageLoss`, `assert_monotone` contract check, `LOSSES` registry.
  Deliberately decoupled from any `PredictionSetConstructor` — operates on
  generic `(predicted_set, label)` boolean arrays.
- `wfcrc.ambiguity`: `AmbiguityFamily`/`DualAmbiguityFamily` contracts + the
  four frozen supported families — `CVaRFamily`, `KLFamily`,
  `FiniteGroupFamily`, `KnownWeightFamily` — plus a `FAMILIES` registry.
  `KLFamily`'s 1-D dual solve is a dependency-free bracket-and-golden-section
  minimizer (no new numerical library) built on `utils.numerics.logsumexp`.
- `wfcrc.calibration`: `LossTable` (the minimal calibration input contract),
  `Splitter` (the sole stochastic step, seeded via `utils.seeds.derive_seed`),
  `ThresholdSearch` (monotone binary search), and `WFCRCCalibrator` /
  `CalibrationResult` (the single integration point — dispatches to the
  dual, finite-group, or known-weight branch per Algorithm Spec §7/§7').
- `wfcrc.exceptions`: added `PreconditionError` and `FamilyError` (purely
  additive; MS1's own docstring anticipated exactly this extension).
- 201 new unit tests across `losses`/`ambiguity`/`calibration`; repository
  total 346 tests, 100% line and branch coverage maintained.
- `CLAIMS_TRACEABILITY.md`: repository-side traceability record
  (implementation facts the frozen, read-only vault matrix does not and
  should not contain — see that file for what it covers and why).

### Fixed

- **KL dual minimizer boundary bug** (found via an end-to-end smoke test
  during MS2 development, before first freeze): `_bracket_and_minimize`
  compared `h(eta_min)` against `h(1.0)` to short-circuit the boundary
  case. For `eta_min` orders of magnitude below `1.0`, convexity between
  two *distant* points does not imply monotonicity *between* them, so this
  could misreport an interior minimum as a boundary solution. Fixed by
  comparing `eta_min` against an *adjacent* point instead; regression-tested
  against brute-force grid search.
- **Fixed-η fallback (Algorithm Spec §15, F-4)** (post-freeze audit
  finding): `KLFamily.estimate_dual` previously raised `FamilyError`
  whenever dual estimation hit the `eta_min` boundary (a degenerate/
  near-constant A-block, including the `n_A=1` singleton case) and the
  resulting transform of `B` overflowed. This discarded a case the frozen
  spec explicitly provides a fallback for. `estimate_dual` now detects the
  boundary condition and substitutes a fixed, data-independent `eta`
  (`fallback_eta`, a new optional `KLFamily` constructor parameter,
  default `1.0`), recomputing `mu` via the same closed-form profile step.
  Valid by weak duality for any `eta > 0` (Math Spec §5); explicitly named
  as an exact special case by **CL-8** ("single-split (or fixed-η) with
  `n_B` inflation is exact"). `FamilyError` is still raised if the
  fallback's own transform is also non-finite (genuine, unfixable F-3).
  See `CLAIMS_TRACEABILITY.md` §1 for full detail and test references.

### Documentation

- `CLAIMS_TRACEABILITY.md` also records two intentional API signature
  differences from the frozen text (`assert_monotone`'s parameters;
  `WFCRCCalibrator.calibrate`'s parameter shape), both required because
  `sets`/`prediction_sets` is out of MS2 scope, and the FPR loss's
  documented gap-fill (no exact formula exists in the frozen sources).

### Not implemented (explicitly out of MS2 scope)

- No `sets`/`prediction_sets`, `datasets`, `models`, `evaluation`,
  `visualization`, or experiment runner code. No Wasserstein/
  Lévy-Prokhorov/optimal-transport family. No `verify`/§20-checklist
  module (checklist items are instead exercised via named unit tests).

## [0.1.0] — 2026-07-15 — MS1: core infrastructure

### Added

- `wfcrc.utils.io`: canonical content hashing (`content_hash`), atomic file
  writes, JSON/`.npz` array serialization, numpy-aware encoding.
- `wfcrc.utils.numerics`: numerically stable `logsumexp`,
  `weighted_logsumexp`, `clamp`, `safe_div`, `quantile` (float64 only).
- `wfcrc.utils.seeds`: deterministic RNG fanout (`set_global_seed`,
  `derive_seed`, `rng_for`) with no hidden global RNG mutation.
- `wfcrc.utils.logging`: structured, golden-diffable JSONL run logging.
- `wfcrc.utils.cache`: content-addressed, read-through cache built on
  `utils.io`.
- `wfcrc.utils.reproducibility`: git commit + environment fingerprint
  capture for future run manifests.
- `wfcrc.config`: typed, immutable, strictly validated, hashable, layered
  YAML configuration system (`load_config`, `Config.hash`/`.to_yaml`/`.get`).
- `wfcrc.exceptions`: structured exception hierarchy rooted at `WFCRCError`.
- Project infrastructure: `pyproject.toml` (Python 3.12+), Ruff, Black,
  MyPy (strict), Pytest + coverage, pre-commit, GitHub Actions CI, MkDocs
  documentation skeleton.
- `requirements/lock.txt`: fully pinned dependency closure for exact
  environment reproduction (`make install-locked` / `make lock`).
- 145 unit tests across `utils`/`config`, 100% line and branch coverage.

### Changed (post-audit cleanup, same milestone)

- `wfcrc.utils.cache.make_key` now uses the full, untruncated 256-bit
  SHA-256 digest (`CACHE_KEY_HASH_WIDTH`) instead of `content_hash`'s
  shorter default width, to keep cache-key collision probability
  negligible across long-running research sweeps.
- `wfcrc.utils.numerics.safe_div` now returns a `float64` scalar for
  scalar inputs (previously always returned a 0-d array), matching the
  scalar/array return convention already used by `logsumexp`, `clamp`,
  and `quantile`. No numerical behavior changed.
- Corrected the `Repository`/`repo_url` metadata in `pyproject.toml` and
  `mkdocs.yml` to the actual git remote.

### Documentation

- Added `docs/architecture.md`, `docs/configuration.md`,
  `docs/reproducibility.md`, `docs/getting-started.md`, and API reference
  pages.
- `docs/architecture.md` now documents the naming divergence between the
  frozen Implementation Blueprint's `src/wfcrc/{data,sets,losses,families,
  calibration,verify,metrics,viz,runner}` layout and this repository's
  actual (pre-existing, unmodified) flat layout and directory names.
- Added `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.

### Not implemented (explicitly out of MS1 scope)

- No mathematical logic: no ambiguity families, prediction sets, losses,
  calibration procedure, verification, metrics, visualization, or
  experiment runner. `make reproduce` exists only as a documented stub.
