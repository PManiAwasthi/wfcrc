# Dataset Split Policy — WFCRC Research Program

> **Status:** Proposed and adopted for first use (MS7, the first end-to-end
> pipeline), **not yet reviewed by a domain expert / PI**. This document is
> a scientific methodology artifact, not an engineering design note — see
> §0 below for exactly what "frozen" means here and what would revise it.
> **Version:** 1.0. **Date:** 2026-07-18. **Scope:** all Phase-A datasets
> named in `Paper 1 - DATASET SELECTION AUDIT.md` — Cityscapes, ACDC
> (driving), MSD Task04_Hippocampus, MSD Task07_Pancreas, CIFAR-10,
> CIFAR-10.1, Kvasir-SEG.

---

## 0 · What "frozen" means for this document

No dataset split proportion or split-generation methodology exists anywhere
in the frozen Research Vault — an explicit, exhaustive search of the
Experiment Blueprint, Algorithm Specification, MS2/MS4 Implementation
Specs, and the Experiment Environment Audit (performed during MS6.3A, and
re-confirmed while writing this document) found only the WFCRC-*internal*
A/B calibration-split ratio (`π ∈ {0.2, 0.3, 0.5}`, Experiment Blueprint
§18) — a completely different, already-frozen quantity from what this
document defines (see §2, "Two splits, not one," for the distinction).
**This document is therefore the first time a dataset-level train/
calibration/test partitioning methodology is defined for this research
program.** It is written now under an explicit instruction to "create and
freeze" it before any concrete model/ScoreProvider work proceeds, per that
instruction's own stated authority to do so, and is treated as binding
policy for the MS7 pipeline and beyond unless explicitly revised (§7). It
has **not**, however, been reviewed by a statistician, clinical
collaborator, or the datasets' own stewarding bodies — that review is
recommended before this document's numeric choices are cited in a
submitted manuscript's methodology section. Where a genuinely open
question exists rather than a defensible default, it is listed in §8, not
guessed at.

---

## 1 · Purpose

Conformal risk control's central statistical guarantee — that the deployed
threshold's realized risk is controlled at level `α` with the exchangeability-
based coverage argument the Mathematical Specification proves — holds only
if the calibration data is **exchangeable with the data risk is later
measured on**, and **independent of anything the base model saw during
training or model selection**. Both properties are assumptions the theorem
requires as hypotheses; neither is automatically true of an arbitrary
dataset partition, and violating either invalidates the guarantee silently
— the pipeline still runs, still returns a threshold, and the resulting
risk bound is simply wrong, with no runtime signal that anything went
wrong. This is qualitatively different from a numerical bug: a shape
mismatch crashes; a leaked split does not.

Two failure modes follow directly from improper partitioning:

- **Optimistic bias from train/calibration overlap.** If the base model
  saw an example during training (or during architecture/hyperparameter
  selection informed by held-out performance), its score on that example
  is systematically more confident than on a genuinely unseen example. A
  calibration set containing such examples produces a threshold that looks
  tighter (smaller prediction sets) than the true risk-controlling
  threshold — an invalid guarantee that fails silently in deployment.
- **Invalid coverage from calibration/test overlap, or non-exchangeability
  between calibration and test.** The single-split exact validity theorem
  (Algorithm Specification §17, Proof Obligation P3/P4) is a
  finite-sample, exchangeability-based guarantee about the relationship
  between the calibration sample and *future* exchangeable data. If the
  test set used to *evaluate* that guarantee is not disjoint from
  calibration, or is drawn from a systematically different distribution
  without that shift being the explicit object of study (as it is, by
  design, in E3/E4), the empirical check is not evaluating the theorem at
  all — it is circular in the first case, and comparing apples to oranges
  in the second.

This document exists so that every dataset's partitioning is decided once,
explicitly, with its reasoning on record — not re-derived ad hoc per
experiment, per engineer, or per pull request.

---

## 2 · General Principles

**Two splits, not one — kept explicitly distinct throughout this
document.**

1. **The dataset-level partition** (this document's subject): which raw
   examples are used to (a) train/select the base model, (b) calibrate the
   WFCRC threshold, (c) evaluate the realized guarantee. Frozen per-dataset
   in §3; recorded per-run in a `SplitManifest` (§4).
2. **The WFCRC-internal calibration-pool A/B split** (already frozen,
   Experiment Blueprint §18, Algorithm Specification §7/§17): once a
   calibration pool exists (per item 1(b) above), `wfcrc.calibration.splitter.Splitter`
   further divides *that pool* into blocks `A` (dual-parameter estimation)
   and `B` (threshold search), by fraction `π`. This is **not** re-decided
   by this document — it is orthogonal, already-frozen, and operates
   entirely inside whatever calibration pool this document's per-dataset
   policy hands it.

**Principles governing item 1 (frozen for every dataset in §3):**

- **Deterministic, reproducible split generation.** A partition is a pure
  function of (dataset identity, a documented generation method, a fixed
  random seed). Re-running the generation procedure against the same
  dataset snapshot with the same seed must reproduce the identical
  partition, byte for byte, forever — the same reproducibility bar the
  rest of this project already holds itself to (`PROJECT_CONTEXT.md` §9,
  "Deterministic behavior... the seeded A/B split is the *only* stochastic
  component anywhere in the codebase" — item 1's split generation is a
  second, dataset-preparation-time stochastic component, outside that
  codebase-runtime claim, but held to the identical determinism standard).
- **Fixed random seeds, recorded, never silently changed.** A split's seed
  is part of its identity (§4). Changing the seed changes the split;
  changing the split without a version bump and a recorded reason is
  exactly the "silently changing the methodology" this document exists to
  prevent (§7).
- **The correct experimental unit is whatever is exchangeable, not
  whatever is convenient.** For patient/subject-derived data (MSD
  Hippocampus, MSD Pancreas), the unit is the **patient** (here, one
  patient per case — see §3), because two scans of the same patient are
  not independent draws from the population WFCRC's guarantee is stated
  over. For scene/video-derived data (Kvasir-SEG's endoscopic frames), the
  correct unit is the **source procedure**, not the individual frame,
  whenever that grouping is recoverable (§3, §5). For single, independently
  photographed natural images (CIFAR-10, CIFAR-10.1) and single,
  independently captured street scenes across geographically distinct
  cities (Cityscapes, ACDC), the image itself is already the correct,
  independent unit.
- **Complete three-way separation: train ⟂ calibration ⟂ test, always.**
  No example may appear in more than one role. This is the dataset-level
  analogue of the already-frozen, already-implemented A1 hygiene gate
  (`wfcrc.datasets.base.assert_split_disjoint`, `SplitLeakageError`) — this
  document defines the *scientific* requirement; that gate is the
  *mechanical* enforcement of it (§4).
- **No data leakage between any experimental stage**, including the subtler
  forms in §5 (augmentation, benchmark contamination, cross-validation
  leakage) — not just the direct three-way overlap the mechanical gate
  already catches.
- **Version-controlled `SplitManifest` records.** Every concrete split used
  in any reported experiment is a named, versioned, retrievable artifact
  (§4, §6) — never an unrecorded, re-derived-on-the-fly random sample.

---

## 3 · Dataset-Specific Split Policy

Every subsection below distinguishes explicitly between **the official
dataset partition** (as published by the dataset's own stewards) and **the
WFCRC experimental split** (this document's own policy, built from
whichever part of the official partition is actually usable for WFCRC's
purposes). Two upstream partitioning realities recur across multiple
datasets and are stated once here rather than seven times: (a) several
benchmarks (Cityscapes, ACDC) withhold ground-truth labels for their
official "test" partition entirely — it exists only for a leaderboard
server, not for local use — so it **cannot** serve as WFCRC calibration or
test data (no labels means no measurable loss); (b) the Medical
Segmentation Decathlon's official "test" partition (both MSD tasks below)
is the same kind of unlabelled, leaderboard-only pool, already documented
in this codebase's own MSD loader (`wfcrc/datasets/loaders/msd.py` module
docstring §3). In both cases, WFCRC's own train/calibration/test partition
is built entirely from the officially-*labelled* pool, re-divided; the
unlabelled official "test" pool is excluded from every WFCRC role, not
merely from calibration/test.

### 3.1 · MSD Task04_Hippocampus

- **Official partition.** 260 labelled cases (`dataset.json`'s
  `"training"` list) + 130 unlabelled challenge-test cases (`"test"` list,
  no `labelsTs`) — real-data-verified counts, MS6.3A validation pass.
- **Experimental unit.** Patient (one 3-D volume = one patient; no
  multi-scan-per-patient structure in this task, so case-level splitting
  *is* patient-level splitting here — verified against the loader's own
  discovery, which yields one id per `dataset.json` training entry).
- **Official partition reused or extended?** The official train/test
  boundary is **not** reused directly (the "test" half has no labels and
  is useless for WFCRC purposes, per the general note above); WFCRC
  extends the official 260-case *labelled* pool into its own three-way
  split.
- **WFCRC split.** `60% / 20% / 20%` (train / calibration / test) of the
  260-case labelled pool, patient-level, deterministically seeded (§4,
  §6) — approximately 156 / 52 / 52 cases. **Justification:** this is the
  conventional ML train/validation/test ratio family, adapted here to
  train/calibration/test; it is not dataset-specific tuning, and is chosen
  because (a) MSD Hippocampus's whole purpose in the Dataset Selection
  Audit is "cheap, high-n medical validity + fast statistics" (§4 of that
  audit) — a calibration pool of ~52 cases plus the internal `π`-split
  (§2) still leaves both A/B blocks non-trivial (`n_A, n_B` on the order
  of 10-40 depending on `π`), and (b) reserving the *majority* of cases for
  base-model training reflects that Hippocampus's segmentation task itself
  (not the conformal layer) is the harder-to-estimate quantity here. No
  literature-specific citation determines this exact ratio; it is a
  disclosed, conventional default, not a claim of statistical optimality —
  see §8.
- **Calibration data strategy.** The 20% calibration allocation is handed
  to WFCRC as one pool; `wfcrc.calibration.splitter.Splitter` performs its
  own already-frozen internal `π`-fraction A/B split on top of it (§2) —
  this document does not re-decide that internal split.
  - **Reduced-scale exception (MS7 smoke pipeline only).** For the MS7
    end-to-end framework-validation smoke run (a "handful of cases,"
    explicitly not a statistically meaningful experiment — see the MS7
    task brief's own Task 6), a tiny illustrative subset of the labelled
    pool is used instead of the full 60/20/20 partition, generated by the
    same deterministic mechanism and disjointness guarantee, but sized for
    fast iteration rather than statistical power. This exception is
    recorded here explicitly so it is never mistaken for the frozen
    research-scale policy above.
- **Evaluation strategy.** Realized risk / effective set size / (for dual
  families) realized worst-case risk and duality gap are measured on the
  20% test pool only, never on calibration or training cases.
- **Expected `SplitManifest` inputs.** Patient/case ids exactly as
  `MSDNiftiLoader` discovers them (`hippocampus_XXX`, derived from the
  original filename stem — `wfcrc/datasets/loaders/msd.py` §"stable
  example ids"); three disjoint id lists keyed `"train"`/`"calibration"`/
  `"test"`.
- **Potential leakage risks.** None beyond the general three-way
  separation, given one volume = one patient. The unlabelled official
  "test" pool must never be referenced by any `SplitManifest` id (already
  mechanically impossible — `MSDNiftiLoader` never discovers those ids at
  all, per its own docstring §3).
- **Assumptions that must remain frozen.** The 260-case labelled pool
  itself (the specific archive acquired and validated in MS6.3A); the
  60/20/20 ratio (until revised per §7); patient-level = case-level for
  this specific task.

### 3.2 · MSD Task07_Pancreas

- **Official partition.** 281 labelled cases + an unlabelled challenge-test
  pool (exact count not yet independently re-verified against a real
  archive the way Hippocampus's was in MS6.3A — Pancreas real-data
  acquisition remains future work, per the MS6.3A task brief's own
  explicit exclusion).
- **Experimental unit.** Patient (same reasoning as §3.1; MSD tasks are
  one-volume-per-patient throughout).
- **Official partition reused or extended?** Same pattern as §3.1: the
  official unlabelled "test" pool is excluded entirely; WFCRC re-divides
  the labelled pool.
- **WFCRC split.** `60% / 20% / 20%` of the labelled pool, patient-level,
  deterministically seeded — the same ratio as §3.1, for
  **methodological consistency across the two MSD tasks** (the Dataset
  Selection Audit explicitly pairs them as "cheap-statistics vs
  hard-conditional" duals within one modality; using a different ratio
  for each without a specific reason would itself be an unjustified,
  undocumented choice). Given Pancreas's severe foreground/background
  class imbalance (Dataset Selection Audit §4, "✓✓ severe imbalance"), the
  calibration and test pools should additionally be checked for
  non-degenerate per-class representation once real Pancreas data is
  acquired — flagged here, not resolved (§8).
- **Calibration data strategy / evaluation strategy.** Identical structure
  to §3.1.
- **Expected `SplitManifest` inputs.** Same shape as §3.1, once a
  `MSDNiftiLoader(task="Task07_Pancreas", ...)` instantiation is
  authorized and real Pancreas data acquired (both explicitly out of scope
  through MS6.3A and MS7).
- **Potential leakage risks.** Same as §3.1.
- **Assumptions that must remain frozen.** The 60/20/20 ratio (shared with
  §3.1, §7); patient-level = case-level (to be re-verified once real
  Pancreas data is on hand, per the "verify against the actual dataset
  before assuming" discipline MS6.3A already established for Hippocampus).

### 3.3 · Cityscapes

- **Official partition.** `train` (2,975 finely-annotated images), `val`
  (500 images), `test` (1,525 images, **ground truth withheld** — held by
  the Cityscapes benchmark server for leaderboard scoring only, never
  distributed). The official train/val boundary is itself split **by
  city** — train and val contain images from entirely disjoint sets of
  cities, not merely disjoint images — a stronger separation guarantee
  than ordinary image-level splitting.
- **Experimental unit.** Image (frame). The official train/val boundary's
  own city-level separation is a *stronger* property than this document
  requires generally and is preserved, not re-derived, wherever the
  official train/val boundary itself is reused (see below).
- **Official partition reused or extended?** **Reused for the train
  boundary; extended for calibration/test.** Official `train` (2,975
  images, disjoint cities from val) is the base-model training pool,
  unchanged. The official `test` split is unusable (no local labels, per
  the general note above). The official `val` split (500 labelled images)
  is therefore the **entire** labelled pool available for anything
  requiring measured loss — it must itself be re-divided into WFCRC
  calibration and WFCRC test.
- **WFCRC split.** Train: official `train`, unchanged (2,975 images). The
  500-image official `val` pool is split `50% / 50%` (calibration / test),
  image-level, deterministically seeded — approximately 250 / 250 images.
  **Justification:** 500 labelled images is a comparatively small pool to
  begin with (versus e.g. Hippocampus's 260 *patients*, each contributing
  one full annotated volume); an even split maximizes statistical power on
  both sides rather than favoring one, and is standard practice when
  reusing a benchmark's small official val set for a downstream calibration
  task it was not originally designed for.
- **Calibration data strategy.** The 250-image calibration half feeds
  WFCRC's internal A/B split (§2) exactly as in §3.1.
- **Evaluation strategy.** Realized metrics measured on the 250-image test
  half only. Per-region/per-class conditional risk (E2) uses row-index
  group masks built from labels within this same test half (Group Mask
  Builder, MS6.7 — not yet built; out of scope here) — never a
  reshuffled or re-derived subset.
- **Expected `SplitManifest` inputs.** Cityscapes image ids (its own
  stable `<city>_<sequence>_<frame>` naming convention); three disjoint id
  lists, with `"train"` fixed to the entire official `train` set by
  construction.
- **Potential leakage risks.** None from patient/subject identity (not
  applicable to street scenes); the only risk is failing to preserve the
  official train/val city-disjointness when re-deriving calibration/test
  from `val` — mitigated by construction, since calibration/test are both
  carved *from* `val` only, never mixed with `train`.
- **Assumptions that must remain frozen.** The 50/50 calibration/test
  split of `val` (until revised, §7); the official train/val city boundary
  itself (never touched).

### 3.4 · ACDC (driving)

- **Official partition.** Follows the same Cityscapes-benchmark-server
  convention (train/val/test, test labels withheld for leaderboard
  scoring) — exact published counts not yet independently re-verified
  against a real local archive (real ACDC acquisition remains future
  work, unlike Hippocampus).
- **Experimental unit.** Image (frame), same reasoning as Cityscapes.
- **Official partition reused or extended?** **Neither in the training
  role — ACDC contributes zero training data.** Per the Dataset Selection
  Audit ("reuses the Cityscapes model unchanged... zero extra training"),
  ACDC's entire role in WFCRC is post-hoc evaluation of a Cityscapes-
  trained model under real distribution shift (E4). Its official train
  split (if used at all) would only be relevant to someone *fine-tuning*
  on ACDC, which this research program does not do.
- **WFCRC split.** The **entire usable (labelled) ACDC pool** — official
  `train` + official `val`, combined, since neither ever trains anything
  here — is re-divided `50% / 50%` (calibration / test), image-level,
  deterministically seeded, mirroring §3.3's reasoning for a small reused
  benchmark pool. **Justification:** because no ACDC data trains the
  model, there is no "official train" role to preserve the way Cityscapes'
  is preserved in §3.3 — every labelled ACDC image is equally eligible for
  calibration or test, so pooling before splitting (rather than
  preserving whatever arbitrary train/val boundary ACDC's own
  documentation used) is the more defensible choice, not less.
- **Calibration data strategy / evaluation strategy.** Same structure as
  §3.3, on the pooled-then-split ACDC data.
- **Expected `SplitManifest` inputs.** ACDC image ids (its own naming
  convention, TBD until a concrete `CityscapesFormatLoader`-family ACDC
  entry is built — out of scope for MS6.3/MS7); three disjoint id lists
  with `"train"` empty by construction (no ACDC image ever trains the
  model).
- **Potential leakage risks.** The R2 name-collision risk already flagged
  in the Dataset Selection Audit (driving ACDC vs. cardiac-MRI ACDC) is a
  data-*acquisition* risk, not a splitting risk, and is out of this
  document's scope (see that audit §7). No patient/subject identity
  concern (street scenes).
- **Assumptions that must remain frozen.** Pooling official train+val
  before the 50/50 calibration/test split (a deliberate departure from
  §3.3's "preserve the official boundary" policy, justified above by
  ACDC's different role); the 50/50 ratio itself (§7).

### 3.5 · CIFAR-10

- **Official partition.** 50,000 train + 10,000 test images. Unlike
  Cityscapes/ACDC/MSD, the official CIFAR-10 **test set's labels are
  publicly distributed** — it is not a withheld-label leaderboard split.
- **Experimental unit.** Image. No patient/scene/procedure structure
  applies; CIFAR-10 images are independently sourced natural-image crops.
- **Official partition reused or extended?** **Train boundary reused
  unchanged; the official test set is re-divided** (not reused as a single
  WFCRC "test" role) — because WFCRC additionally needs a calibration
  pool, and CIFAR-10 provides no separate official pool for that purpose
  beyond train/test.
- **WFCRC split.** Train: official 50,000-image train set, unchanged. The
  official 10,000-image test set is split `50% / 50%` (calibration /
  test), image-level, deterministically seeded, class-stratified (CIFAR-10
  is by design exactly class-balanced, 1,000 images/class in the official
  test set; the split should preserve that balance in both halves rather
  than risk an accidental class-imbalanced calibration or test pool).
  **Justification:** 10,000 images is large enough that a straight 50/50
  split still leaves 5,000 examples per side — ample for both the internal
  `π`-split and stable quantile/threshold estimates — so there is no
  strong reason to favor test over calibration or vice versa, unlike the
  small-pool cases in §3.3/§3.4.
- **Calibration data strategy / evaluation strategy.** Standard structure,
  as above.
- **Expected `SplitManifest` inputs.** CIFAR-10's own integer/batch-index
  ids (from the official binary batch files); three disjoint id lists.
- **Potential leakage risks.** CIFAR-10's own well-documented near-duplicate
  issue with the Tiny Images source pool (some images are extremely similar
  or identical crops) is a *within-official-test* leakage concern
  documented in the literature (relevant specifically to the CIFAR-10.1
  comparison, §3.6) — noted here, not resolved (§8): this document does
  not currently require near-duplicate detection before splitting the
  official test set, since CIFAR-10.1 was constructed by its own authors
  specifically to test generalization past that concern.
- **Assumptions that must remain frozen.** The 50/50, class-stratified
  split of the official test set (§7); the official train set used
  unchanged.

### 3.6 · CIFAR-10.1

- **Official partition.** No train/val/test substructure at all — a single
  pool (2,021 images in the original "v4" variant; 2,000 images,
  class-balanced, in the "v6" variant this project's own `DATASET_METADATA`
  already records as `recommended_variant`). It exists purely as an
  independent, naturally-shifted evaluation set (Recht et al., "Do CIFAR-10
  Classifiers Generalize to CIFAR-10.1?").
- **Experimental unit.** Image.
- **Official partition reused or extended?** Not applicable — there is no
  official partition to reuse or extend.
- **WFCRC split.** **100% test, 0% train, 0% calibration.** The entire v6
  pool (2,000 images) is reserved for evaluation only.
  **Justification:** CIFAR-10.1's entire scientific purpose (both in the
  original paper and in this project's own E4 role, "real classification
  shift") is to measure generalization to *unseen*, naturally-shifted data
  using a model and calibration threshold established entirely on
  CIFAR-10. Using any CIFAR-10.1 data for training or calibration would
  directly undermine the one property that makes it useful — this is the
  one dataset in this document where the "correct" split is not a ratio
  question at all, but a categorical one.
- **Calibration data strategy.** None — CIFAR-10.1 contributes no
  calibration data. The calibration threshold `λ̂` evaluated against
  CIFAR-10.1 is the one already selected from CIFAR-10's own calibration
  pool (§3.5); E4 measures how that threshold's realized risk behaves
  under distribution shift, which is only a meaningful question if the
  threshold itself was never touched by CIFAR-10.1 data.
- **Evaluation strategy.** Realized risk / effective set size measured on
  the full 2,000-image pool.
- **Expected `SplitManifest` inputs.** CIFAR-10.1's own image ids; a
  `SplitManifest` with `"train"` and `"calibration"` both empty by
  construction, `"test"` containing every id.
- **Potential leakage risks.** The Recht et al. authors' own documented
  construction methodology already addresses near-duplicate overlap with
  CIFAR-10 at the source; this document does not re-derive that check, but
  flags reliance on it explicitly (§8).
- **Assumptions that must remain frozen.** The "test-only, zero train/
  calibration" categorical policy (this is the one dataset-specific
  decision in this document least likely to ever need revision, since it
  follows directly from CIFAR-10.1's stated purpose rather than from a
  disclosed-but-arbitrary ratio choice).

### 3.7 · Kvasir-SEG

- **Official partition.** **None.** Kvasir-SEG is distributed as a single,
  undivided pool of 1,000 annotated polyp images; unlike Cityscapes/MSD/
  CIFAR-10, there is no train/val/test boundary defined by the dataset's
  own stewards (Simula Research Laboratory / Cancer Registry of Norway) at
  the time of the `DATASET_METADATA` verification (MS6.2).
- **Experimental unit.** **Source endoscopic procedure, not individual
  frame — with an explicitly unresolved caveat.** Endoscopic-image
  datasets frequently contain multiple, highly-correlated frames drawn
  from the same procedure/video; splitting at the frame level risks
  placing near-duplicate frames from one procedure into different roles
  (train/calibration/test), which is a leakage failure mode by the same
  logic as patient-level MSD splitting (§2). **This document could not
  confirm, from the metadata independently verified in MS6.2, whether
  Kvasir-SEG's public release exposes per-image procedure/patient
  identifiers** (the official site's license/documentation was not
  extractable at that time — see `DATASET_METADATA["kvasir_seg"]`'s own
  recorded caveat). This is listed as an open methodological decision
  (§8), not resolved by assumption here.
- **Official partition reused or extended?** Not applicable (none exists).
- **WFCRC split.** **Not frozen — see §8.** Pending confirmation of
  whether procedure-level grouping is recoverable, this document
  deliberately does not commit to a specific ratio or unit for Kvasir-SEG.
  A structurally-consistent placeholder (mirroring §3.3/§3.4's small-pool
  reasoning: something in the neighborhood of a majority-train,
  even-remainder-split-for-calibration/test shape) is *not* proposed here,
  because doing so before the procedure-grouping question is answered
  risks exactly the silently-invalid-guarantee failure mode §1 describes,
  if frame-level splitting is later found to leak.
- **Calibration data strategy / evaluation strategy.** Deferred (§8).
- **Expected `SplitManifest` inputs.** Kvasir-SEG's own image filenames as
  ids; format otherwise identical to the other datasets once the split
  unit is decided.
- **Potential leakage risks.** Procedure-level leakage (see above) is the
  dominant, currently-unresolved risk. Kvasir-SEG's role in the Dataset
  Selection Audit (E12, "clinical FP control") also involves a "~1 hr
  fine-tune" step (§6 of that audit) — meaning, unlike ACDC, some
  Kvasir-SEG data **does** train/fine-tune the base model, so the
  train/calibration/test three-way separation applies here in full, not
  the "zero train" pattern of §3.4/§3.6.
- **Assumptions that must remain frozen.** None yet — this subsection is
  intentionally incomplete pending §8.

---

## 4 · SplitManifest Scientific Contract

This section defines what information a `SplitManifest` scientific record
must contain to serve as a citable, reproducible artifact — the
methodological requirement, not any particular file format or class
(`wfcrc.datasets.base.SplitManifest`, the current frozen engineering
implementation, mechanically enforces disjointness over three id lists but
does **not** yet carry most of the fields below — see §8, item on
extending it, which this document deliberately leaves as a future
engineering task rather than specifying an implementation here).

A scientifically complete split record must contain:

- **Dataset identifier.** Which dataset (and, where applicable, which
  named subset/variant — e.g. CIFAR-10.1 "v6" specifically, not "v4") this
  split partitions.
- **Split version.** A monotonically increasing identifier (e.g. `v1`,
  `v2`) distinguishing this split from any prior split of the same
  dataset — never silently overwritten (§7).
- **Sample identifiers.** The complete, explicit list of every example id
  assigned to every role — never a formula or a range that could evaluate
  differently later ("the first 60%" is not an identifier list; the
  actual 156 patient ids is).
- **Split assignments.** Which role (train / calibration / test) each
  listed id belongs to, with no id appearing under more than one role.
- **Generation method.** A description precise enough that the split
  could, in principle, be regenerated from scratch and verified identical
  — e.g. "stratified random sample of the official test set, by class,
  using generator G seeded with S," not merely "randomly split."
- **Random seed.** The exact seed value used, if the generation method is
  stochastic (every method in §3 is, except CIFAR-10.1's categorical
  100%-test policy and Cityscapes' train-boundary reuse, which are
  deterministic by construction and need no seed).
- **Creation timestamp.** When this specific split was generated —
  distinguishing it from a later regeneration attempt with the same
  nominal method and seed, in case the underlying dataset snapshot itself
  ever changes (e.g. a corrected re-release of a dataset).
- **Provenance information.** Which document (this one, a specific
  version of it) and which concrete tool/script/commit produced this
  split — the chain of custody from "the policy" (this document) to "the
  specific list of ids used in a specific reported experiment."

---

## 5 · Leakage Prevention

| Scenario | Description | WFCRC methodology's mitigation |
|---|---|---|
| **Patient leakage** | Multiple examples from the same patient/subject span more than one split role. | Patient-level splitting unit for MSD Hippocampus/Pancreas (§3.1, §3.2); mechanically moot here since one volume = one patient, but the *principle* is what generalizes to any future patient-derived dataset (Paper 2/3, §7). |
| **Duplicate image leakage** | The identical (or near-identical) image appears more than once, possibly across split roles, without being recognized as the same example. | Not currently checked programmatically for any Phase-A dataset; CIFAR-10's own known near-duplicate issue is flagged (§3.5) rather than silently assumed absent. Listed as an open item, §8. |
| **Augmentation leakage** | An augmented (rotated/cropped/jittered) version of a training example ends up in calibration/test, inflating apparent calibration confidence. | No data augmentation is used anywhere in this research program's frozen scope (base-model training itself is explicitly out of this repository's scope per Q1, §8.1 of `MS6_ARCHITECTURE_SPEC.md` — training happens externally, "prefer a public pretrained checkpoint"); if/when augmentation is introduced by whatever external training process produces a checkpoint, this document's train/calibration/test id lists remain the authority on which *raw* examples may not cross roles, and augmented views of a training example must never be scored as if they were calibration/test data. |
| **Calibration leakage** | Calibration data (or its labels) is visible to the model during training or architecture/hyperparameter selection. | Enforced by construction: every split in §3 assigns each example to exactly one role, and Checkpoint Management's provenance check (`MS6_ARCHITECTURE_SPEC.md` §3.5, R-CKPT1 — "a public pretrained checkpoint whose training split overlaps the calibration pool would leak") is the mechanical backstop for externally-sourced checkpoints whose training data this project does not directly control. |
| **Cross-validation leakage** | Reusing the same calibration/test split repeatedly across many analyses (e.g. hyperparameter sweeps over `α`/family/grid) inflates the *reported* validity of whichever configuration happens to look best, even though each individual run's guarantee is still exactly valid. | E6-E11's own design (Dataset Selection Audit §2, "re-analyses of the same cached logits under different families/splits/grids") reuses one fixed calibration/test split across many configurations by design — this is scientifically legitimate for reporting the sensitivity/ablation results those experiments target, but it means any single configuration's apparent superiority across that sweep should not be reported as an independently-validated result without a further held-out check; noted here as a genuine methodological subtlety of the multi-experiment design, not a defect. |
| **Benchmark contamination** | A model's pretraining corpus (if it uses a large pretrained backbone) already contained calibration/test images or their labels, e.g. via web-scale pretraining data that happens to include benchmark images. | Not resolved by this document — depends on the specific pretrained checkpoint's own training data, which is exactly what Checkpoint Management's provenance record (§3.5 of `MS6_ARCHITECTURE_SPEC.md`) is for. Flagged for explicit checkpoint-provenance review whenever a checkpoint is sourced externally rather than trained from scratch inside this project's controlled pipeline. |

---

## 6 · Reproducibility

- **Random seed policy.** Every stochastic split-generation step (§3)
  uses a project-level base seed, from which a component-specific seed is
  derived via the same deterministic-fanout mechanism the rest of this
  codebase already uses for its own internal RNG needs
  (`wfcrc.utils.seeds.derive_seed`) — not a bare, ad hoc call to a random
  number generator. The exact derivation name/base-seed pair used for each
  dataset's split is recorded in that split's `SplitManifest` (§4).
- **Deterministic split generation.** Given the same dataset snapshot,
  method, and seed, split generation must be exactly reproducible — no
  reliance on filesystem iteration order, dictionary ordering before
  Python 3.7's guarantee, or any other non-deterministic enumeration.
- **Versioning of `SplitManifest` files.** Every split is a versioned,
  named artifact (§4); superseding a split means creating a new version,
  never editing an existing one in place.
- **Storage location.** `SplitManifest` records for every dataset used in
  a reported experiment are stored under version control (or an
  equivalently durable, citable artifact store) alongside the experiment
  configuration that consumed them — never generated transiently at
  experiment-run time with no persisted record of exactly which ids were
  used.
- **Reproducibility requirement.** Any reported result must be
  traceable, via its `SplitManifest` record, back to the exact set of
  example ids used for training, calibration, and test — sufficient for
  an independent party to verify the three-way separation held, without
  needing to trust that it did.

---

## 7 · Future Research Program

Papers 2 and 3 (the geometric/region-based and ontological/label-space
extensions named in `PROJECT_CONTEXT.md` §2) should **reuse this policy's
general principles (§2) and, where the same datasets recur, its
per-dataset splits (§3) unchanged** — a region-based or label-space
extension of WFCRC does not, on its own, change which raw examples are
exchangeable with which, so there is no default reason to re-split a
dataset just because a later paper adds a new ambiguity-family dimension
on top of the same underlying data.

Any future modification to this policy — a changed ratio, a newly-resolved
open question from §8, a dataset added outside Phase-A's current seven —
must be recorded as a new, explicitly versioned revision of this document
(a new §-numbered decision with a stated reason and date), never as a
silent edit to an existing section's numbers. A revision changes which
`SplitManifest` version (§4, §6) is current for that dataset; it does not
retroactively invalidate the record of which split a *previously reported*
result used.

---

## 8 · Open Methodological Decisions

Listed explicitly, per this document's own instruction not to guess at
what is not yet decided:

1. **Kvasir-SEG's split unit and ratio (§3.7) — fully open.** Whether
   per-image procedure/patient grouping is recoverable from the public
   Kvasir-SEG release is unknown from what MS6.2's metadata verification
   could independently confirm. Resolving this requires either (a)
   contacting the dataset stewards / re-examining the release for
   procedure metadata, or (b) an explicit, disclosed decision to accept
   frame-level splitting with the documented residual leakage risk. No
   Kvasir-SEG loader work should proceed past discovery/pairing (the
   MS6.3-equivalent step) until this is resolved, since the split
   ultimately gates what any Kvasir-SEG `SplitManifest` can look like.
2. **Duplicate/near-duplicate detection before splitting — not currently
   performed for any dataset.** CIFAR-10's known Tiny-Images-derived
   near-duplicate issue (§3.5) and Kvasir-SEG's potential procedure-level
   frame correlation (§3.7, item 1) are both instances of a more general
   open question: should this project run an explicit near-duplicate
   detection pass before finalizing any split, rather than relying on
   each dataset's own upstream construction? Currently: no. Flagged for a
   future decision, not silently assumed safe.
3. **The `SplitManifest` engineering format does not yet carry §4's full
   scientific contract.** The current, frozen, mechanically-enforced
   `wfcrc.datasets.base.SplitManifest` (train_ids/cal_ids/test_ids +
   disjointness) and the per-loader `split_manifest` JSON input format
   (e.g. `MSDNiftiLoader`'s, MS6.3A) are both intentionally minimal —
   mechanical consumption only, no split-version/seed/timestamp/
   provenance fields. Extending either to carry §4's full record is a
   future, purely additive engineering task (explicitly not performed by
   this document, per its own "must remain independent of the
   implementation" instruction, and explicitly not performed by MS7's
   pipeline work either, per that milestone's own "do not modify frozen
   components" instruction) — not a methodological gap, but an
   implementation gap this document's existence now makes visible.
4. **MSD Task07_Pancreas's exact official case counts and per-class
   balance within its own labelled pool — not yet independently verified
   against a real archive.** §3.2's 60/20/20 split is written against the
   Dataset Selection Audit's informal "281" figure; MS6.3A's own
   experience with Hippocampus (where the real count, 260/130, differed
   from an initially-assumed 263/131 sourced from secondary references)
   is a direct precedent for treating any pre-acquisition Pancreas number
   here as provisional until Pancreas real-data acquisition happens.
5. **Cityscapes/ACDC exact official split counts used in §3.3/§3.4 are
   standard, well-documented benchmark figures but have not been
   re-verified against a locally-acquired archive the way MSD Hippocampus
   was in MS6.3A** (Cityscapes/ACDC acquisition remains future work, per
   the Dataset Selection Audit's own registration-latency risk, R3). If a
   locally-acquired archive's real counts differ, this section's numbers
   are provisional pending that verification, by the same discipline
   item 4 states for Pancreas.
6. **Whether a domain expert / PI should review §3's numeric ratios before
   they are cited in a submitted manuscript.** Recommended (§0), not yet
   done. This is listed as an open decision rather than resolved silently
   in either direction.

---

## Connections

`MS6_ARCHITECTURE_SPEC.md` · `wfcrc/datasets/loaders/msd.py` ·
`wfcrc/datasets/base.py` (`SplitManifest`, `assert_split_disjoint`) ·
`PROJECT_CONTEXT.md` · Research Vault: `Paper 1 - EXPERIMENT BLUEPRINT.md`,
`Paper 1 - DATASET SELECTION AUDIT.md`, `Paper 1 - ALGORITHM
SPECIFICATION.md`
