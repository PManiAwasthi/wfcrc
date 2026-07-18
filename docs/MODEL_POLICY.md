# Model Policy — WFCRC Research Program (Paper 1)

> **Status:** Frozen for first use (MS8), **not yet reviewed by a domain
> expert / PI**. **Version:** 1.0. **Date:** 2026-07-18. **Scope:** every
> base model / `ScoreProvider` consumed by E1–E12 against the frozen
> Phase-A dataset suite. This document defines **policy only** — it does
> not train, benchmark, or select a specific architecture's hyperparameters;
> those are engineering acts governed by `MS6_ARCHITECTURE_SPEC.md` §3.4/
> §3.5 (Score Providers, Checkpoint Management) and, for Hippocampus
> specifically, already executed once at MS7 (`wfcrc/models/scores/
> hippocampus_segmenter.py`).

---

## 1 · Accepted Baseline Models

Two distinct model roles exist in this research program and must not be
conflated:

### 1.1 Base models (produce scores the WFCRC layer calibrates over)

Per dataset, per the frozen Dataset Selection Audit (§2, §5) and
`MS6_ARCHITECTURE_SPEC.md` §3.4:

| Dataset(s) | Base model role | Checkpoint strategy | Status at MS8 |
|---|---|---|---|
| Cityscapes (+ ACDC, Cityscapes-C — reused unchanged) | Semantic segmentation, 19 classes | Prefer a public pretrained checkpoint (Dataset Selection Audit §6); train only if none is found suitable | Not yet acquired |
| MSD Task04_Hippocampus | 3-D medical segmentation | A compact 3-D U-Net (`_TinyUNet3D`, ~88K params, 2 pooling levels), inference-only against a deterministically-seeded, **never-trained** checkpoint | **Built (MS7)** — framework-validation smoke test only, explicitly not a segmentation-accuracy claim (see §4) |
| MSD Task07_Pancreas | 3-D medical segmentation | nnU-Net-style, trained externally (~1–2 GPU-days/fold, the single heaviest Phase-A item, Dataset Selection Audit §6) | Not yet acquired |
| CIFAR-10 (+ CIFAR-10.1 — reused unchanged) | Image classifier | Small train or pretrained checkpoint | Not yet acquired |
| Kvasir-SEG | Polyp segmentation | Small fine-tune (~1 hr, Dataset Selection Audit §6) | Not yet acquired |

**Accepted model families are constrained only by:** (a) producing scores in
exactly the shape `wfcrc.prediction_sets.PredictionSetConstructor.construct`
already expects (`[H,W,K]`/`[K,H,W]` segmentation, `[K]` classification —
`MS6_ARCHITECTURE_SPEC.md` §3.4), and (b) running through the PyTorch
runtime confined to `wfcrc/models/` (frozen decision Q1, §3 below). No
architecture is preferred or excluded on scientific grounds beyond those
two structural constraints and the reproducibility/provenance requirements
in §3–§5 — this document does not pick a specific Cityscapes/Pancreas/CIFAR/
Kvasir architecture, that remains an engineering decision for whoever
implements each remaining `ScoreProvider`.

### 1.2 Comparator baselines (compete against WF-CRC's calibration layer)

Per Experiment Blueprint §9–§10 — these are **calibration-layer**
comparators, not base-model alternatives; they consume the *same* cached
base-model scores as WF-CRC itself wherever architecturally possible:

| Baseline | Role |
|---|---|
| Vanilla CRC / split conformal (LAC) | Marginal-risk/coverage reference; exposes the conditional & robust gaps (E1, E2, E3, E6, E7) |
| Temperature / selective scaling | Standard, non-guaranteed UQ pre-empting "missing baseline" (E11) |
| Deep ensembles, MC-dropout | Standard, non-guaranteed UQ (baseline suite) |
| Gibbs–Cherian–Candès (conditional coverage) | Closest *coverage* competitor (E2) |
| Cauchois–Duchi (f-div) / Lévy–Prokhorov robust-CP | Closest *coverage* competitor under shift (E3, E4) |
| AA-CRC, sem-CRC | Closest *conditional-risk* competitors (E2) |
| Pooled K-fold WF-CRC, total-n inflation WF-CRC, fixed-η WF-CRC | Negative-control / ablation baselines confirming frozen P3/P4 (E7) — already implemented as test-only harnesses, `tests/unit/calibration/test_negative_controls.py` |

Comparator baselines are **not** subject to this document's checkpoint/
provenance requirements when they are pure recomputation over an already-
cached WF-CRC score (e.g. vanilla CRC, the ablation harnesses); they **are**
subject to §2–§5 in full when they require their own separately-trained
model (temperature scaling's own calibration, deep ensembles, MC-dropout,
Gibbs, robust-CP, AA-CRC/sem-CRC), since those are themselves base models
under §1.1's definition.

---

## 2 · Inference-Only Policy

**No model is trained inside this repository.** This is a repeated,
explicit frozen decision (`MS6_ARCHITECTURE_SPEC.md` §8.1 Q1: "No
training-framework functionality is introduced into WFCRC... MS6 only
loads and runs inference against an already-trained checkpoint"), not an
MS8-specific choice — MS8 only restates it as binding policy for E1–E12
execution:

- Base-model training (or fine-tuning, e.g. Kvasir's documented ~1 hr step)
  happens **outside** this repository, using whatever external tooling is
  appropriate (e.g. an nnU-Net-style pipeline for MSD-Pancreas) — WFCRC
  wraps the resulting checkpoint as a `ScoreProvider` **adapter**, never
  reimplements the training framework internally.
- Where a suitable public pretrained checkpoint exists (Cityscapes, CIFAR-10
  by default), it is preferred over training from scratch (Dataset
  Selection Audit §6: "prefer a public pretrained checkpoint").
- The one disclosed exception — MSD-Hippocampus's MS7 checkpoint — is a
  **never-trained**, deterministically-seeded network used purely to
  validate that scores flow correctly end-to-end through the frozen
  pipeline (`LossTableBuilder → WFCRCCalibrator → Verifier → run_experiment`).
  It carries **no segmentation-accuracy claim whatsoever** and must not be
  cited as a Hippocampus result in any E1/E9/E10/E11 table — those
  experiments require a genuinely trained (or credibly pretrained) checkpoint
  before their results can be reported as scientific findings, not smoke
  tests. Continuing to use the never-trained checkpoint past the smoke-test
  stage would silently convert a pipeline-validation artifact into a
  fabricated scientific result — this is flagged here explicitly so it is
  never done by omission.
- No optimizer, forward+backward training step, or gradient update may ever
  be constructed anywhere in `wfcrc/models/` outside of a clearly-labeled,
  never-checkpointed pipeline-validation utility (the same category the MS7
  `create_untrained_checkpoint` function already occupies).

---

## 3 · Checkpoint Acquisition

- **Source priority:** (1) public pretrained checkpoint from the
  architecture's original authors or a well-known reproduction, (2) a
  checkpoint trained externally by this research program's own team using
  a documented, reproducible external training pipeline, (3) never a
  checkpoint of unknown or undocumented origin.
- **Recorded per checkpoint, before it is used by any experiment:**
  architecture name/version, training dataset and its **exact** split
  boundary (which images/cases trained it — required input to §4's
  provenance check), training framework and version, and a stable download
  URL or internal training run identifier.
- **Storage:** checkpoints are content-addressed by `checkpoint_fingerprint`
  (a SHA-256-family hash of the checkpoint file's bytes,
  `wfcrc.utils.io.content_hash`) and loaded via `load_checkpoint`
  (`wfcrc/models/checkpoint.py`, MS7). The fingerprint is a **file-identity**
  hash, not a semantic weight-identity hash — re-saving numerically
  identical weights through `torch.save` is not guaranteed to reproduce the
  same fingerprint (disclosed at MS7 freeze); this is acceptable for its
  actual purpose (cache-key uniqueness) but must not be read as a stronger
  "these weights are provably identical" claim.

---

## 4 · Checkpoint Provenance (R4 / R-CKPT1)

**Mandatory, not optional, before any `ScoreProvider` using an externally-
sourced checkpoint is registered for E1–E12 use** (Dataset Selection Audit
risk R4; `MS6_ARCHITECTURE_SPEC.md` §3.5, §7):

- A `CheckpointProvenance` record (`{checkpoint_id, trained_on_ids}`) must
  be constructed for every checkpoint whose training data this project does
  not directly control (i.e. every case in §3's source-priority items 1–2).
- `assert_no_checkpoint_leakage(provenance, cal_ids, test_ids)` must be
  called — and must pass — before that checkpoint's scores are used to
  build any calibration or test `LossTable`. A failure raises
  `CheckpointProvenanceError` and blocks the experiment; it is never
  silently bypassed.
- This check is **narrower and complementary to**, not a replacement for,
  the frozen dataset-level A1 hygiene gate
  (`wfcrc.datasets.base.assert_split_disjoint`, enforced by
  `docs/DATASET_SPLIT_POLICY.md` §2/§4): the A1 gate verifies WFCRC's own
  train/calibration/test id lists are mutually disjoint; this check
  additionally verifies that a *third-party* checkpoint's own undisclosed
  training data does not overlap WFCRC's calibration/test pools — a
  distinct failure mode the A1 gate has no visibility into.
- The MS7 Hippocampus checkpoint is exempt from this specific check by
  construction (a never-trained checkpoint has no training-time data
  exposure for R-CKPT1 to guard against — already disclosed at MS7
  freeze), **not** because provenance checking is optional in general.

---

## 5 · Deterministic Inference

- Every `ScoreProvider.scores_for`/`scores_batch` call must be
  deterministic given a fixed checkpoint and input: no dropout or other
  stochastic layer may remain active at inference time (`model.eval()` or
  the PyTorch-runtime equivalent, mandatory), and no random augmentation
  may be applied to an evaluation-time input.
- Scores are cached read-through, keyed on `(model_fingerprint, id_)` via
  the frozen `wfcrc.utils.cache.Cache` — an unchanged checkpoint and input
  id must always resolve to the identical cached score, never a fresh
  stochastic recomputation.
- Any randomized *preprocessing* step upstream of inference (none currently
  exists for the frozen Phase-A preprocessing functions,
  `wfcrc.datasets.preprocessing`, which are pure deterministic transforms)
  must route through `wfcrc.utils.seeds.derive_seed`, per the project-wide
  no-bare-global-RNG policy — this would apply to a future MS6.3-family
  loader's corruption-parameter sampling, not to inference itself.

---

## 6 · Preprocessing Requirements

- **Identical across train/calibration/test roles** (Blueprint §6: "no
  split-dependent preprocessing") — a `ScoreProvider` must apply the exact
  same resize/normalize (2-D) or resample/intensity-normalize (3-D)
  transform regardless of which WFCRC role the input example belongs to.
- **2-D (Cityscapes/ACDC/CIFAR/Kvasir):** `wfcrc.datasets.preprocessing.
  resize_and_normalize` — target size and per-channel mean/std are
  externally supplied constants (frozen at loader-construction time), never
  a per-image or per-split statistic.
- **3-D (MSD):** `wfcrc.datasets.preprocessing.resample_volume` for spatial
  resampling (image data only — **never** applied to label volumes, since
  it is linear-interpolation-only and would corrupt discrete label values;
  already empirically confirmed at MS6.3A, "3.8% non-integer voxels" on a
  real resampled label) plus a **per-volume z-score normalization**
  (`_zscore_normalize`, MS7 finding) for real, unnormalized MRI scanner
  intensities — disclosed as a necessary, standard CNN input-prep step
  distinct from the frozen 2-D transform, not a new preprocessing
  *policy* beyond "normalize consistently."
- **No data augmentation** is used anywhere in this research program's
  frozen scope (`docs/DATASET_SPLIT_POLICY.md` §5, "Augmentation leakage"
  row) — if a future external training process introduces augmentation
  upstream of a checkpoint this project consumes, augmented views of a
  training example must never be scored as if they were calibration/test
  data.

---

## 7 · Versioning

- **Model/architecture version:** recorded per checkpoint (§3) —
  architecture name and, where applicable, a version tag or commit hash of
  the reference implementation.
- **Checkpoint version:** `checkpoint_fingerprint` (file-content hash, §3);
  every `Manifest` records the fingerprint transitively via the
  `LossTable`/score cache key, so a run's manifest is traceable back to the
  exact checkpoint file used (not merely "a Cityscapes segmenter" in the
  abstract).
- **`ScoreProvider` code version:** ordinary git provenance — `Manifest.
  git_commit` (`get_git_commit`) records the commit that produced a given
  run, covering both calibration-engine code and the `ScoreProvider`
  implementation itself.
- **Superseding a checkpoint** (a retrain, a newer public release) is a
  new, distinct `checkpoint_fingerprint` and therefore a new cache
  namespace and a new set of `Manifest` records — never an in-place
  overwrite of a previously-reported result's checkpoint.

---

## 8 · Hardware Requirements

- **DL runtime:** PyTorch, confined strictly to `wfcrc/models/`
  (`MS6_ARCHITECTURE_SPEC.md` §8.1 Q1) — the calibration/evaluation core
  (`wfcrc.calibration`, `wfcrc.ambiguity`, `wfcrc.losses`,
  `wfcrc.prediction_sets`, `wfcrc.evaluation`, `wfcrc.runner`) remains
  NumPy-only and must never acquire a PyTorch dependency, directly or
  transitively.
- **Current environment:** CPU-only (`torch==2.13.0+cpu`, no CUDA
  toolkit) — every `ScoreProvider` must support `device="cpu"` and must not
  hard-require a GPU to run inference. This is sufficient for
  Hippocampus-scale inference (already validated, MS7) but was not
  benchmarked for the heavier Cityscapes/Pancreas models; wall-clock
  expectations for those follow the Dataset Selection Audit's own §6
  planning estimates (base-model **training**, which per §2 above happens
  outside this repository regardless of device).
- **GPU use, if provisioned for training or heavier inference:** governed
  by the same determinism requirement (§5) — any GPU-specific
  nondeterminism (e.g. non-deterministic cuDNN kernels) must be disabled or
  explicitly accounted for, since this project's reproducibility bar
  (`docs/reproducibility.md`) makes no exception for GPU execution.
- **Base-model training hardware** (external to this repository, §2): per
  Dataset Selection Audit §6, a single modern 24–40 GB GPU is the planning
  assumption; MSD-Pancreas (~1–2 GPU-days/fold) is the single heaviest
  item in the entire Phase-A campaign.

---

## Connections

`docs/EXPERIMENT_PROTOCOL.md` · `docs/DATASET_SPLIT_POLICY.md` ·
`docs/RESULTS_SCHEMA.md` · `MS6_ARCHITECTURE_SPEC.md` (§3.4 Score
Providers, §3.5 Checkpoint Management, §8.1 Q1) · `PROJECT_CONTEXT.md`
(MS7 section) · `wfcrc/models/checkpoint.py` ·
`wfcrc/models/scores/hippocampus_segmenter.py` · Research Vault:
`Paper 1 - DATASET SELECTION AUDIT.md`, `Paper 1 - EXPERIMENT BLUEPRINT.md`
