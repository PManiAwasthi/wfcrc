# Production Readiness Audit — WFCRC Research Program (Paper 1)

> **Status:** Frozen planning/audit artifact (MS11). **Version:** 1.0.
> **Date:** 2026-07-18. **Scope:** Tasks 1–3 of the MS11 brief — a
> production model acquisition plan, a dataset readiness audit, and a
> review of every methodological decision left unresolved by MS8–MS10.
> **This document downloads nothing, trains nothing, and resolves no
> open methodology silently.** Every recommendation below is explicitly
> labeled as a recommendation, distinct from any frozen decision; nothing
> here becomes binding merely by appearing in this file.

---

## Task 1 · Production Model Strategy

For every dataset in the frozen Phase-A suite (`docs/EXPERIMENT_PROTOCOL.md`
§3, `Paper 1 - DATASET SELECTION AUDIT.md`). Checkpoint-source leads below
were checked against real web search results during this milestone (not
recalled from training data alone) and are graded by confidence; **no URL
is asserted as the acquisition source without that grading**, and none has
been downloaded or verified end-to-end.

### 1.1 Cityscapes

- **Required model:** semantic segmentation, 19-class Cityscapes taxonomy.
- **Pretrained checkpoint exists?** **Yes, with high confidence** — Cityscapes
  is one of the most standard segmentation benchmarks; `open-mmlab/mmsegmentation`'s
  own model zoo publishes multiple architectures' checkpoints trained on it
  (PSPNet, DeepLabV3+, and others), confirmed present via a live search
  during this milestone (`mmsegmentation.readthedocs.io`, GitHub
  `open-mmlab/mmsegmentation`). NVIDIA's NGC catalog also lists a
  pretrained SegFormer-on-Cityscapes checkpoint. **Not yet downloaded,
  loaded, or verified against this repository's shape contracts.**
- **Satisfies frozen MODEL_POLICY?** **Not yet determinable** — MODEL_POLICY
  §4 (R4/checkpoint provenance) requires knowing the *specific* checkpoint's
  training split before it can be used: standard public Cityscapes
  checkpoints are conventionally trained on the official `train` split
  only, which per `docs/DATASET_SPLIT_POLICY.md` §3.3 never overlaps
  WFCRC's own calibration/test pool (carved entirely from `val`) — a
  favorable, low-risk situation *if* that convention holds for the
  specific checkpoint selected, but this must be confirmed per-checkpoint
  (`assert_no_checkpoint_leakage`), never assumed from the general
  convention alone.
- **Local training required?** No — a public pretrained checkpoint should
  be preferred (Dataset Selection Audit §6).
- **Checkpoint source:** `open-mmlab/mmsegmentation` model zoo (best
  identified lead; not yet selected or acquired).
- **Expected preprocessing:** standard resize/normalize
  (`wfcrc.datasets.preprocessing.resize_and_normalize`), with mean/std
  constants matching whichever checkpoint's own training normalization is
  selected (not yet known until a specific checkpoint is chosen).
- **Blocking issue:** the raw Cityscapes dataset itself is **not** locally
  present in this environment (registration-gated, Dataset Selection Audit
  R3) — see Task 2.

### 1.2 Cityscapes-C

- **Required model:** the same Cityscapes segmenter, reused unchanged
  (Dataset Selection Audit: "zero extra training").
- **Pretrained checkpoint / training:** identical status to §1.1.
- **Checkpoint source:** identical to §1.1.
- **Expected preprocessing:** Cityscapes preprocessing, plus the frozen
  15-corruption × 5-severity suite (`wfcrc.datasets.corruptions.apply_corruption`,
  already built and protocol-verified, MS6.2) applied on the fly to the
  Cityscapes `val` images at read time — no separate model or checkpoint
  needed.

### 1.3 ACDC (driving)

- **Required model:** the Cityscapes segmenter, reused unchanged (Dataset
  Selection Audit: "reuses the Cityscapes model unchanged... zero extra
  training").
- **Pretrained checkpoint / training:** identical status to §1.1 — ACDC
  itself never trains anything (`docs/DATASET_SPLIT_POLICY.md` §3.4).
- **Checkpoint source:** identical to §1.1.
- **Expected preprocessing:** identical Cityscapes-style resize/normalize.
  **Real-data finding (this milestone, §Task 2):** ACDC's labels are
  genuinely Cityscapes-*format*-compatible (`_gt_labelTrainIds.png`/
  `_gt_labelIds.png` naming, matching Cityscapes' own convention exactly),
  but the on-disk layout differs materially from plain Cityscapes (weather-
  condition and GoPro-sequence subdirectories, not `<split>/<city>/`) — a
  Cityscapes-format loader will need ACDC-specific path-discovery logic,
  not merely a different `root_dir`/`label_map` constructor parameter as
  `MS6_ARCHITECTURE_SPEC.md` §3.3's own sketch anticipated.

### 1.4 MSD Task04_Hippocampus

- **Required model:** 3-D medical segmentation (nnU-Net-style architecture
  family).
- **Pretrained checkpoint exists?** **No** — confirmed by a repository-wide
  search for `*.pt`/`*.pth`/`*.ckpt`/`*.h5`/`*.onnx`/`*.safetensors` files
  at MS7 freeze, re-confirmed this milestone (only the MS10 pilot's own
  never-trained smoke checkpoint exists anywhere in this repository).
- **Satisfies MODEL_POLICY?** N/A — the only checkpoint in existence
  (`create_untrained_checkpoint`'s output) is explicitly, by design, **not**
  a scientific checkpoint (MODEL_POLICY §2's own explicit warning: "must
  not be cited as a Hippocampus result in any scientific table").
- **Local training required?** **Yes**, for a scientifically meaningful
  result — no suitable public pretrained checkpoint for this exact task
  was identified.
- **Checkpoint source:** none identified; would need to be trained
  externally (Q1's own "training happens outside this repository" policy).
- **Expected preprocessing:** already built and validated (MS7):
  per-volume z-score normalization (`_zscore_normalize`) plus shape-padding
  to a multiple of 4 (`_pad_to_multiple`) — both already exercised against
  real data.

### 1.5 MSD Task07_Pancreas

- **Required model:** 3-D medical segmentation, same architecture family
  as Hippocampus (Dataset Selection Audit: "identical architecture family
  scales to Task07_Pancreas... without a different architecture class").
- **Pretrained checkpoint exists?** **Possibly, moderate confidence, not
  confirmed this milestone** — nnU-Net's own project historically
  distributed pretrained models for several Medical Segmentation Decathlon
  tasks (including, by recollection, a Task07_Pancreas entry) via its own
  `nnUNet_download_pretrained_model` mechanism and/or Zenodo; a live search
  this milestone corroborates that nnU-Net pretrained models are hosted on
  Zenodo and downloadable via the official `MIC-DKFZ/nnUNet` tooling, but
  did **not** directly confirm a specific, currently-live Task07_Pancreas
  entry — **this lead must be verified directly against the nnU-Net
  project before being relied on**, not assumed from this search alone.
- **Satisfies MODEL_POLICY?** Not yet determinable (same R4 caveat as
  §1.1) — additionally, an nnU-Net-family checkpoint would need adapting
  into this repository's own `_TinyUNet3D`-style `ScoreProvider` shape
  contract (or a new provider class built specifically for whatever
  architecture is actually obtained), not a drop-in replacement.
- **Local training required?** **Yes, if no suitable pretrained checkpoint
  is confirmed** — per the Dataset Selection Audit's own cost estimate
  (§6), this is **the single heaviest compute item in the entire Phase-A
  campaign** (~1–2 GPU-days per fold).
- **Checkpoint source:** `MIC-DKFZ/nnUNet` project / Zenodo (lead only,
  unconfirmed) or local training.
- **Expected preprocessing:** same family as Hippocampus (per-volume
  normalization, shape padding), **not yet validated against real Pancreas
  volumes** — MS6.3A's own real-data validation pass covered Hippocampus
  only; Pancreas real-data preprocessing compatibility remains unverified
  (see Task 2).

### 1.6 CIFAR-10

- **Required model:** image classifier, 10 classes.
- **Pretrained checkpoint exists?** **Yes, high confidence** — multiple
  actively-maintained, directly loadable repositories were confirmed via
  live search this milestone: `chenyaofo/pytorch-cifar-models` (ResNet/VGG/
  MobileNetV2/ShuffleNetV2/RepVGG on CIFAR-10/100, loadable via PyTorch's
  own `torch.hub` API — a natural fit for this project's PyTorch-confined
  Q1 architecture) and `huyvnphan/PyTorch_CIFAR10` (TorchVision-architecture
  weights). Neither has been downloaded or verified in this environment.
- **Satisfies MODEL_POLICY?** Not yet determinable (R4 caveat) — but
  structurally lower-risk than Cityscapes/Pancreas: CIFAR-10's own
  official test set is what WFCRC's calibration/test pool is carved from
  (`docs/DATASET_SPLIT_POLICY.md` §3.5), and any checkpoint trained on the
  official 50,000-image train set (the universal convention for every
  public CIFAR-10 classifier) would not overlap it.
- **Local training required?** No — Dataset Selection Audit itself frames
  this as "small train or pretrained" (~1–2 hrs either way); a pretrained
  checkpoint is the lower-effort, equally-valid choice.
- **Checkpoint source:** `chenyaofo/pytorch-cifar-models` (best identified
  lead, `torch.hub`-loadable).
- **Expected preprocessing:** standard per-channel normalization (CIFAR-10's
  own well-known mean/std constants); no new preprocessing utility
  required (`resize_and_normalize` already covers this shape).

### 1.7 CIFAR-10.1

- **Required model:** the CIFAR-10 classifier, reused unchanged (Dataset
  Selection Audit: "answers real cls shift", zero extra training).
- **Pretrained checkpoint / training:** identical status to §1.6.
- **Checkpoint source:** identical to §1.6.
- **Expected preprocessing:** must exactly match whatever preprocessing
  the CIFAR-10 classifier was itself trained/calibrated with — no
  independent preprocessing decision for this dataset.

### 1.8 Kvasir-SEG

- **Required model:** binary polyp segmentation.
- **Pretrained checkpoint exists?** **Plausible, confidence not fully
  confirmed this milestone** — PraNet (Fan et al., 2020) is a well-known,
  frequently-cited architecture with a strong published Kvasir-SEG
  benchmark (mean Dice ≈0.898, confirmed via live search this milestone,
  arXiv:2006.11392); ColonSegNet is a second, faster-inference candidate.
  Live search corroborated both architectures' existence and benchmark
  results but did **not** directly surface a confirmed, currently-live
  checkpoint download link for either during this session — **this lead
  requires direct verification against the original authors' own
  repository before being relied on.**
- **Satisfies MODEL_POLICY?** Not yet determinable — additionally, per the
  Dataset Selection Audit's own plan (§6), Kvasir-SEG involves a "~1 hr
  fine-tune" step, meaning (unlike ACDC/Cityscapes-C/CIFAR-10.1) **some
  Kvasir-SEG data trains/fine-tunes the model** — the full train ⟂
  calibration ⟂ test separation applies here in full
  (`docs/DATASET_SPLIT_POLICY.md` §3.7's own note), which in turn depends
  on Kvasir-SEG's still-open split-unit question (§3 below) being resolved
  *before* any fine-tuning split is drawn.
- **Local training required?** A brief fine-tune (~1 hr, not full
  from-scratch training) of a suitable pretrained polyp-segmentation or
  general medical-image-segmentation backbone.
- **Checkpoint source:** PraNet / ColonSegNet original-author repositories
  (leads only, unconfirmed).
- **Expected preprocessing:** resize/normalize (already anticipated,
  `DATASET_METADATA["kvasir_seg"]`'s own recorded note).

### 1.9 Cross-cutting requirement (every dataset)

**No checkpoint anywhere in this plan can be declared MODEL_POLICY-
compliant until it is actually selected and its own training-split
provenance is documented** (`docs/MODEL_POLICY.md` §3–§4,
`assert_no_checkpoint_leakage`). This audit identifies *leads*, grades
their confidence, and flags the structural risk each carries — it does
not, and cannot, pre-clear any specific checkpoint file that has not yet
been examined.

---

## Task 2 · Dataset Readiness Audit

**Filesystem state independently verified this milestone** (not assumed
from prior records): `datasets/` now contains real, locally-acquired data
for **six of the seven** Phase-A dataset artifacts — a materially more
advanced state than MS6–MS10's own audits recorded, and reported here
precisely.

| Dataset | Extraction | Directory layout | Loader compatibility | Metadata | Annotations | Split readiness | Preprocessing compatibility |
|---|---|---|---|---|---|---|---|
| **Cityscapes** | ❌ Not present locally (registration-gated) | N/A | ❌ No `CityscapesFormatLoader` exists in `wfcrc/datasets/loaders/` (confirmed: only `msd.py` exists there) | ✅ `DATASET_METADATA["cityscapes"]` frozen (MS6.2) | N/A (no local data) | N/A | N/A |
| **Cityscapes-C** | N/A (derived from Cityscapes) | N/A | ❌ No corruption-wrapper loader built yet; the underlying `apply_corruption` utility is built and protocol-verified (MS6.2) | ✅ frozen | N/A | N/A | ✅ `apply_corruption` already validated on synthetic arrays |
| **ACDC (driving)** | ✅ Extracted, real, 18,046 files (`datasets/ACDC/`) | Weather-condition (`fog`/`night`/`rain`/`snow`) → `train`/`val`/`test`(+`_ref`) → GoPro-sequence-folder (`GOPR0475`, ...) → per-frame files — **differs from plain Cityscapes' `<split>/<city>/` structure**, as anticipated only partially by `MS6_ARCHITECTURE_SPEC.md` §3.3 | ❌ No loader exists; real layout confirms more accommodation is needed than a parameter swap (see §1.3) | ✅ frozen | ✅ Confirmed genuinely Cityscapes-compatible label encoding (`_gt_labelTrainIds.png`, `_gt_labelIds.png`, `_gt_labelColor.png`, `_gt_invGray.png`/`_gt_invIds.png` all present) | ⚠️ `docs/DATASET_SPLIT_POLICY.md` §3.4's 50/50 calibration/test policy (pooled official train+val) was written before real data was confirmed; ratio itself unchanged, but should be re-checked against the real per-condition case counts before use | ⚠️ Not yet exercised — the frozen 2-D `resize_and_normalize` should apply, but has not been run against a real ACDC image |
| **MSD Task04_Hippocampus** | ✅ Extracted, real, 260 labelled cases (MS6.3A/MS7-validated) | ✅ Standard MSD `imagesTr`/`imagesTs`/`labelsTr`/`dataset.json` | ✅ `MSDNiftiLoader` (registered `DATASETS["msd_hippocampus"]`) | ✅ frozen | ✅ Real-data-validated (MS6.3A: zero malformed/duplicate/mismatched cases) | ✅ Frozen 60/20/20 split policy defined; MS10 pilot additionally validated the reduced-scale (8/4) smoke variant end to end | ✅ Real-data-validated (MS7: z-score normalization, shape padding) |
| **MSD Task07_Pancreas** | ✅ Extracted, real, **281 images + 281 labels** — resolves `docs/DATASET_SPLIT_POLICY.md` §8 item 4's open item (the Dataset Selection Audit's provisional "281" figure is now confirmed exact against a real archive) | ✅ Standard MSD layout (`imagesTr`/`labelsTr`/`dataset.json`), same shape as Hippocampus | ✅ Mechanically compatible — `MSDNiftiLoader(root_dir, task="Task07_Pancreas", ...)` should work unchanged (the class is already `task`-parameterized), but **this has not been exercised against the real Pancreas archive** the way Hippocampus was in MS6.3A | ✅ frozen | ⚠️ Not yet independently validated the way Hippocampus's 260 cases were (no MS6.3A-equivalent pass has run against this real archive: duplicate ids, NaN/Inf, label-value range, image/label shape match, per-class imbalance — Dataset Selection Audit's own flagged "severe imbalance" — are all unverified) | ⚠️ Split policy defined (60/20/20, shared with Hippocampus for consistency) but never exercised against real per-class balance, which `docs/DATASET_SPLIT_POLICY.md` §3.2 itself flags as needing a non-degenerate-representation check once real data is on hand — **that check has still not been performed** | ⚠️ Not yet exercised — no real Pancreas volume has been run through `resample_volume`/normalization in this repository |
| **CIFAR-10** | ✅ Extracted, real, standard 5 train batches + 1 test batch + `batches.meta` (`datasets/CIFAR10/`) | ✅ Standard Krizhevsky binary/pickle format | ❌ No `CifarLoader` exists in `wfcrc/datasets/loaders/` | ✅ frozen | N/A (labels are simple integer class ids, standard format) | N/A (no loader to split against yet) | N/A |
| **CIFAR-10.1** | ✅ Extracted, real — both `v4` (2,021 images) and the recommended `v6` (2,000 images, class-balanced) `.npy` files present (`datasets/CIFAR10_1/datasets/`), plus the full upstream repository (code/scripts not needed) | ✅ Matches `DATASET_METADATA["cifar10_1"]`'s own recorded `recommended_variant: v6` exactly | ❌ Same `CifarLoader` gap as CIFAR-10 (shares the loader family) | ✅ frozen | N/A | N/A | N/A |
| **Kvasir-SEG** | ✅ Extracted, real, **exactly 1,000 images + 1,000 masks** (`datasets/Kvasir_SEG/kvasir-seg/Kvasir-SEG/`), matching `DATASET_METADATA`'s expected count | ✅ `images/`, `masks/`, plus a `kavsir_bboxes.json` (object-detection bounding-box annotations, not needed for segmentation) | ❌ No `KvasirLoader` exists | ✅ frozen (license still unconfirmed, per MS6.2's own honest disclosure) | ✅ Image/mask pairing appears 1:1 by count; **not yet independently verified id-by-id** the way Hippocampus was | ❌ **Still fully open** — inspecting the real, now-locally-available data did **not** resolve `docs/DATASET_SPLIT_POLICY.md` §8 item 1's split-unit question: filenames are opaque hashes (e.g. `cju0qkwl35piu0993l0dewei2`) with no visible procedure/patient grouping, and `kavsir_bboxes.json` is bounding-box metadata, not procedure identifiers — **whether per-procedure grouping is recoverable remains unknown even with the real archive in hand** | ⚠️ Not yet exercised |

### 2.1 Dataset-specific issues identified before experiments begin

1. **Cityscapes itself is the single largest blocker in the entire Phase-A
   suite**: neither the raw data nor a concrete loader exists, and three
   other artifacts (Cityscapes-C, ACDC's model, ACDC's own preprocessing
   parity) all depend on it.
2. **No concrete `DatasetLoader` exists for six of the seven datasets** —
   only `MSDNiftiLoader` (Hippocampus, and mechanically for Pancreas) is
   built. The Cityscapes-format, CIFAR, and Kvasir loader families
   (`MS6_ARCHITECTURE_SPEC.md` §3.3) remain entirely unbuilt.
3. **ACDC's real on-disk layout is more complex than the architecture
   spec's own sketch anticipated** (weather-condition + GoPro-sequence
   nesting, not a flat city/split structure) — flagged now, before loader
   work begins, rather than discovered mid-implementation.
4. **MSD Task07_Pancreas has never been real-data-validated** the way
   Hippocampus was at MS6.3A — id/shape/label/imbalance integrity checks
   remain outstanding despite the raw archive now being present and
   count-verified.
5. **Kvasir-SEG's split-unit question is still open** even with the real
   archive available for direct inspection — this is not merely a
   documentation gap; the underlying data itself does not obviously carry
   the answer.

---

## Task 3 · Methodological Decision Audit

Every methodological item MS8–MS10 left explicitly open, reviewed here —
**recommendations are recommendations, not frozen decisions**; adopting
any of them requires explicit sign-off, per this project's own "stop and
document" discipline.

| # | Item | Why it exists | Must it be resolved before E1? | Recommended default (a recommendation, not a decision) |
|---|---|---|---|---|
| 1 | **λ-grid resolution per dataset** (`docs/EXPERIMENT_PROTOCOL.md` §4) | No frozen document specifies a numeric grid density/range — it is genuinely dataset- and score-range-dependent, and the Algorithm Specification deliberately leaves it as a free input, not a derived quantity. | **Yes** — every experiment's `CalibrationConfig.lambda_grid` must be fixed before that experiment's calibration runs (a grid changed mid-campaign would silently change results). | A grid dense enough that the finest step does not change `lambda_hat` by more than a negligible fraction of the target metric's own natural scale — concretely, start from a moderately fine, evenly-spaced grid per dataset/set-constructor pairing (e.g. the same order of density already used for classification (`ThresholdSets`, `λ ∈ [0,1]`) vs. segmentation dilation-radius grids (`MorphologicalSets`, integer or half-integer radii)), and confirm via E9's own hyperparameter-sensitivity sweep that efficiency varies smoothly rather than jumping — i.e., **use E9 itself to validate the grid choice**, not to select it post hoc. |
| 2 | **Campaign base seed** (`docs/EXPERIMENT_PROTOCOL.md` §4, §7) | The project's own reproducibility discipline requires one fixed base seed per campaign, derived-fanout to every component — but no specific integer has been chosen for the E1–E12 campaign as a whole (MS7's own smoke pipeline used `seed=0` for a single illustrative run, not a campaign commitment). | **Yes** — must be fixed and recorded before E1 executes, then held fixed for the primary run (only `R` resample sub-seeds vary, derived from it). | `seed = 0`, for continuity with every prior milestone's own convention (MS5's golden-file reference experiment, MS7's smoke pipeline, MS10's pilot) — not a claim that `0` is statistically special, purely a documented, consistent choice avoiding an arbitrary-looking large integer. |
| 3 | **E3/E4 shift-radius (`ρ`) estimation procedure** (`docs/EXPERIMENT_PROTOCOL.md` §4) | Robust-family experiments need `ρ` set to (or credibly bounding) the actual shift magnitude; no frozen document specifies *how* to estimate that magnitude from data rather than guessing it. | **Yes**, for E3/E4 specifically (not for E1/E2/E6–E12, which do not need a shift-calibrated `ρ`). | A held-out-domain KL/CVaR-radius estimate computed **only from the source domain's own calibration data plus a small labelled sample of the target/shifted domain reserved exclusively for this estimation step (never touching calibration or test)** — i.e., treat `ρ`-selection itself as a third, disjoint role requiring its own explicit "radius-estimation" id list in a future `SplitManifest`-equivalent, not silently reuse calibration or test ids for it. This needs explicit sign-off before E3/E4 execute, since it is itself a new methodological decision, not a mechanical default. |
| 4 | **E3 monotone-trend test choice** (`docs/EXPERIMENT_PROTOCOL.md` §5.2) | The Blueprint names "a monotone-trend test" with no specific procedure; `wfcrc.evaluation.metrics` does not implement one. | **Yes**, for E3 specifically. | Mann-Kendall trend test (rank-based, no distributional assumption, natural pairing with the already-adopted Wilcoxon/rank-based family) — already recommended in `docs/EXPERIMENT_PROTOCOL.md` §5.2; repeated here as still awaiting sign-off, not yet adopted. |
| 5 | **E6 effect-size measure** (`docs/EXPERIMENT_PROTOCOL.md` §5.4) | The Blueprint specifies significance tests and CIs but no effect-size measure, so "statistically significant" and "practically meaningful" risk being conflated in reported tables. | **Yes**, for E2/E6 (wherever `paired_wilcoxon` is reported). | Matched-pairs rank-biserial correlation `r = Z/√N`, the standard companion to a Wilcoxon signed-rank test — already recommended in `docs/EXPERIMENT_PROTOCOL.md` §5.4; awaiting sign-off. |
| 6 | **E11 ECE binning scheme** (`docs/EXPERIMENT_PROTOCOL.md` §6.4) | `wfcrc.evaluation.metrics` has no ECE (Expected Calibration Error) function; a binning scheme (equal-width vs. equal-mass, bin count) materially changes the reported number. | **Yes**, for E11 specifically. | Equal-mass (quantile) binning with 10–15 bins — a standard, less bin-placement-sensitive default than equal-width binning for a small-to-moderate test set (E11's own datasets, Cityscapes + MSD-Hippocampus); still requires explicit sign-off and, ideally, its own small sensitivity check (does the reported ECE change materially at 10 vs. 15 bins?) before being treated as final. |
| 7 | **AA-CRC / sem-CRC identification** (`docs/baseline_catalog.md` rows 10–11) | No algorithm is identifiable anywhere in the frozen Research Vault for either acronym (confirmed by exhaustive grep, MS9). | **Only for E2's own baseline-completeness claim** — E2 can still execute and report WF-CRC vs. vanilla CRC vs. `GroupConditionalCRC` (the Gibbs proxy) without them; their absence should be disclosed in any E2 write-up, not silently omitted. | No default can be scientifically recommended for an unidentified method — this requires either a citation from the user/PI, or an explicit, dated decision to drop these two rows from the baseline suite (`docs/baseline_catalog.md`'s own summary already frames this choice). |
| 8 | **Lévy–Prokhorov robust-CP** (`docs/baseline_catalog.md` row 9) | The frozen `Paper 1 - FRAMEWORK SPECIFICATION.md` itself names this divergence family "future work... open gap" — not a gap this program ever committed to closing before E1–E12. | **No** — E3/E4 already have a robust-CP comparator (`RobustFDivergenceCP`, the Cauchois-Duchi KL-ball construction); this item does not block any frozen experiment's execution. | Leave deferred, exactly as the frozen Framework Specification itself already scopes it — no recommendation to build it now. |
| 9 | **Generic cross-baseline `Verifier`** (`docs/PILOT_REPORT.md` §8 item 1) | The frozen `Verifier` only applies to `wfcrc`/`vanilla_crc`; MS10 confirmed no equivalent exists for the other eight registered baselines. | **No** — this is an engineering nicety for reporting confidence, not a blocker: every baseline's *calibration procedure itself* already ran correctly in MS9's own tests and MS10's real-data pilot; the *frozen* `wfcrc`/`vanilla_crc` results are already verifier-checked. | Not urgent; worth building before a from-scratch baseline is trusted at full campaign scale without any deterministic self-check, but not before E1 can begin. |
| 10 | **Multi-baseline `ExperimentRunner`** (`docs/PILOT_REPORT.md` §8 item 2) | `ExperimentRunner.run()` is hardcoded to `_build_family`/`WFCRCCalibrator`; every non-`wfcrc` baseline's pilot manifest was written by a one-off script helper, not the frozen orchestrator. | **No**, for the same reason as item 9 — a convenience/consistency improvement, not a validity blocker; E1–E12 could, in principle, be run via per-baseline scripts mirroring `scripts/pilot_ms10.py`'s own pattern, at the cost of some repetition. | Worth building once E1–E12's actual config-layer needs are clearer (this is exactly the still-unbuilt MS6.5/MS6.6 Config Resolver's own scope); not a prerequisite for starting E1. |

**Cross-cutting note:** items 1–6 (all methodology, not engineering) are
the genuine E1-execution blockers; items 7–8 are baseline-completeness
disclosures, not blockers; items 9–10 are engineering conveniences, not
blockers. This distinction — repeated explicitly in
`docs/EXPERIMENT_EXECUTION_CHECKLIST.md`'s own pre-flight gate — is itself
the main output of this audit.

---

## Connections

`docs/EXPERIMENT_PROTOCOL.md` · `docs/MODEL_POLICY.md` ·
`docs/DATASET_SPLIT_POLICY.md` · `docs/baseline_catalog.md` ·
`docs/PILOT_REPORT.md` · `docs/EXPERIMENT_EXECUTION_CHECKLIST.md` ·
`PROJECT_CONTEXT.md` (MS11 section) · Research Vault:
`Paper 1 - DATASET SELECTION AUDIT.md`, `MS6_ARCHITECTURE_SPEC.md`
