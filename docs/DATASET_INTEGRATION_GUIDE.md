# Dataset Integration Guide — WFCRC Research Program (Paper 1)

> **Status:** Frozen implementation guide (DI-1). **Version:** 1.0.
> **Date:** 2026-07-18. **Purpose:** the production-quality
> `DatasetLoader` standard every future dataset loader must follow,
> established using `wfcrc.datasets.loaders.msd.MSDNiftiLoader`/
> `MSDDataset` (MSD Task04_Hippocampus) as the **reference
> implementation** — read this document alongside that module's own
> docstring, which remains the fullest worked example of every rule
> below.
>
> **This document freezes a standard, not new datasets.** No Pancreas,
> ACDC, Kvasir-SEG, CIFAR-10, or CIFAR-10.1 loader is implemented here —
> see §6 for what each would require.

---

## 1 · Loader Architecture

Three layers, none of which this guide changes:

```
Dataset (ABC)              -- __iter__, __len__, ids(), labels(id_), meta()
DatasetLoader (ABC)        -- load(split_name) -> Dataset
SplitManifest              -- {train_ids, cal_ids, test_ids}, A1 hygiene gate
```

plus, additive since DI-1:

```
IntegrityIssue / IntegrityReport   -- shared value types for a concrete
                                       Dataset's own optional integrity
                                       check (wfcrc/datasets/base.py)
```

A concrete loader family (one per dataset *format*, not one per named
dataset — see §6) pairs exactly one `DatasetLoader` subclass with exactly
one `Dataset` subclass. The `DatasetLoader` subclass owns **discovery**
(finding and pairing raw files, validating a caller-supplied split
assignment, constructing the `SplitManifest`); the `Dataset` subclass owns
**access** (lazy per-id reads, no discovery logic of its own). This split
is not incidental — `MSDNiftiLoader.__init__` does all discovery/
validation once, up front, and hands `MSDDataset` only its own,
already-resolved slice of the case pool.

## 2 · Required Interfaces

Every concrete `Dataset` subclass **must** implement (the frozen ABC,
unchanged since M3):

| Method | Contract |
|---|---|
| `__iter__() -> Iterator[(id, X, Y)]` | Yield every example in this split, in a **stable, deterministic order** (never dict/set iteration order, never filesystem-listing order unless that listing is itself sorted deterministically). |
| `__len__() -> int` | The number of examples in this split. |
| `ids() -> Sequence[Hashable]` | The same ids `__iter__` yields, same order, as a standalone accessor. |
| `labels(id_) -> Any` | The label for one id. Must satisfy whatever downstream contract the frozen calibration/evaluation core requires for this data modality (e.g. `wfcrc.losses.base.LossEvaluator`'s `dtype == bool` requirement for segmentation masks — `MSDDataset.labels` binarizes for exactly this reason, see its own docstring §4). |
| `meta() -> dict[str, Any]` | At least `version`/`license`, sourced from `wfcrc.datasets.metadata.DATASET_METADATA` — never inline literal strings in the loader (`MSDDataset.meta`'s own pattern). |

Every concrete `DatasetLoader` subclass **must** implement:

| Method | Contract |
|---|---|
| `load(split_name) -> Dataset` | Raise `ValueError` for an unrecognized `split_name` (the frozen ABC's own sanctioned failure mode). |

**Recommended, additive per-id accessors** (not required by the ABC, but
established as the production standard by this milestone — see §2.1):

| Method | Contract |
|---|---|
| `image(id_)` (or the modality-appropriate equivalent name) | Direct, on-demand access to the raw input for one id, **without** requiring the label. Added to `MSDDataset` in DI-1 specifically because its absence forced `HippocampusScoreProvider` to consume `__iter__` once at construction just to build its own internal `id -> image` lookup — a workaround, not a design. A future loader should expose this from the start. |
| Format-specific header accessors (`spacing(id_)`, `orientation(id_)` for NIfTI; the natural analogues for other formats — pixel resolution, color space, bit depth, whatever a downstream `ScoreProvider`/preprocessing step might need) | Expose per-case metadata a caller *might* need, rather than assuming a dataset-wide constant. `MSDDataset.spacing`/`.orientation` never assume a shared value across cases — they read each case's own header, every time — because assuming uniformity is exactly the kind of claim that must be verified against real data (§3), not assumed from a spec. |

### 2.1 Why `image(id_)` is now required, not merely convenient

Before DI-1, `MSDDataset` exposed `labels(id_)`/`raw_labels(id_)`/
`spacing(id_)` but no image-only accessor — a real, previously-disclosed
gap (`wfcrc.models.scores.hippocampus_segmenter`'s own module docstring,
"Where the raw image comes from"). Every future loader must expose a
direct, per-id accessor for **every** piece of per-example data a
`ScoreProvider` might need (raw input, and any raw label variant beyond
the binarized/collapsed form `labels()` returns) — not just whatever
`__iter__` happens to yield in one pass. `__iter__` should be implemented
**in terms of** these accessors (as `MSDDataset.__iter__` now is), never
the reverse.

## 3 · Validation Rules

**Before writing a single line of loader code for a new dataset, verify
its real, on-disk structure — never assume it from a specification or a
secondary source.** This is not a suggestion; it is the single rule this
milestone's own real-data re-validation of Hippocampus (§4) exists to
demonstrate by example, and it is the rule the MS6.3A/MS7 record itself
already learned the hard way (a pre-acquisition case-count estimate was
wrong by 3 cases; a pre-acquisition spacing-uniformity guess was wrong in
the opposite direction from the truth). Concretely, verify:

1. **Directory hierarchy** — the actual nesting, not the nesting a
   specification implies (ACDC's real layout, found during MS11, nests by
   weather-condition **and** camera-sequence subdirectory — materially
   more complex than `MS6_ARCHITECTURE_SPEC.md` §3.3's own sketch assumed;
   found *before* any ACDC loader was written, exactly because this rule
   was followed).
2. **Counts** — declared (in whatever manifest/metadata file the format
   provides) vs. actual on-disk file counts, for every split/role the
   format distinguishes.
3. **Filename/id conventions** — the exact stem pattern, and whether
   image/label pairing is by identical stem, an index, or an explicit
   manifest field.
4. **Identifier mapping** — every declared pairing actually resolves to
   two real, present files.
5. **Metadata consistency** — does the format's own declared metadata
   (e.g. `dataset.json`'s `numTraining`) match reality?
6. **Per-example dimensions** — fixed, or variable (MSD Hippocampus is
   variable — 236 distinct shapes across 260 real cases; never assume a
   canonical size without checking).
7. **Spacing/resolution information**, if the modality has it (voxel
   spacing for 3-D medical volumes; DPI/pixel-size metadata rarely applies
   to natural images but may apply to some medical 2-D formats).
8. **Orientation** — for any format with a physical-space convention
   (NIfTI affines; any other coordinate-bearing header) — **added to this
   checklist by DI-1's own real-data pass**, which found a fact no
   previous milestone had checked or documented (all 260 real Hippocampus
   cases share a single, uniform orientation, `("R", "A", "S")`).
9. **Non-finite values (NaN/Inf)** in every array read, image and label
   independently — **and confirm this check runs on the array's native
   dtype, before any cast** (§4.1 below is the concrete cautionary tale).
10. **Missing files** — a manifest/metadata entry that does not resolve
    to a real file must be a documented failure mode (§4), not a runtime
    `FileNotFoundError` surfacing three layers away from the loader.

## 4 · Integrity Checks

Every concrete `Dataset` **should** expose a `verify_integrity() ->
IntegrityReport` method (the `IntegrityIssue`/`IntegrityReport` types,
`wfcrc/datasets/base.py`, are shared and reusable — see that module's own
docstring for why they are concrete, reusable value types rather than a
new abstract method on the frozen `Dataset` ABC). It should, at minimum,
check per example:

- Image readable (a clear, typed exception otherwise — reuse
  `SerializationError` for content problems, plain `ValueError` for
  structural/config problems, exactly the split `MSDNiftiLoader`'s own
  module docstring §6 already establishes; do not invent a new exception
  type for a new loader family without a genuine reason no existing type
  covers).
- Label readable.
- Image/label shape (or equivalent spatial-extent) agreement.
- Image finite (no NaN/Inf).
- Label finite (no NaN/Inf).
- Label values are a subset of whatever this dataset's own declared label
  vocabulary is (derived from the format's own metadata — e.g.
  `dataset.json`'s `"labels"` map for MSD — never hardcoded).

**Collect every issue found; do not raise on the first one.** This
mirrors `wfcrc.evaluation.verifier.Verifier`'s own established pattern
(run every check, aggregate, let the caller decide how strict to be via
an explicit `assert_ok()`) — a caller validating an entire real archive
needs the complete list of problems in one pass, not one exception per
re-run.

### 4.1 The cast-order cautionary tale (a real bug found and fixed in DI-1)

`MSDDataset.raw_labels(id_)` casts its NIfTI-native float array to
`int64` (the correct, intentional representation for a discrete label
volume). **A finiteness check performed *after* that cast is silently
meaningless**: `numpy` has no `int64` representation of NaN, so
`float('nan')` cast to `int64` becomes an arbitrary garbage integer, not
a value `np.isfinite` can flag — the corruption survives the cast
undetected. `MSDDataset.verify_integrity` was initially written against
`self.raw_labels()`'s *already-cast* output and found, via a genuinely
malformed-input unit test (not by inspection), that a NaN label was
silently passing the check. The fix: `verify_integrity()` reads both
image and label directly at their native float dtype (the same private
`_load_nifti_volume` helper the rest of the module already uses) and
checks finiteness **before** any format-specific type conversion. **Any
future loader's own integrity check must run its finiteness/NaN check on
the raw, native-precision array — never on a value that has already
passed through a lossy or undefined-on-NaN type cast.**

## 5 · Metadata Requirements

Every concrete `Dataset.meta()` must return
`wfcrc.datasets.metadata.DATASET_METADATA[<key>].to_dict()`, plus any
format-specific detail the base record doesn't carry (e.g. `MSDDataset`
adds `task`/`task_labels`) — never an inline, hand-written
`{"version": ..., "license": ...}` literal in the loader itself. Every
Phase-A dataset's `DATASET_METADATA` entry already exists (MS6.2); a new
loader consumes it, it does not create a competing copy.

## 6 · Future Compatibility Assessment (DI-1 Task 6)

Whether each remaining Phase-A dataset fits this architecture **without**
extending it:

| Dataset | Fits without architectural change? | Detail |
|---|---|---|
| **MSD Task07_Pancreas** | ✅ **Yes — zero architectural change.** | `MSDNiftiLoader`/`MSDDataset` are already task-generic (`_TASK_METADATA_KEYS`, `task_labels` read from `dataset.json`, `verify_integrity`'s label-vocabulary check already derived from `self._task_labels` rather than hardcoded `{0,1,2}`). Adding Pancreas is one `_TASK_METADATA_KEYS` dict entry (once its own `DATASET_METADATA` key and real-data validation, per §3's rules, are done) — not a redesign. This was already true before DI-1 and remains true after it. |
| **ACDC (driving)** | ⚠️ **New concrete loader required (expected, by design)** — but the *abstract* architecture (this document) needs no change. | A new `DatasetLoader`/`Dataset` pair for the Cityscapes-PNG format is required (different file format entirely — not a NIfTI variant). The real on-disk layout (MS11 finding: weather-condition + GoPro-sequence nesting) means this loader's discovery logic must be written against that real structure, not `MS6_ARCHITECTURE_SPEC.md` §3.3's own simpler sketch. The `split_manifest`-supplied, no-invented-ratio convention (§7 below) and the `IntegrityIssue`/`IntegrityReport` pattern both carry over directly. |
| **Kvasir-SEG** | ⚠️ **New concrete loader required, additionally blocked on an open methodological question.** | A new loader (JPG images + mask images) is required, same as ACDC. **Beyond that:** `docs/DATASET_SPLIT_POLICY.md` §8 item 1's split-unit question (is per-procedure grouping recoverable?) remains unresolved even with the real archive locally available (`docs/PRODUCTION_READINESS_AUDIT.md` Task 2) — filenames are opaque hashes with no visible grouping. This loader's `split_manifest` mechanism itself would work identically once ids are decided; **deciding which ids constitute a defensible split is the actual blocker**, not the architecture. |
| **CIFAR-10** | ⚠️ **New concrete loader required; a materially different internal shape, though the abstract contract is unaffected.** | CIFAR-10 examples are rows within a shared binary/pickle batch file, not one file per example — `MSDNiftiLoader`'s internal `_MSDCase(id_, image_path, label_path)` per-case-file-path pattern does not transfer as-is; a CIFAR loader's internal case representation is naturally an array index into an in-memory batch, not a pair of paths. The outer `Dataset`/`DatasetLoader` ABC contract (§2) is unaffected — `ids()` can be integer indices, `__iter__` can still yield deterministically. |
| **CIFAR-10.1** | ✅ **Fits the existing `split_manifest` mechanism without change**, once a CIFAR loader family exists. | `docs/DATASET_SPLIT_POLICY.md` §3.6's own policy is categorical (100% test, 0% train/calibration) — already exercised as a code path (`test_empty_split_has_zero_length`, this repository's own existing test) — no new split mechanism is needed, only the CIFAR-format loader itself (shared with CIFAR-10, above). |

**Net assessment:** the `Dataset`/`DatasetLoader`/`SplitManifest`
architecture itself requires **no interface extension** for any of the
five remaining datasets — every one of them fits within `Dataset`'s five
abstract methods and `DatasetLoader`'s one. What each new dataset
requires is a **new concrete loader family** (expected and already
anticipated by `MS6_ARCHITECTURE_SPEC.md` §3.3's own "four loader
families" framing) implementing this guide's §2–§5 rules against that
format's own real, verified structure. Kvasir-SEG additionally needs a
methodological decision (its split unit) before its loader's `split_manifest`
inputs can be defensibly chosen — a data-policy blocker, not an
architecture one.

## 7 · Split Assignment Convention (unchanged, restated for future loaders)

No frozen WFCRC document specifies a dataset-level split ratio anywhere
except `docs/DATASET_SPLIT_POLICY.md` §3's own per-dataset policy (a
document, not code). Every loader must therefore accept an **externally
supplied** split assignment (`MSDNiftiLoader`'s own `split_manifest`
constructor parameter is the established shape: a mapping or a path to a
JSON file with `{"train": [...], "calibration": [...], "test": [...]}`)
and must **never** invent, default, or silently choose a split ratio or
random seed itself. Validate the supplied assignment (every id resolves
to a real, discovered example; hand the three id lists unchanged to the
frozen `SplitManifest`, which enforces A1 hygiene) — do not reimplement
that disjointness check.

## 8 · Testing Requirements

Every new loader's test suite (synthetic-fixture-only for the default
suite, per `MS6_ARCHITECTURE_SPEC.md` §8.3's frozen Q3 policy — real data
is opt-in, `@pytest.mark.real_data`, and skips cleanly if absent) must
cover, at minimum, everything `tests/unit/datasets/loaders/test_msd.py`
already demonstrates for MSD (now 57 tests, DI-1-updated):

- **Normal loading** — every split loads; `len()`/`ids()` agree; iteration
  yields exactly `len()` triples.
- **Missing files** — a manifest entry pointing at a nonexistent file
  raises the documented exception (`SerializationError` for content,
  `ValueError` for structural problems), both at discovery time (fail
  fast) and, separately, for a file removed *after* discovery but before
  first lazy access.
- **Malformed files** — an unparsable/corrupt file raises cleanly, not a
  silent empty/garbage result.
- **Duplicate ids** — both within the format's own manifest/metadata file
  (already covered for MSD's `dataset.json`) **and** within a single
  caller-supplied split's own id list (the DI-1 addition, `MSDDataset.__init__`'s
  new check) — these are two distinct failure modes, both must be tested
  independently, since a fix for one does not imply the other is covered.
- **Invalid metadata** — a malformed or incomplete split manifest, an
  unrecognized split name, an id outside the discovered pool.
- **Iterator behaviour** — deterministic order across repeated calls;
  correct behavior on an empty split (`len() == 0`, `list(dataset) == []`).
- **Random (per-id) access** — every additive per-id accessor
  (`image`/`labels`/`raw_labels`/`spacing`/`orientation`/whatever a new
  format's own accessors are) tested independently for a known id, an
  unknown id (`ValueError`), and a missing/malformed underlying file
  (`SerializationError`).
- **Dataset length** — `len()` matches `ids()`'s own length, always.
- **Reproducibility** — identical loader construction + identical split
  assignment reproduces identical `ids()`/iteration order, every time.
- **Integrity checking** — a clean fixture reports `ok=True`; each
  distinct corruption mode (shape mismatch, non-finite image, non-finite
  label, out-of-vocabulary label value, unreadable image, unreadable
  label) reports `ok=False` with a matching, findable issue — tested
  independently per corruption mode, not only in combination.
- **Preprocessing compatibility** — the loader's own native-resolution
  output must feed cleanly into whatever frozen `wfcrc.datasets.preprocessing`
  function applies to this modality, without the loader itself performing
  that preprocessing (`MSDNiftiLoader`'s own §5 reasoning: no unfrozen
  target spacing/normalization scheme may be invented by a loader).

**Real-data validation** (opt-in, marker-gated) should, once real data is
acquired for a new dataset, independently re-verify every claim in §3
against the actual archive — not reuse a synthetic fixture's own
assumptions as if they were verified facts about the real data.

---

## Connections

`wfcrc/datasets/base.py` · `wfcrc/datasets/loaders/msd.py` ·
`wfcrc/datasets/metadata.py` · `wfcrc/datasets/registry.py` ·
`tests/unit/datasets/loaders/test_msd.py` ·
`docs/DATASET_SPLIT_POLICY.md` · `docs/PRODUCTION_READINESS_AUDIT.md` ·
`docs/MODEL_POLICY.md` · `MS6_ARCHITECTURE_SPEC.md` §3.3 ·
`PROJECT_CONTEXT.md`
