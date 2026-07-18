"""``MSDNiftiLoader`` / ``MSDDataset`` — the MSD/NIfTI concrete dataset loader (MS6.3A, §3.3).

First real-data target: **MSD Task04_Hippocampus**. Per the MS6.3 task brief,
this pass implements and validates the MSD/NIfTI loader path only —
Cityscapes-format (+ ACDC + Cityscapes-C), CIFAR, and Kvasir are explicitly
out of scope here (MS6_ARCHITECTURE_SPEC.md §3.3 names all four families;
only this one is built in this pass).

**1 · Authoritative MSD Task04_Hippocampus structure (verified against
upstream, not assumed).** Before writing this loader, the real dataset was
inspected via its official AWS Open Data mirror (the Dataset Selection
Audit's own acquisition note: "medicaldecathlon.com itself returned a TLS
certificate mismatch ... license was instead confirmed via the AWS Open Data
registry mirror" — the same TLS issue reproduced when independently
re-verified here, so the same mirror was used):

- **Archive.** A single object, ``Task04_Hippocampus.tar``, in the public
  ``s3://msd-for-monai`` bucket (``us-west-2``; a Registry-of-Open-Data
  dataset, ARN ``arn:aws:s3:::msd-for-monai``). Directly confirmed via an
  unsigned HTTPS HEAD request in this session: ``Content-Length: 28425216``
  bytes (~27.1 MiB), ``Last-Modified: Fri, 14 Aug 2020``,
  ``ETag: "81aa748a1df6d95ecb6f1b5812f132b1-4"`` (an S3 multipart-upload
  ETag — the ``-4`` suffix means it is a hash-of-part-hashes, *not* a plain
  MD5 of the file; it cannot be recomputed with a local ``md5sum`` for
  independent verification, only compared against itself across
  re-downloads from the same bucket). No independently-published MD5/SHA256
  file checksum from the Decathlon organizers themselves was found anywhere
  upstream — this ETag is the only integrity signal available, recorded
  honestly rather than presented as a formal checksum.
- **Extracted directory structure** (confirmed via the original Decathlon
  paper (Antonelli et al., arXiv:1902.09063) plus independent corroboration
  from the dataset's own HuggingFace/MONAI mirrors — file *contents* were
  not fetched, since downloading real data is outside this pass's authority
  without user go-ahead; see §7/Task 8 below):
  ```
  Task04_Hippocampus/
    dataset.json
    imagesTr/hippocampus_XXX.nii.gz     # 260 files, XXX = case number
    labelsTr/hippocampus_XXX.nii.gz     # 260 files, same stem as imagesTr
    imagesTs/hippocampus_XXX.nii.gz     # 130 files, no labelsTs (unlabeled)
  ```
  Critically, image/label filenames are **identical stems** in both
  directories (e.g. ``hippocampus_367.nii.gz`` in both ``imagesTr`` and
  ``labelsTr``) — this is the *original* Decathlon distribution format, not
  the ``_0000``-suffixed convention nnU-Net's own converter produces after
  ``nnUNet_convert_decathlon_task``; this loader targets the former (what
  the AWS mirror actually serves), not the latter.
- **Case counts — real-data validated.** The real, locally-acquired archive's
  own ``dataset.json`` declares (and the on-disk file counts confirm exactly)
  ``numTraining: 260``, ``numTest: 130`` — 390 total. The MS6.3A implementation
  pass (before real data was available) had instead recorded "263 + 131 = 394"
  here, sourced from secondary web mirrors/documentation rather than the
  actual archive; that figure is **superseded and was wrong** — this is the
  corrected, ground-truth count, confirmed by opening the real ``dataset.json``
  post-acquisition. It matches the Dataset Selection Audit's own informal
  "~260" planning estimate far more closely than the earlier, incorrect
  web-sourced figure did. Either way, nothing in ``MSDNiftiLoader``'s
  interface or logic hardcodes a case count anywhere — it is always derived
  from ``dataset.json``'s own ``training``/``test`` lists at discovery time,
  so this correction changes documentation only, not behavior.
- **``dataset.json`` schema** (the standard MSD schema, identical in shape
  across all ten tasks): ``name``, ``description``, ``reference``,
  ``licence``, ``tensorImageSize`` (``"3D"``), ``modality``
  (``{"0": "MRI"}`` — single-modality, so no channel axis), ``labels``
  (``{"0": "background", "1": "Anterior", "2": "Posterior"}``),
  ``numTraining``, ``numTest``, ``training`` (a list of
  ``{"image": "./imagesTr/hippocampus_XXX.nii.gz", "label":
  "./labelsTr/hippocampus_XXX.nii.gz"}`` objects — the authoritative
  image/label pairing source this loader uses), and ``test`` (a list of
  bare ``"./imagesTs/hippocampus_XXX.nii.gz"`` image paths, **no labels** —
  per Task 3 below, this loader never exposes these as a loadable split).
  **Real-data note:** the actual archive's top-level release-identifier key
  is spelled ``"relase"`` (a typo in the upstream file itself, confirmed
  present, not ``"release"`` as pre-acquisition documentation here assumed)
  — harmless, since this loader never reads that key at all (only
  ``training``/``test``/``labels`` are consumed).
- **NIfTI dimensionality.** Images and labels are both 3-D single-channel
  volumes (``(D, H, W)``, no time/channel axis) — labels are voxel-wise
  integer class indices (``0``/``1``/``2``), not one-hot. **Real-data
  validated:** per-case spatial shape varies considerably (236 distinct
  shapes across the 260 real cases, ranging roughly `30-43` voxels per axis)
  — each volume is a tight per-case bounding-box crop around the hippocampus,
  not a fixed canonical size, exactly as this loader already assumes (it
  never asserts a fixed shape).
- **Voxel spacing — real-data validated, corrects a pre-acquisition
  assumption.** All 260 real cases have **identical** voxel spacing,
  ``(1.0, 1.0, 1.0)`` mm (already-isotropic-resampled by the Decathlon
  organizers before release) — not "not uniform across cases" as this
  section speculated before real data was available. That earlier claim is
  superseded. This does not change the loader's design: no target spacing is
  frozen anywhere regardless, :meth:`MSDDataset.spacing` still reports each
  case's own header value rather than assuming a constant, and §5's
  reasoning for not baking resampling into the loader holds independently
  (confirmed directly on real data, §5 below) — the uniformity found here
  is a property of *this* dataset, not a property :func:`resample_volume`
  or this loader may assume in general (e.g. it need not hold for
  Task07_Pancreas).

None of the above required any redesign of ``MSDNiftiLoader``'s public
interface as sketched in MS6_ARCHITECTURE_SPEC.md §3.3 — every difference
found (exact case count, exact archive byte size, the multipart-ETag
caveat, the ``"relase"`` typo, per-case shape variability, spacing
uniformity) is an implementation/documentation detail the loader already
treats generically (derived from ``dataset.json``/each file's own header at
runtime, never hardcoded), not a contradiction requiring a STOP. One
genuine **gap**, not a contradiction, was found and is resolved below (§2).

**2 · Constructor signature — a disclosed, task-authorized gap-fill.**
§3.3's own interface sketch shows ``MSDNiftiLoader.__init__(root_dir, task)``
with no way to pass split-assignment information at all, yet the same
section requires the loader to "construct and validate a ``SplitManifest``
... from the ids assigned to each split **at construction time**." The
sketch does not say where those ids come from — an internal gap in the
frozen document itself, not something this pass invented an answer to
silently: the MS6.3 task brief's own Task 3 explicitly anticipates and
resolves it ("you may implement the mechanical ability to consume an
explicitly supplied split manifest or split-definition file, but do not
choose the research split proportions or seed without authorization").
Accordingly, ``__init__`` takes one additional, required, keyword-only
parameter, ``split_manifest``, beyond the sketch's ``root_dir``/``task`` —
purely mechanical, additive, and never populated with an invented
proportion or seed (§3 below). ``.load(split_name)``'s signature is
unchanged from the sketch.

**3 · Split policy — no proportions or seed invented (Task 3).** No frozen
WFCRC document anywhere specifies how the 263 labelled Hippocampus cases
should divide into base-model-train / calibration / test (confirmed by an
explicit cross-document search of the Experiment Blueprint, Algorithm
Specification, MS2/MS4 Implementation Specs, and the Experiment Environment
Audit — the only frozen split-related figure anywhere, Experiment Blueprint
§18's "split ratio π ∈ {0.2, 0.3, 0.5}", is WFCRC's own *internal*
calibration-pool A/B split (Algorithm Specification §17's single-split
theorem), a completely different split from the one this loader assigns).
Per the task brief's explicit instruction, this loader does **not** invent
one. ``split_manifest`` must be supplied by the caller as either:

- a mapping ``{"train": [...ids...], "calibration": [...ids...], "test":
  [...ids...]}`` (every id a case id this loader discovers from
  ``dataset.json``'s ``training`` list), or
- a path to a JSON file with that exact shape.

The loader validates (never invents) this externally-supplied assignment:
every id must resolve to a real, discovered, labelled training case (never
one of the 131 unlabelled challenge ``test`` entries — Task 3's "do not use
unlabelled challenge test cases for loss evaluation" holds by construction,
since those ids are never even in the discoverable id pool this loader
exposes); the three id lists are handed unchanged to the frozen
:class:`~wfcrc.datasets.base.SplitManifest`, so disjointness (A1 hygiene) is
enforced by the existing, unmodified gate
(:func:`~wfcrc.datasets.base.assert_split_disjoint`), not reimplemented.
The **official MSD challenge train/test division** (260 labelled vs. 130
unlabelled — real-data validated, see §1) and **WFCRC's own train/
calibration/test partition** of the 260 labelled cases are two distinct
concepts, kept distinct here: this loader's "training pool" is exactly the
260 labelled cases (MSD's own "training" list); WFCRC's train/calibration/
test are a caller-supplied re-partition of that same pool, via
``split_manifest``, with no default.

**4 · ``labels()`` binarization — a frozen-contract requirement, not a
choice (disclosed).** :meth:`wfcrc.losses.base.LossEvaluator._validate_shapes`
requires ``label.dtype == bool`` — a hard requirement of the already-frozen
calibration/evaluation core that every concrete ``Dataset.labels()``
implementation must satisfy, MSD included. Hippocampus's raw label volumes
are 3-valued (``0``=background, ``1``=Anterior, ``2``=Posterior);
:meth:`MSDDataset.labels` therefore returns the single binary foreground
mask ``raw_label > 0`` (any hippocampus tissue, either substructure) — the
standard, unambiguous "segment the organ" reading, not a per-structure
split. This does **not** discard the Anterior/Posterior distinction: it
remains available, unbinarized, via :meth:`MSDDataset.raw_labels`, an
additive method beyond the frozen ``Dataset`` ABC's minimum contract,
specifically so a future Group Mask Builder (MS6.7, not yet built, and
explicitly out of scope for MS6.3A) can still derive the anterior/posterior
two-group conditional the Dataset Selection Audit names, without needing to
reparse any NIfTI file. Deciding *how* MS6.7 will consume it is left to
that milestone, per the "do not invent behavior for a later milestone"
rule.

**5 · Preprocessing — no forced resample/normalize, by design (disclosed
deviation from §3.3's literal text).** §3.3 says a loader should "apply
Dataset Preprocessing transforms inside ``__iter__``/``labels()``" — but no
frozen document specifies a target voxel spacing or an intensity
normalization scheme for Hippocampus (no model/config exists yet to imply
one; that is Score-Provider-specific information, MS6.4, not yet built).
Forcing an arbitrary target spacing now would mean inventing an unfrozen
hyperparameter. A second, independent reason applies specifically to
resampling: the frozen :func:`wfcrc.datasets.preprocessing.resample_volume`
performs linear interpolation (correct for continuous MRI intensity, but it
would corrupt a discrete-valued label volume — e.g. blending class ``1``
and class ``2`` at a boundary voxel into a meaningless ``1.5``). Rather than
silently reinterpreting the frozen preprocessing module's contract for
label data it was not written for, or picking an unfrozen target spacing,
``MSDDataset.__iter__``/``.labels()``/``.raw_labels()`` return volumes at
their **native, per-case resolution and intensity scale**, and
:meth:`MSDDataset.spacing` exposes each case's own NIfTI-header voxel
spacing so a caller *can* invoke ``resample_volume`` explicitly once a
target spacing is actually decided. "Preprocessing compatibility" (the
MS6.3 test brief's own phrase) is demonstrated by a dedicated unit test
that feeds this loader's own image output and spacing directly into the
frozen, unmodified ``resample_volume`` — proving the interop contract
without this loader prescribing a resampling policy nothing upstream has
frozen yet. No preprocessing logic is duplicated: the loader calls no
resampling code at all until a caller asks for it externally.

**6 · Failure modes — reusing frozen/sanctioned exception types only, no
new exception added (per the MS6 Architecture Specification's own §6 test
plan for MS6.3, which names exactly ``SerializationError``/"loader-specific
``ValueError``" as the expected failure types, and ``SplitLeakageError`` for
split overlap).**
:class:`~wfcrc.exceptions.SerializationError` (reused, not modified) is
raised for anything that is a *content* problem — a missing image/label
file, an unreadable/corrupt/truncated NIfTI file, a non-3-D volume, a
missing or unparsable ``dataset.json``, or a malformed ``training`` entry.
A plain ``ValueError`` (the same type the frozen ``DatasetLoader.load()``
ABC's own docstring already sanctions for "``split_name`` not recognized")
is raised for *structural/config* problems — an unsupported ``task``, a
nonexistent ``root_dir``/task directory, a duplicate case id inside
``dataset.json``, a malformed or incomplete ``split_manifest``, or a
``split_manifest`` id that does not resolve to a discovered case.
:class:`~wfcrc.exceptions.SplitLeakageError` (frozen, MS4) is raised
unchanged, by the existing :class:`~wfcrc.datasets.base.SplitManifest`
gate, for an overlapping ``split_manifest``.

**7 · Real-data acquisition.** See the MS6.3A implementation conversation's
FINAL REPORT for the full acquisition guide (source, archive, size,
extraction, expected tree, destination path) — summarized here for anyone
reading only this module: download
``https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar``
(28,425,216 bytes) and extract it, producing a ``Task04_Hippocampus/``
directory with the structure in §1 — this repository never downloads it
automatically. The Dataset Selection Audit's own documented cache
convention (:data:`wfcrc.datasets.metadata.DATASET_METADATA`
``["msd_hippocampus"].extra["repo_cache_dir"]``) is ``data/msd/
Task04_Hippocampus/``; this loader itself is location-agnostic (``root_dir``
is always caller-supplied), so any extraction location works as long as
``root_dir/task/`` points at the directory containing ``dataset.json``.

**8 · Real-data validation record (MS6.3A validation pass).** Every claim in
§1-§7 above has now been exercised against the real, locally-acquired
archive (260 labelled cases, all 260 read/verified: correct pairing, stable
ids, correct binarization, correct ``raw_labels()``/``spacing()``, zero
NaN/Inf/shape-mismatch/unreadable cases, zero duplicate ids, label value
union exactly ``{0, 1, 2}``). Two extraction-time artifacts were observed
and confirmed harmless: (a) the extracted tree also contains macOS
AppleDouble companion files (``._*``) and ``.DS_Store`` — this loader never
globs ``imagesTr``/``labelsTr``/``imagesTs`` directly, only the paths
``dataset.json`` itself lists, so these are never discovered or read; (b)
resampling a real label volume to a genuinely different target spacing
(source ``(1.0, 1.0, 1.0)`` mm to a synthetic ``(1.4, 1.4, 1.4)`` mm target)
produced non-integer values in 3.8% of voxels — direct, real-data
confirmation of the label-corruption risk §5 cites as the reason this
loader does not auto-resample labels. See the validation session's own
final report for the full record (performance timings, visual-inspection
renders, split-manifest mechanism checks).

**9 · DI-1 real-data re-validation and production-quality additions.**
Per an explicit "establish the production DatasetLoader standard, using
Hippocampus as the reference implementation" instruction, every claim in
§1-§8 above was independently re-exercised against the real, locally-
acquired archive (all 260 real cases, fresh inspection, not reused from
memory of the MS6.3A pass): declared counts (260 training / 130 test)
match on-disk file counts exactly; zero duplicate ids, missing files,
shape mismatches, or non-finite (NaN/Inf) values in any image or label;
label values exactly ``{0, 1, 2}`` across all 260 cases; voxel spacing
uniformly ``(1.0, 1.0, 1.0)`` mm (confirms MS6.3A). **One genuinely new
fact, not previously checked or documented anywhere in this codebase:**
every one of the 260 real cases shares a single, uniform NIfTI
orientation, ``("R", "A", "S")`` (right-anterior-superior), confirmed via
``nibabel.aff2axcodes`` — now exposed via the new
:meth:`MSDDataset.orientation` method (below), the same way
:meth:`spacing` already exposes each case's own header value rather than
assuming it. Real per-axis shape ranges, also newly recorded precisely
(previously only "30-43 voxels per axis" in prose): ``D`` (first axis)
31-43, ``H`` 40-59, ``W`` 24-47, across 236 distinct shapes.

Three real, previously-disclosed-as-gaps issues were closed, all
additively (no existing method signature changed, no existing test
behavior changed):

- **No direct per-id image accessor existed** (`wfcrc.models.scores.
  hippocampus_segmenter`'s own module docstring already named this gap,
  "Where the raw image comes from," and worked around it by consuming
  `__iter__` once at construction) — closed by the new
  :meth:`MSDDataset.image` method, mirroring :meth:`labels`'s existing
  shape exactly. `__iter__` now composes it instead of duplicating the
  NIfTI-read call.
- **A within-split duplicate id was silently tolerated** — if a caller's
  `split_manifest` listed the same case id twice within one split (never
  across splits, which the frozen A1 gate already catches),
  `MSDDataset.__init__`'s own `dict`-comprehension lookup table silently
  kept only the last entry, leaving `len(dataset)` (from the untouched
  `tuple`) larger than the number of *distinct* examples actually
  reachable via `ids()`/`_by_id`. `__init__` now raises `ValueError`
  naming the duplicate(s) — a real, previously-reachable gap, not a
  hypothetical one, and exactly the DI-1 task brief's own "duplicate
  identifiers are rejected" requirement, generalized beyond the
  already-covered "duplicate within `dataset.json`'s own training list"
  case (`_discover_cases`, unchanged) to "duplicate within a caller-
  supplied split assignment."
- **No integrity-checking method existed** beyond MS6.3A's own one-off,
  external validation script — closed by the new
  :meth:`MSDDataset.verify_integrity`, an additive, concrete method (not
  a new abstract requirement on the frozen `Dataset` ABC — see
  `wfcrc/datasets/base.py`'s own `IntegrityReport` docstring for why) that
  any caller can now run mechanically, inside this repository, without a
  bespoke ad hoc script, and that a future loader family may adopt in the
  same shape (`docs/DATASET_INTEGRATION_GUIDE.md`).

No existing public method's signature, return type, or documented
behavior changed. `MSDNiftiLoader`'s own constructor/`.load()` interface
is completely unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from wfcrc.datasets.base import (
    Dataset,
    DatasetLoader,
    IntegrityIssue,
    IntegrityReport,
    SplitManifest,
)
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.exceptions import SerializationError

__all__ = ["MSDDataset", "MSDNiftiLoader"]

#: MSD task name -> `wfcrc.datasets.metadata.DATASET_METADATA` key. MS6.3A
#: implements and real-data-validates only `"Task04_Hippocampus"` (module
#: docstring §1); the discovery/pairing/split machinery is already
#: task-format-generic (MS6 Architecture Specification §3.3's own
#: requirement to support `Task07_Pancreas` later "without introducing a
#: second loader architecture") — adding Pancreas later is one dict entry,
#: not a redesign, once its own real-data integration is authorized.
_TASK_METADATA_KEYS: dict[str, str] = {
    "Task04_Hippocampus": "msd_hippocampus",
}

#: The three split names this loader (and the frozen `SplitManifest`) recognizes.
_SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "test")

#: `SplitManifest` field name for each split name above.
_MANIFEST_FIELD: dict[str, str] = {
    "train": "train_ids",
    "calibration": "cal_ids",
    "test": "test_ids",
}


@dataclass(frozen=True)
class _MSDCase:
    """One discovered, paired MSD labelled training case."""

    id_: str
    image_path: Path
    label_path: Path


def _strip_nifti_suffix(filename: str) -> str:
    """Return `filename` without its `.nii.gz`/`.nii` extension.

    Args:
        filename: A bare filename (no directory components).

    Returns:
        `filename` with the NIfTI extension removed.

    Raises:
        SerializationError: If `filename` has neither extension.
    """
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    if filename.endswith(".nii"):
        return filename[: -len(".nii")]
    raise SerializationError(f"expected a .nii or .nii.gz filename, got {filename!r}")


def _resolve_relative(task_dir: Path, rel: str) -> Path:
    """Resolve a `dataset.json`-style relative path (e.g. `"./imagesTr/x.nii.gz"`).

    Args:
        task_dir: The MSD task's root directory.
        rel: A path as written in `dataset.json` (`"./..."` or bare).

    Returns:
        The resolved absolute :class:`~pathlib.Path`.
    """
    cleaned = rel[2:] if rel.startswith("./") else rel
    return task_dir / cleaned


def _load_nifti_volume(path: Path) -> tuple[NDArray[np.float64], tuple[float, float, float]]:
    """Read a 3-D NIfTI volume and its voxel spacing.

    Args:
        path: Path to a `.nii`/`.nii.gz` file.

    Returns:
        `(array, spacing)`: a `float64` `(D, H, W)` array and its
        `(sz, sy, sx)` voxel spacing from the NIfTI header.

    Raises:
        SerializationError: If `path` does not exist, cannot be parsed as
            NIfTI, or is not a 3-D volume.
    """
    if not path.is_file():
        raise SerializationError(f"referenced NIfTI file does not exist: {path}")
    try:
        image = nib.load(str(path))
        # nib.load()'s declared return type is the abstract FileBasedImage
        # base, which mypy sees as lacking `.dataobj`/`.header.get_zooms()`
        # — every concrete image nibabel actually returns (Nifti1Image for
        # both .nii and .nii.gz) has both at runtime.
        array = np.asarray(image.dataobj, dtype=np.float64)  # type: ignore[attr-defined]
        zooms = image.header.get_zooms()  # type: ignore[attr-defined]
    except Exception as exc:  # nibabel raises varied types for corrupt/truncated input
        raise SerializationError(f"could not read NIfTI file {path}: {exc}") from exc
    if array.ndim != 3:
        raise SerializationError(f"expected a 3-D NIfTI volume at {path}, got ndim={array.ndim}")
    # nibabel's get_zooms() always returns exactly one entry per array
    # dimension, so ndim == 3 (just checked) guarantees len(zooms) == 3.
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return array, spacing


def _load_nifti_affine(path: Path) -> NDArray[np.float64]:
    """Read a NIfTI file's affine transform (header only, no full array read).

    DI-1 addition, additive: backs :meth:`MSDDataset.orientation`. Kept as a
    separate helper from :func:`_load_nifti_volume` (rather than extending
    that function's return shape) so every existing caller of
    `_load_nifti_volume` is completely unaffected.

    Args:
        path: Path to a `.nii`/`.nii.gz` file.

    Returns:
        The `(4, 4)` affine transform matrix.

    Raises:
        SerializationError: If `path` does not exist or cannot be parsed.
    """
    if not path.is_file():
        raise SerializationError(f"referenced NIfTI file does not exist: {path}")
    try:
        image = nib.load(str(path))
        affine = np.asarray(image.affine, dtype=np.float64)  # type: ignore[attr-defined]
    except Exception as exc:
        raise SerializationError(f"could not read NIfTI file {path}: {exc}") from exc
    return affine


class MSDDataset(Dataset):
    """A loaded MSD/NIfTI split: lazily reads image/label volumes on access.

    See :mod:`wfcrc.datasets.loaders.msd`'s module docstring (§4, §5) for
    why :meth:`labels` returns a binarized foreground mask and why no
    resampling/normalization is applied here.
    """

    def __init__(
        self,
        cases: Sequence[_MSDCase],
        *,
        metadata_key: str,
        task: str,
        task_labels: Mapping[str, Any],
    ) -> None:
        """Construct a split from its already-selected, already-paired cases.

        Args:
            cases: The cases assigned to this split, in stable order.
            metadata_key: The :data:`~wfcrc.datasets.metadata.DATASET_METADATA` key.
            task: The MSD task name (e.g. `"Task04_Hippocampus"`).
            task_labels: `dataset.json`'s own `"labels"` map (e.g.
                `{"0": "background", "1": "Anterior", "2": "Posterior"}`),
                surfaced via :meth:`meta`.

        Raises:
            ValueError: If `cases` contains two entries sharing the same
                `id_` (DI-1 addition: a within-split duplicate id was
                previously silently collapsed to one entry by the
                `dict`-comprehension lookup table below, without
                `__len__`/`ids()` agreeing about how many examples this
                split actually has — see `docs/DATASET_INTEGRATION_GUIDE.md`
                for the full disclosure).
        """
        self._cases: tuple[_MSDCase, ...] = tuple(cases)
        ids_seen = [c.id_ for c in self._cases]
        if len(ids_seen) != len(set(ids_seen)):
            duplicates = sorted({str(i) for i in ids_seen if ids_seen.count(i) > 1})
            raise ValueError(f"duplicate id(s) within this split: {duplicates}")
        self._by_id: dict[Hashable, _MSDCase] = {c.id_: c for c in self._cases}
        self._metadata_key = metadata_key
        self._task = task
        self._task_labels = dict(task_labels)

    def _case_for(self, id_: Hashable) -> _MSDCase:
        """Look up the case for `id_`, or raise if it is not in this split."""
        case = self._by_id.get(id_)
        if case is None:
            raise ValueError(f"unknown id {id_!r} for this split")
        return case

    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        """Yield `(id, image, label)` triples, image/label at native resolution."""
        for case in self._cases:
            yield case.id_, self.image(case.id_), self.labels(case.id_)

    def __len__(self) -> int:
        return len(self._cases)

    def ids(self) -> Sequence[Hashable]:
        return tuple(c.id_ for c in self._cases)

    def image(self, id_: Hashable) -> NDArray[np.float64]:
        """Return the native-resolution image volume for `id_`.

        DI-1 addition, additive: direct per-id image access, mirroring
        :meth:`labels`/:meth:`raw_labels`/:meth:`spacing`'s existing shape.
        Closes a previously-disclosed gap (`wfcrc.models.scores.
        hippocampus_segmenter`'s own module docstring, "Where the raw
        image comes from"): a `ScoreProvider` no longer needs to consume
        `__iter__` once to build its own internal `id -> image` lookup —
        it may call this directly instead. `__iter__` itself now composes
        this method rather than duplicating the NIfTI-read call.
        """
        image, _ = _load_nifti_volume(self._case_for(id_).image_path)
        return image

    def labels(self, id_: Hashable) -> NDArray[np.bool_]:
        """Return the binarized foreground mask (`raw_label > 0`) for `id_`.

        See the module docstring §4 for why this is binarized rather than
        multi-class, and :meth:`raw_labels` for the unbinarized volume.
        """
        raw, _ = _load_nifti_volume(self._case_for(id_).label_path)
        return raw > 0.0

    def raw_labels(self, id_: Hashable) -> NDArray[np.int64]:
        """Return the un-binarized, multi-class MSD label volume for `id_`.

        Additive beyond the frozen :class:`~wfcrc.datasets.base.Dataset`
        contract (see module docstring §4): preserves per-structure class
        indices that :meth:`labels` collapses into a single foreground mask.
        """
        raw, _ = _load_nifti_volume(self._case_for(id_).label_path)
        return raw.astype(np.int64)

    def spacing(self, id_: Hashable) -> tuple[float, float, float]:
        """Return `id_`'s image's native `(sz, sy, sx)` voxel spacing.

        Additive beyond the frozen :class:`~wfcrc.datasets.base.Dataset`
        contract; see module docstring §5 for why this loader does not
        resample volumes itself.
        """
        _, spacing = _load_nifti_volume(self._case_for(id_).image_path)
        return spacing

    def orientation(self, id_: Hashable) -> tuple[str, str, str]:
        """Return `id_`'s image's NIfTI orientation axis codes (e.g. `("R", "A", "S")`).

        DI-1 addition, additive: real-data inspection of all 260 real
        MSD Task04_Hippocampus cases (`docs/DATASET_INTEGRATION_GUIDE.md`)
        found a single, uniform orientation (`"R", "A", "S"`) across every
        case — not previously checked or documented anywhere in this
        codebase. Exposed here the same way :meth:`spacing` exposes each
        case's own header value, rather than assumed constant, since
        nothing guarantees a future dataset (or a future Hippocampus
        release) shares this uniformity.

        Returns:
            The 3-letter axis-code tuple from `nibabel.aff2axcodes` on this
            case's image affine (e.g. `("R", "A", "S")`).
        """
        affine = _load_nifti_affine(self._case_for(id_).image_path)
        codes = nib.aff2axcodes(affine)  # type: ignore[no-untyped-call]
        return (str(codes[0]), str(codes[1]), str(codes[2]))

    def verify_integrity(self) -> IntegrityReport:
        """Check every case in this split for content-level integrity problems.

        DI-1 addition, additive: closes the "integrity checking" gap named
        in the DI-1 task brief. Checks, per case: image/label readability,
        image/label shape agreement, image/label finiteness (no NaN/Inf),
        and that every label voxel value is one this split's own
        `dataset.json`-declared label map actually names (e.g. only
        `{0, 1, 2}` for Hippocampus's `{"0": ..., "1": ..., "2": ...}` map)
        — a garbage/corrupted label file with out-of-range values is
        caught here even though it would still "read" successfully as a
        NIfTI file. Mirrors `wfcrc.evaluation.verifier.Verifier`'s own
        "run every check, collect results, let the caller decide how
        strict to be" pattern (`IntegrityReport`, `wfcrc/datasets/base.py`)
        rather than raising on the first problem found — a caller
        validating an entire real archive wants the complete list of
        problems, not just the first one.

        Returns:
            An `IntegrityReport` listing every problem found (empty if the
            split is fully intact).
        """
        allowed_label_values = {int(k) for k in self._task_labels}
        issues: list[IntegrityIssue] = []
        for case in self._cases:
            try:
                image, _ = _load_nifti_volume(case.image_path)
            except SerializationError as exc:
                issues.append(IntegrityIssue(case.id_, f"image unreadable: {exc}"))
                continue
            try:
                # Read the label at its native float dtype, *not* via
                # `self.raw_labels()` (which casts to `int64`): a NaN cast
                # to `int64` silently becomes an arbitrary garbage integer
                # (no `int64` NaN representation exists), which would make
                # a finiteness check downstream of that cast never able to
                # detect the very corruption it exists to catch. Checking
                # the pre-cast float array is what makes this check real.
                raw_label, _ = _load_nifti_volume(case.label_path)
            except SerializationError as exc:
                issues.append(IntegrityIssue(case.id_, f"label unreadable: {exc}"))
                continue

            if image.shape != raw_label.shape:
                issues.append(
                    IntegrityIssue(
                        case.id_,
                        f"image/label shape mismatch: {image.shape} vs {raw_label.shape}",
                    )
                )
            if not np.all(np.isfinite(image)):
                issues.append(IntegrityIssue(case.id_, "image contains NaN/Inf"))
            if not np.all(np.isfinite(raw_label)):
                issues.append(IntegrityIssue(case.id_, "label contains NaN/Inf"))
            elif allowed_label_values:
                found = {int(v) for v in np.unique(raw_label)}
                unexpected = sorted(found - allowed_label_values)
                if unexpected:
                    issues.append(
                        IntegrityIssue(
                            case.id_,
                            f"label contains value(s) outside the declared label map "
                            f"{sorted(allowed_label_values)}: {unexpected}",
                        )
                    )
        return IntegrityReport(issues=tuple(issues))

    def meta(self) -> dict[str, Any]:
        """Return :data:`~wfcrc.datasets.metadata.DATASET_METADATA`'s record, plus task info."""
        record = DATASET_METADATA[self._metadata_key].to_dict()
        record["task"] = self._task
        record["task_labels"] = dict(self._task_labels)
        return record


class MSDNiftiLoader(DatasetLoader):
    """Concrete :class:`~wfcrc.datasets.base.DatasetLoader` for the MSD/NIfTI format.

    See :mod:`wfcrc.datasets.loaders.msd`'s module docstring for the full
    design record (real-dataset structure, the constructor-signature
    gap-fill, split policy, binarization, and preprocessing decisions).
    """

    def __init__(
        self,
        root_dir: str | Path,
        task: str,
        *,
        split_manifest: Mapping[str, Sequence[Hashable]] | str | Path,
    ) -> None:
        """Discover cases and validate a caller-supplied split assignment.

        Args:
            root_dir: Directory containing `<task>/` (e.g. `"data/msd"`).
            task: The MSD task subdirectory name (currently only
                `"Task04_Hippocampus"` is supported; see module docstring §1).
            split_manifest: Either a mapping `{"train": [...], "calibration":
                [...], "test": [...]}` of case ids, or a path to a JSON file
                with that exact shape. Never defaulted or inferred — see
                module docstring §3: this loader does not choose a split
                policy.

        Raises:
            ValueError: If `task` is not supported, `root_dir/task` is not a
                directory, `dataset.json`'s `training` list contains a
                duplicate case id, or `split_manifest` is malformed
                (wrong shape, missing/unrecognized split names, or
                references an id outside the discovered training pool).
            SerializationError: If `dataset.json` (or the `split_manifest`
                file, if a path was given) is missing or unparsable, or a
                referenced image/label file is missing.
            SplitLeakageError: If `split_manifest`'s three id lists are not
                pairwise disjoint (raised by the frozen
                :class:`~wfcrc.datasets.base.SplitManifest`).
        """
        if task not in _TASK_METADATA_KEYS:
            raise ValueError(
                f"unsupported MSD task {task!r}; this pass (MS6.3A) implements and "
                f"registers only {sorted(_TASK_METADATA_KEYS)} — see module docstring §1"
            )
        self._task = task
        self._metadata_key = _TASK_METADATA_KEYS[task]

        task_dir = Path(root_dir) / task
        if not task_dir.is_dir():
            raise ValueError(f"MSD task directory not found: {task_dir}")

        dataset_json = self._read_dataset_json(task_dir)
        self._task_labels: dict[str, Any] = dict(dataset_json.get("labels", {}))
        self._cases: dict[str, _MSDCase] = self._discover_cases(dataset_json, task_dir)

        manifest_dict = self._read_split_manifest(split_manifest)
        self._validate_manifest_ids(manifest_dict)
        self._manifest = SplitManifest(
            train_ids=tuple(manifest_dict["train"]),
            cal_ids=tuple(manifest_dict["calibration"]),
            test_ids=tuple(manifest_dict["test"]),
        )

    @staticmethod
    def _read_dataset_json(task_dir: Path) -> Mapping[str, Any]:
        """Read and parse `<task_dir>/dataset.json`."""
        path = task_dir / "dataset.json"
        if not path.is_file():
            raise SerializationError(f"dataset.json not found at {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                parsed = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SerializationError(f"could not read/parse {path}: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise SerializationError(f"{path} must contain a JSON object, got {type(parsed)}")
        return parsed

    @staticmethod
    def _discover_cases(dataset_json: Mapping[str, Any], task_dir: Path) -> dict[str, _MSDCase]:
        """Discover and pair every labelled case from `dataset.json`'s `training` list.

        Deliberately ignores `dataset.json`'s `test` list (the official
        MSD challenge's unlabelled cases) — see module docstring §3.
        """
        training = dataset_json.get("training")
        if not isinstance(training, list) or not training:
            raise SerializationError(
                f"dataset.json at {task_dir} has no non-empty 'training' list "
                "(the required image/label pairing source for this loader)"
            )
        cases: dict[str, _MSDCase] = {}
        for entry in training:
            if not isinstance(entry, Mapping) or "image" not in entry or "label" not in entry:
                raise SerializationError(
                    "malformed dataset.json 'training' entry (expected image/label keys): "
                    f"{entry!r}"
                )
            image_path = _resolve_relative(task_dir, str(entry["image"]))
            label_path = _resolve_relative(task_dir, str(entry["label"]))
            image_id = _strip_nifti_suffix(image_path.name)
            label_id = _strip_nifti_suffix(label_path.name)
            if image_id != label_id:
                raise SerializationError(
                    f"image/label filename mismatch in dataset.json: "
                    f"{image_path.name!r} vs {label_path.name!r}"
                )
            if image_id in cases:
                raise ValueError(f"duplicate case id in dataset.json 'training' list: {image_id!r}")
            if not image_path.is_file():
                raise SerializationError(f"referenced image file does not exist: {image_path}")
            if not label_path.is_file():
                raise SerializationError(f"referenced label file does not exist: {label_path}")
            cases[image_id] = _MSDCase(id_=image_id, image_path=image_path, label_path=label_path)
        return cases

    @staticmethod
    def _read_split_manifest(
        source: Mapping[str, Sequence[Hashable]] | str | Path,
    ) -> dict[str, list[str]]:
        """Load and shape-validate a `split_manifest` (mapping or JSON file path)."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise ValueError(f"split manifest file not found: {path}")
            try:
                with path.open(encoding="utf-8") as handle:
                    raw: Any = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise SerializationError(
                    f"could not read/parse split manifest {path}: {exc}"
                ) from exc
        else:
            raw = source
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"split_manifest must be a mapping of {list(_SPLIT_NAMES)} to id lists "
                f"(or a path to a JSON file with that shape), got {type(raw)}"
            )
        missing = [name for name in _SPLIT_NAMES if name not in raw]
        if missing:
            raise ValueError(f"split_manifest is missing required split(s): {missing}")
        extra = sorted(set(raw) - set(_SPLIT_NAMES))
        if extra:
            raise ValueError(f"split_manifest has unrecognized split name(s): {extra}")
        return {name: [str(i) for i in raw[name]] for name in _SPLIT_NAMES}

    def _validate_manifest_ids(self, manifest_dict: Mapping[str, Sequence[str]]) -> None:
        """Ensure every `split_manifest` id resolves to a discovered labelled case."""
        known = set(self._cases)
        for split_name, ids in manifest_dict.items():
            unknown = sorted(set(ids) - known)
            if unknown:
                raise ValueError(
                    f"split_manifest '{split_name}' references id(s) not present in the "
                    f"discovered {self._task} training pool: {unknown}"
                )

    def load(self, split_name: str) -> Dataset:
        """Load the named split (`"train"`, `"calibration"`, or `"test"`).

        Args:
            split_name: Name of the split to load.

        Returns:
            An :class:`MSDDataset` over that split's assigned cases.

        Raises:
            ValueError: If `split_name` is not one of the three recognized
                split names.
        """
        if split_name not in _SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {_SPLIT_NAMES}, got {split_name!r}")
        ids = getattr(self._manifest, _MANIFEST_FIELD[split_name])
        cases = tuple(self._cases[i] for i in ids)
        return MSDDataset(
            cases,
            metadata_key=self._metadata_key,
            task=self._task,
            task_labels=self._task_labels,
        )
