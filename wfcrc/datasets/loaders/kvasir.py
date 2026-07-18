"""``KvasirLoader`` / ``KvasirDataset`` — the Kvasir-SEG (polyp JPG) loader (DI-2, §3.3).

Kvasir-SEG (Jha et al., MMM 2020) is a binary polyp-segmentation dataset:
1,000 endoscopic RGB images, each paired with a hand-drawn binary polyp
mask. Its role in this research program is E12 (qualitative failure-case /
clinical false-positive control), with a small fine-tune step
(`docs/DATASET_SPLIT_POLICY.md` §3.7; `docs/MODEL_POLICY.md` §1.1).

**1 · Real on-disk structure (verified against the actual local archive in
DI-2).**

```
<root>/
  images/<hash>.jpg              # endoscopic RGB frame
  masks/<hash>.jpg               # binary polyp mask, SAME <hash>.jpg filename
  kavsir_bboxes.json             # per-image bbox geometry (not consumed here)
```

- ``<root>`` is the directory that directly contains ``images/`` and
  ``masks/`` (in this environment, ``datasets/Kvasir_SEG/kvasir-seg/
  Kvasir-SEG/``). A separate ``kvasir-sessile`` subset (196 images) also
  ships alongside; it is **not** Phase-A Kvasir-SEG and is not loaded by this
  loader (point ``root_dir`` at whichever ``images/``+``masks/`` pair you
  intend).
- **Real-data validated (DI-2):** exactly **1000** images and **1000**
  masks, with **identical filenames** (image ``<hash>.jpg`` ↔ mask
  ``<hash>.jpg``); every image/mask pair has matching pixel dimensions
  (variable per image, e.g. ``622 x 529`` — Kvasir-SEG images are *not* a
  fixed size). Filenames are **opaque content hashes** (e.g.
  ``cju0qkwl35piu0993l0dewei2``) — see §3.
- **Masks are lossy JPGs, not clean bitmaps.** Although a polyp mask is
  conceptually binary (polyp vs. background), it is stored as a 3-channel
  JPG (``R==G==B``, confirmed) whose lossy compression introduces a few
  intermediate values near object edges. Real-data measurement: mask
  intensities cluster almost entirely at ``<10`` (background, ~72%) and
  ``>127`` (polyp, ~28%), with essentially none in between. :meth:`labels`
  therefore thresholds at :data:`_MASK_THRESHOLD` (``127``) — the standard,
  robust binarization for a lossy-JPG binary mask — rather than assuming
  clean ``{0, 255}`` values.

**2 · Label representation.** Unlike ACDC's multi-class label (`acdc.py`
§3), Kvasir-SEG is genuinely binary, so :meth:`KvasirDataset.labels` returns
an unambiguous ``bool`` foreground mask directly (``raw_mask > 127``),
satisfying the frozen :meth:`wfcrc.losses.base.LossEvaluator.evaluate`
``dtype == bool`` contract with **no** binarization-policy decision required.
The raw pre-threshold grayscale mask remains available via
:meth:`KvasirDataset.raw_mask` (so a caller may re-threshold if a future
analysis needs a different cut).

**3 · Split unit — an OPEN methodological question, deliberately left
unresolved (not worked around).** `docs/DATASET_SPLIT_POLICY.md` §8 item 1
flags Kvasir-SEG's experimental unit as unresolved: endoscopic datasets
often contain multiple correlated frames from one procedure/video, and
frame-level splitting then leaks correlated frames across
train/calibration/test roles. **Whether per-procedure grouping is
recoverable was re-investigated directly against the real local archive in
DI-2, and it is not:**

- Filenames are opaque content hashes with no visible procedure/patient/
  sequence structure (955 distinct 6-character prefixes across 1000 images —
  no clustering).
- The only metadata file, ``kavsir_bboxes.json``, carries **only** geometry
  (``height``, ``width``, a ``polyp`` bounding box per image) — **no**
  patient, procedure, video, or sequence field.

Per this project's "if the dataset itself cannot answer it, leave the
decision unresolved, document it, and implement only the loader — do not
invent a split policy" rule (the DI-2 Phase-4 instruction and
`PROJECT_CONTEXT.md` §1), **this loader takes no position on the split
unit.** It is entirely split-agnostic: like every other loader
(`docs/DATASET_INTEGRATION_GUIDE.md` §7) it validates a caller-supplied
``split_manifest`` and never invents a ratio, seed, or grouping. Deciding
whether frame-level splitting is scientifically acceptable — or obtaining
procedure metadata from the dataset stewards — remains a prerequisite for
any *reported* Kvasir-SEG experiment (`docs/DATASET_SPLIT_POLICY.md` §8),
gating the manifest, not this loader.

**4 · No preprocessing baked in.** Images/masks are returned at native
resolution; resize/normalize is applied downstream via the frozen
:func:`wfcrc.datasets.preprocessing.resize_and_normalize`, never here
(the same boundary MSD/ACDC observe).

**5 · Failure modes — frozen/sanctioned exception types only, no new
exception.** :class:`~wfcrc.exceptions.SerializationError` for content
problems (a missing/undecodable image or mask); ``ValueError`` for
structural/config problems (a nonexistent ``root_dir``/subdirectory, an
unrecognized ``split_name``, or a ``split_manifest`` id outside the
discovered pool); :class:`~wfcrc.exceptions.SplitLeakageError` (unchanged)
for an overlapping ``split_manifest``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wfcrc.datasets.base import (
    Dataset,
    DatasetLoader,
    IntegrityIssue,
    IntegrityReport,
)
from wfcrc.datasets.loaders._image_io import read_image_rgb, read_label_image
from wfcrc.datasets.loaders._split_support import (
    MANIFEST_FIELD,
    SPLIT_NAMES,
    build_manifest,
    read_split_manifest,
    validate_manifest_ids,
)
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.exceptions import SerializationError

__all__ = ["KvasirDataset", "KvasirLoader"]

#: The dataset's `DATASET_METADATA` key.
_METADATA_KEY = "kvasir_seg"

#: Grayscale threshold binarizing a lossy-JPG polyp mask (module docstring §1).
#: Intensities above this are foreground (polyp); at/below are background.
_MASK_THRESHOLD = 127

_IMAGE_SUFFIX = ".jpg"


@dataclass(frozen=True)
class _KvasirCase:
    """One discovered, paired Kvasir-SEG image/mask case."""

    id_: str
    image_path: Path
    mask_path: Path


class KvasirDataset(Dataset):
    """A loaded Kvasir-SEG split: lazily reads RGB image / binary mask on access.

    See :mod:`wfcrc.datasets.loaders.kvasir`'s module docstring (§1, §2) for
    the mask-thresholding rationale and the open split-unit question.
    """

    def __init__(self, cases: Sequence[_KvasirCase]) -> None:
        """Construct a split from its already-selected, already-paired cases.

        Args:
            cases: The cases assigned to this split, in stable order.

        Raises:
            ValueError: If ``cases`` contains two entries sharing an ``id_``.
        """
        self._cases: tuple[_KvasirCase, ...] = tuple(cases)
        ids_seen = [c.id_ for c in self._cases]
        if len(ids_seen) != len(set(ids_seen)):
            duplicates = sorted({i for i in ids_seen if ids_seen.count(i) > 1})
            raise ValueError(f"duplicate id(s) within this split: {duplicates}")
        self._by_id: dict[Hashable, _KvasirCase] = {c.id_: c for c in self._cases}

    def _case_for(self, id_: Hashable) -> _KvasirCase:
        """Look up the case for ``id_``, or raise if it is not in this split."""
        case = self._by_id.get(id_)
        if case is None:
            raise ValueError(f"unknown id {id_!r} for this split")
        return case

    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        """Yield ``(id, image, label)`` triples at native resolution."""
        for case in self._cases:
            yield case.id_, self.image(case.id_), self.labels(case.id_)

    def __len__(self) -> int:
        return len(self._cases)

    def ids(self) -> Sequence[Hashable]:
        return tuple(c.id_ for c in self._cases)

    def image(self, id_: Hashable) -> NDArray[np.uint8]:
        """Return the native-resolution ``(H, W, 3)`` ``uint8`` RGB image for ``id_``."""
        return read_image_rgb(self._case_for(id_).image_path)

    def raw_mask(self, id_: Hashable) -> NDArray[np.uint8]:
        """Return the raw, pre-threshold ``(H, W)`` ``uint8`` grayscale mask for ``id_``.

        Additive accessor (`docs/DATASET_INTEGRATION_GUIDE.md` §2.1):
        preserves the lossy-JPG mask intensities that :meth:`labels`
        thresholds, so a caller may apply a different cut if needed.
        """
        return read_label_image(self._case_for(id_).mask_path)

    def labels(self, id_: Hashable) -> NDArray[np.bool_]:
        """Return the binary polyp foreground mask (``raw_mask > 127``) for ``id_``.

        See module docstring §1/§2 for why the raw JPG mask is thresholded
        at :data:`_MASK_THRESHOLD` rather than compared to clean ``{0, 255}``.
        """
        return self.raw_mask(id_) > _MASK_THRESHOLD

    def resolution(self, id_: Hashable) -> tuple[int, int]:
        """Return ``id_``'s image ``(height, width)`` in pixels.

        Additive per-case accessor: Kvasir-SEG images are variable-size
        (module docstring §1), so each case's own dimensions are read rather
        than assumed.
        """
        image = self.image(id_)
        return int(image.shape[0]), int(image.shape[1])

    def verify_integrity(self) -> IntegrityReport:
        """Check every case for content-level integrity problems.

        Per case: image and mask readable; image/mask spatial-extent
        agreement (``(H, W)``). ``uint8`` rasters cannot hold NaN/Inf, and a
        polyp mask has no discrete label vocabulary to validate against (it
        is a thresholded grayscale intensity, module docstring §1), so those
        checks (which apply to MSD/ACDC) do not apply here. Collects every
        issue rather than raising on the first, mirroring
        :class:`~wfcrc.datasets.base.IntegrityReport`.

        Returns:
            An :class:`~wfcrc.datasets.base.IntegrityReport` (empty if intact).
        """
        issues: list[IntegrityIssue] = []
        for case in self._cases:
            try:
                image = read_image_rgb(case.image_path)
            except Exception as exc:
                issues.append(IntegrityIssue(case.id_, f"image unreadable: {exc}"))
                continue
            try:
                mask = read_label_image(case.mask_path)
            except Exception as exc:
                issues.append(IntegrityIssue(case.id_, f"mask unreadable: {exc}"))
                continue
            if image.shape[:2] != mask.shape:
                issues.append(
                    IntegrityIssue(
                        case.id_,
                        f"image/mask shape mismatch: {image.shape[:2]} vs {mask.shape}",
                    )
                )
        return IntegrityReport(issues=tuple(issues))

    def meta(self) -> dict[str, Any]:
        """Return :data:`~wfcrc.datasets.metadata.DATASET_METADATA`'s record, plus label info."""
        record = DATASET_METADATA[_METADATA_KEY].to_dict()
        record["label_kind"] = "binary_polyp_mask"
        record["mask_threshold"] = _MASK_THRESHOLD
        record["split_unit_status"] = (
            "UNRESOLVED — per-procedure grouping not recoverable from the local "
            "archive (opaque-hash filenames, geometry-only bbox metadata); see "
            "docs/DATASET_SPLIT_POLICY.md §8 item 1 and this loader's module "
            "docstring §3. The loader takes no position on the split unit."
        )
        return record


class KvasirLoader(DatasetLoader):
    """Concrete :class:`~wfcrc.datasets.base.DatasetLoader` for the Kvasir-SEG format.

    See :mod:`wfcrc.datasets.loaders.kvasir`'s module docstring for the full
    design record, including the deliberately-unresolved split-unit question
    (§3): this loader is split-agnostic and never invents a split.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split_manifest: Mapping[str, Sequence[Hashable]] | str | Path,
    ) -> None:
        """Discover image/mask pairs and validate a caller-supplied split assignment.

        Args:
            root_dir: Directory that directly contains ``images/`` and
                ``masks/`` (see module docstring §1).
            split_manifest: Either a mapping ``{"train": [...],
                "calibration": [...], "test": [...]}`` of image-stem ids, or a
                path to a JSON file with that shape. Never defaulted — see
                module docstring §3.

        Raises:
            ValueError: If ``root_dir`` or its ``images/``/``masks/``
                subdirectories are missing, or ``split_manifest`` references
                an id outside the discovered pool.
            SerializationError: If a ``split_manifest`` file is
                missing/unparsable, or a discovered image has no paired mask.
            SplitLeakageError: If ``split_manifest``'s splits overlap.
        """
        root = Path(root_dir)
        self._images_dir = root / "images"
        self._masks_dir = root / "masks"
        if not self._images_dir.is_dir():
            raise ValueError(f"Kvasir-SEG images directory not found: {self._images_dir}")
        if not self._masks_dir.is_dir():
            raise ValueError(f"Kvasir-SEG masks directory not found: {self._masks_dir}")

        self._cases: dict[str, _KvasirCase] = self._discover_cases()

        manifest_dict = read_split_manifest(split_manifest)
        validate_manifest_ids(
            manifest_dict, list(self._cases), pool_description="Kvasir-SEG image pool"
        )
        self._manifest = build_manifest(manifest_dict)

    def _discover_cases(self) -> dict[str, _KvasirCase]:
        """Discover and pair every ``images/<hash>.jpg`` with its ``masks/<hash>.jpg``.

        Raises:
            SerializationError: If a discovered image has no identically-named
                mask, or no images are found at all.
        """
        # Ids are the ``.jpg`` stems from a single ``images/`` directory, so
        # they are unique by filesystem construction (no discovery-level
        # duplicate check is needed; the reachable within-split duplicate case
        # — a caller-supplied manifest listing an id twice — is guarded by
        # ``KvasirDataset.__init__`` instead).
        cases: dict[str, _KvasirCase] = {}
        for image_path in sorted(self._images_dir.glob(f"*{_IMAGE_SUFFIX}")):
            stem = image_path.name[: -len(_IMAGE_SUFFIX)]
            mask_path = self._masks_dir / image_path.name
            if not mask_path.is_file():
                raise SerializationError(
                    f"Kvasir-SEG image {image_path} has no paired mask at {mask_path}"
                )
            cases[stem] = _KvasirCase(id_=stem, image_path=image_path, mask_path=mask_path)
        if not cases:
            raise SerializationError(
                f"no Kvasir-SEG images discovered under {self._images_dir} — "
                "is this the directory containing images/ and masks/?"
            )
        return cases

    def load(self, split_name: str) -> Dataset:
        """Load the named split (``"train"``, ``"calibration"``, or ``"test"``).

        Args:
            split_name: Name of the split to load.

        Returns:
            A :class:`KvasirDataset` over that split's assigned cases.

        Raises:
            ValueError: If ``split_name`` is not a recognized split name.
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
        ids = getattr(self._manifest, MANIFEST_FIELD[split_name])
        cases = tuple(self._cases[i] for i in ids)
        return KvasirDataset(cases)
