"""``ACDCLoader`` / ``ACDCDataset`` — the ACDC (Cityscapes-format PNG) loader (DI-2, §3.3).

ACDC (Sakaridis, Dai & Van Gool, ICCV 2021 — the **driving** Adverse
Conditions Dataset, *not* the cardiac-MRI ACDC;
`DATASET_METADATA["acdc"].extra["name_collision_warning"]`) is a semantic
segmentation dataset using the Cityscapes 19-class label format. Its role in
this research program is E4 (real distribution shift), reusing a
Cityscapes-trained segmenter unchanged
(`docs/DATASET_SPLIT_POLICY.md` §3.4; `docs/MODEL_POLICY.md` §1.1).

**1 · Real on-disk structure (verified against the actual local archive in
DI-2, not the MS6 Architecture Specification §3.3 sketch).** The spec sketch
assumed a plain Cityscapes ``<split>/<city>/`` layout; the real ACDC archive
(also found by MS11's Task-2 filesystem audit) nests one level deeper, by
**adverse-weather condition** and by **GoPro camera sequence**:

```
<root>/
  rgb_anon_trainvaltest/rgb_anon/<condition>/<split>/<sequence>/
      <stem>_rgb_anon.png            # the input RGB frame
  gt_trainval/gt/<condition>/<split>/<sequence>/
      <stem>_gt_labelTrainIds.png    # Cityscapes-19 trainId label map
      <stem>_gt_labelIds.png         # (unused: raw Cityscapes ids, 0-33)
      <stem>_gt_labelColor.png       # (unused: colourised visualisation)
      <stem>_gt_invGray.png / _gt_invIds.png   # (unused)
```

- ``<condition>`` ∈ ``{fog, night, rain, snow}``; ``<split>`` ∈
  ``{train, val, test}`` under ``rgb_anon``, but ``gt_trainval`` contains
  **only** ``train``/``val`` — the official ``test`` split's ground truth is
  withheld for the ACDC leaderboard (the same withheld-test convention
  Cityscapes/MSD use, `docs/DATASET_SPLIT_POLICY.md` §3). This loader
  therefore discovers exactly the **labelled** ``train``+``val`` frames and
  never exposes the unlabelled ``test`` frames — the direct analogue of
  `MSDNiftiLoader` ignoring MSD's unlabelled challenge ``test`` pool.
- ``rgb_anon`` additionally contains ``*_ref`` directories (clear-weather
  reference frames aligned to each adverse frame). These carry no
  ``gt_trainval`` labels and are **not** discovered — only ``train``/``val``
  (non-``ref``) frames are.
- **Real-data validated (DI-2):** 1600 ``train`` + 406 ``val`` = **2006**
  labelled frames, each RGB frame paired 1:1 with its
  ``_gt_labelTrainIds.png`` (zero missing labels); every frame stem (e.g.
  ``GOPR0475_frame_000041``) is **globally unique** across all
  conditions/splits/sequences, so the stem alone is a stable id (no
  qualification needed). Frames are ``1920 x 1080`` RGB; label maps are
  single-channel (Pillow mode ``"L"``) with values in ``{0..18}`` (the 19
  Cityscapes train classes) plus ``{255}`` (the ``ignore``/void index).

**2 · Why this is a new loader family, not a `MSDNiftiLoader` reuse.**
`docs/DATASET_INTEGRATION_GUIDE.md` §6 already assessed ACDC as needing a
genuinely new concrete loader (a PNG raster format, not NIfTI) while
requiring **no change** to the frozen
``Dataset``/``DatasetLoader``/``SplitManifest`` architecture — which this
loader honors: it implements exactly ``Dataset``'s five abstract methods and
``DatasetLoader``'s one, reuses the shared `split_manifest` mechanism
(:mod:`wfcrc.datasets.loaders._split_support`) and the shared image reader
(:mod:`wfcrc.datasets.loaders._image_io`), and returns metadata from
:data:`wfcrc.datasets.metadata.DATASET_METADATA`. Cityscapes itself (absent
from this environment, registration-gated, and explicitly out of DI-2 scope)
would be a sibling loader sharing this file's Cityscapes-19 trainId label
semantics — deliberately **not** built here (no data to validate against;
the DI-2 self-audit's "do not over-generalize" rule).

**3 · Label representation — a disclosed binarization decision (Cityscapes-19
is multi-class; the frozen loss contract is binary).** The frozen
:meth:`wfcrc.losses.base.LossEvaluator.evaluate` requires ``label.dtype ==
bool`` (a per-pixel/per-element boolean set), and the frozen segmentation
prediction-set constructor
(:class:`wfcrc.prediction_sets.segmentation.MorphologicalSets`) dilates a
single boolean seed mask — i.e. the frozen segmentation pipeline is
**binary**. A 19-class driving label has **no canonical binary foreground**
(unlike MSD's unambiguous organ foreground, `msd.py` §4): every valid pixel
belongs to one of 19 real classes, and ``255`` is ``ignore``, not a
"background" class. Per this project's "if something is ambiguous, do not
invent behavior" rule (`PROJECT_CONTEXT.md` §1):

- :meth:`ACDCDataset.raw_labels` returns the **faithful, complete**
  ``uint8`` trainId map (``{0..18}`` plus ``{255}``) — the primary label for
  any multi-class or per-class work (E2/E4 group masks, MS6.7).
- :meth:`ACDCDataset.labels` performs an **explicit, caller-supplied
  one-vs-rest** binarization: ``raw == foreground_class`` for a
  ``foreground_class`` given at loader construction. With **no**
  ``foreground_class``, :meth:`labels` **raises** (rather than guessing a
  binary task) — the same "raise, do not invent" pattern
  ``MorphologicalSets(direction='erosion')`` already uses. The
  ``ignore`` (``255``) pixels binarize to ``False`` (not-foreground);
  ``ignore``-aware evaluation (excluding void pixels from the loss) is a
  downstream ScoreProvider/experiment concern, not the loader's — the
  faithful ``ignore`` index is preserved in :meth:`raw_labels`.

This ``foreground_class`` keyword-only constructor parameter is an additive,
disclosed gap-fill beyond MS6 spec §3.3's ``(root_dir, label_map)`` sketch,
exactly as `MSDNiftiLoader`'s ``split_manifest`` parameter was (`msd.py` §2).

**4 · No split policy invented.** Identical discipline to `MSDNiftiLoader`
(`msd.py` §3) and `docs/DATASET_INTEGRATION_GUIDE.md` §7: this loader
discovers the full 2006-frame labelled pool and validates a caller-supplied
``split_manifest`` against it; it never chooses a ratio or seed.
`docs/DATASET_SPLIT_POLICY.md` §3.4's policy (pool ``train``+``val``, 50/50
calibration/test, ``train`` empty — ACDC trains nothing) is a *document*,
consumed by whoever builds the manifest, not enforced here.

**5 · No preprocessing baked in.** Frames are returned at native
``1920 x 1080`` resolution; resize/normalize (a Cityscapes-segmenter-specific
choice, `docs/MODEL_POLICY.md` §6) is applied downstream via the frozen
:func:`wfcrc.datasets.preprocessing.resize_and_normalize`, never by this
loader — the same boundary MSD observes (`msd.py` §5).

**6 · Failure modes — frozen/sanctioned exception types only, no new
exception (matching `msd.py` §6).** :class:`~wfcrc.exceptions.SerializationError`
for content problems (a missing/undecodable image or label file);
``ValueError`` for structural/config problems (an unsupported ``label_map``
or ``foreground_class``, a nonexistent ``root_dir``/subdirectory, an
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

__all__ = ["ACDCDataset", "ACDCLoader"]

#: The dataset's `DATASET_METADATA` key.
_METADATA_KEY = "acdc"

#: Supported label vocabularies. Only the Cityscapes 19-class trainId scheme
#: exists for ACDC (and for the future Cityscapes loader that would share it);
#: an unknown value raises rather than guessing a vocabulary.
_CITYSCAPES_19_TRAINIDS: frozenset[int] = frozenset(range(19))
#: The Cityscapes ``ignore``/void trainId (excluded from the 19 evaluated classes).
_IGNORE_INDEX = 255
_LABEL_MAPS: dict[str, frozenset[int]] = {"cityscapes_19": _CITYSCAPES_19_TRAINIDS}

#: The adverse-weather conditions ACDC organizes frames by.
_CONDITIONS: tuple[str, ...] = ("fog", "night", "rain", "snow")
#: The labelled splits (ACDC's official ``test`` split has withheld labels).
_LABELLED_SPLITS: tuple[str, ...] = ("train", "val")

_RGB_SUFFIX = "_rgb_anon.png"
_LABEL_SUFFIX = "_gt_labelTrainIds.png"


@dataclass(frozen=True)
class _ACDCFrame:
    """One discovered, paired ACDC labelled frame."""

    id_: str
    image_path: Path
    label_path: Path
    condition: str
    official_split: str


class ACDCDataset(Dataset):
    """A loaded ACDC split: lazily reads RGB frame / trainId label on access.

    See :mod:`wfcrc.datasets.loaders.acdc`'s module docstring (§3) for why
    :meth:`labels` requires an explicit ``foreground_class`` and
    :meth:`raw_labels` returns the faithful multi-class trainId map.
    """

    def __init__(
        self,
        frames: Sequence[_ACDCFrame],
        *,
        label_map: str,
        foreground_class: int | None,
    ) -> None:
        """Construct a split from its already-selected, already-paired frames.

        Args:
            frames: The frames assigned to this split, in stable order.
            label_map: The label vocabulary name (e.g. ``"cityscapes_19"``).
            foreground_class: The trainId :meth:`labels` binarizes against
                (``raw == foreground_class``), or ``None`` to make
                :meth:`labels` raise (no canonical binary foreground exists;
                see module docstring §3).

        Raises:
            ValueError: If ``frames`` contains two entries sharing an ``id_``.
        """
        self._frames: tuple[_ACDCFrame, ...] = tuple(frames)
        ids_seen = [f.id_ for f in self._frames]
        if len(ids_seen) != len(set(ids_seen)):
            duplicates = sorted({i for i in ids_seen if ids_seen.count(i) > 1})
            raise ValueError(f"duplicate id(s) within this split: {duplicates}")
        self._by_id: dict[Hashable, _ACDCFrame] = {f.id_: f for f in self._frames}
        self._label_map = label_map
        self._foreground_class = foreground_class

    def _frame_for(self, id_: Hashable) -> _ACDCFrame:
        """Look up the frame for ``id_``, or raise if it is not in this split."""
        frame = self._by_id.get(id_)
        if frame is None:
            raise ValueError(f"unknown id {id_!r} for this split")
        return frame

    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        """Yield ``(id, image, label)`` triples at native resolution."""
        for frame in self._frames:
            yield frame.id_, self.image(frame.id_), self.labels(frame.id_)

    def __len__(self) -> int:
        return len(self._frames)

    def ids(self) -> Sequence[Hashable]:
        return tuple(f.id_ for f in self._frames)

    def image(self, id_: Hashable) -> NDArray[np.uint8]:
        """Return the native-resolution ``(H, W, 3)`` ``uint8`` RGB frame for ``id_``."""
        return read_image_rgb(self._frame_for(id_).image_path)

    def raw_labels(self, id_: Hashable) -> NDArray[np.uint8]:
        """Return the faithful ``(H, W)`` ``uint8`` Cityscapes-19 trainId map for ``id_``.

        Values are in ``{0..18}`` (the 19 evaluated classes) plus ``{255}``
        (``ignore``/void). This is the primary label for multi-class /
        per-class work; see module docstring §3.
        """
        return read_label_image(self._frame_for(id_).label_path)

    def labels(self, id_: Hashable) -> NDArray[np.bool_]:
        """Return the one-vs-rest boolean foreground mask for ``id_``.

        Returns ``raw_labels(id_) == foreground_class``. See module docstring
        §3 for why a ``foreground_class`` must be chosen explicitly.

        Raises:
            ValueError: If no ``foreground_class`` was supplied at loader
                construction (no canonical binary foreground exists for a
                19-class label — supply ``foreground_class`` for a specific
                one-vs-rest task, or use :meth:`raw_labels` for the full
                multi-class map).
        """
        if self._foreground_class is None:
            raise ValueError(
                "ACDCDataset.labels() needs an explicit foreground_class: the "
                "Cityscapes-19 label is multi-class with no canonical binary "
                "foreground, and no frozen document specifies one. Construct the "
                "loader with foreground_class=<trainId> for a one-vs-rest task, "
                "or use raw_labels() for the full multi-class trainId map "
                "(see this module's docstring §3)."
            )
        # ndarray.__eq__ is typed as returning Any (it accepts arbitrary
        # objects); make the boolean-array result explicit for mypy --strict.
        return np.asarray(self.raw_labels(id_) == self._foreground_class, dtype=np.bool_)

    def resolution(self, id_: Hashable) -> tuple[int, int]:
        """Return ``id_``'s frame ``(height, width)`` in pixels.

        Additive per-case accessor (`docs/DATASET_INTEGRATION_GUIDE.md` §2's
        "format-specific header accessors ... pixel resolution"): reads each
        frame's own size rather than assuming the dataset-wide
        ``1080 x 1920``, since nothing guarantees a future release keeps it.
        """
        image = self.image(id_)
        return int(image.shape[0]), int(image.shape[1])

    def condition(self, id_: Hashable) -> str:
        """Return ``id_``'s adverse-weather condition (``"fog"``/``"night"``/``"rain"``/``"snow"``).

        Additive per-case accessor: ACDC's condition is a natural grouping
        axis for the E4 real-shift analysis; exposed here so a future Group
        Mask Builder (MS6.7) can derive per-condition groups without
        re-walking the directory tree.
        """
        return self._frame_for(id_).condition

    def verify_integrity(self) -> IntegrityReport:
        """Check every frame for content-level integrity problems.

        Per frame: image and label readable; image/label spatial-extent
        agreement (``(H, W)``); label values within the declared vocabulary
        (``{0..18}`` plus ``{255}`` for ``cityscapes_19``). ``uint8`` rasters
        cannot hold NaN/Inf, so no finiteness check applies (unlike the
        float NIfTI volumes of `msd.py` §9 / `docs/DATASET_INTEGRATION_GUIDE.md`
        §4.1). Collects every issue rather than raising on the first, mirroring
        :class:`~wfcrc.datasets.base.IntegrityReport`'s established pattern.

        Returns:
            An :class:`~wfcrc.datasets.base.IntegrityReport` (empty if intact).
        """
        allowed = set(_LABEL_MAPS[self._label_map]) | {_IGNORE_INDEX}
        issues: list[IntegrityIssue] = []
        for frame in self._frames:
            try:
                image = read_image_rgb(frame.image_path)
            except Exception as exc:
                issues.append(IntegrityIssue(frame.id_, f"image unreadable: {exc}"))
                continue
            try:
                label = read_label_image(frame.label_path)
            except Exception as exc:
                issues.append(IntegrityIssue(frame.id_, f"label unreadable: {exc}"))
                continue
            if image.shape[:2] != label.shape:
                issues.append(
                    IntegrityIssue(
                        frame.id_,
                        f"image/label shape mismatch: {image.shape[:2]} vs {label.shape}",
                    )
                )
            found = {int(v) for v in np.unique(label)}
            unexpected = sorted(found - allowed)
            if unexpected:
                issues.append(
                    IntegrityIssue(
                        frame.id_,
                        f"label contains trainId value(s) outside the declared "
                        f"'{self._label_map}' vocabulary {sorted(allowed)}: {unexpected}",
                    )
                )
        return IntegrityReport(issues=tuple(issues))

    def meta(self) -> dict[str, Any]:
        """Return :data:`~wfcrc.datasets.metadata.DATASET_METADATA`'s record, plus label info."""
        record = DATASET_METADATA[_METADATA_KEY].to_dict()
        record["label_map"] = self._label_map
        record["num_classes"] = len(_LABEL_MAPS[self._label_map])
        record["ignore_index"] = _IGNORE_INDEX
        record["foreground_class"] = self._foreground_class
        return record


class ACDCLoader(DatasetLoader):
    """Concrete :class:`~wfcrc.datasets.base.DatasetLoader` for the ACDC PNG format.

    See :mod:`wfcrc.datasets.loaders.acdc`'s module docstring for the full
    design record (real directory layout, the multi-class-label binarization
    decision, and the no-invented-split discipline).
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split_manifest: Mapping[str, Sequence[Hashable]] | str | Path,
        foreground_class: int | None = None,
        label_map: str = "cityscapes_19",
    ) -> None:
        """Discover labelled frames and validate a caller-supplied split assignment.

        Args:
            root_dir: Directory containing ``rgb_anon_trainvaltest/`` and
                ``gt_trainval/`` (i.e. the extracted ACDC root).
            split_manifest: Either a mapping ``{"train": [...],
                "calibration": [...], "test": [...]}`` of frame-stem ids, or a
                path to a JSON file with that shape. Never defaulted — see
                module docstring §4.
            foreground_class: The Cityscapes trainId :meth:`ACDCDataset.labels`
                binarizes against (one-vs-rest). ``None`` (the default) makes
                :meth:`ACDCDataset.labels` raise; :meth:`ACDCDataset.raw_labels`
                works regardless. See module docstring §3.
            label_map: The label vocabulary; only ``"cityscapes_19"`` is
                supported.

        Raises:
            ValueError: If ``label_map`` is unsupported, ``foreground_class``
                is not a valid trainId for ``label_map``, ``root_dir`` or its
                expected subdirectories are missing, or ``split_manifest``
                references an id outside the discovered pool.
            SerializationError: If a ``split_manifest`` file is missing/unparsable.
            SplitLeakageError: If ``split_manifest``'s splits overlap.
        """
        if label_map not in _LABEL_MAPS:
            raise ValueError(
                f"unsupported label_map {label_map!r}; supported: {sorted(_LABEL_MAPS)}"
            )
        if foreground_class is not None and foreground_class not in _LABEL_MAPS[label_map]:
            raise ValueError(
                f"foreground_class {foreground_class!r} is not a valid trainId for "
                f"label_map {label_map!r} (valid: {sorted(_LABEL_MAPS[label_map])})"
            )
        self._label_map = label_map
        self._foreground_class = foreground_class

        root = Path(root_dir)
        self._rgb_root = root / "rgb_anon_trainvaltest" / "rgb_anon"
        self._gt_root = root / "gt_trainval" / "gt"
        if not self._rgb_root.is_dir():
            raise ValueError(f"ACDC rgb_anon directory not found: {self._rgb_root}")
        if not self._gt_root.is_dir():
            raise ValueError(f"ACDC gt directory not found: {self._gt_root}")

        self._frames: dict[str, _ACDCFrame] = self._discover_frames()

        manifest_dict = read_split_manifest(split_manifest)
        validate_manifest_ids(
            manifest_dict, list(self._frames), pool_description="ACDC labelled (train+val) pool"
        )
        self._manifest = build_manifest(manifest_dict)

    def _discover_frames(self) -> dict[str, _ACDCFrame]:
        """Discover and pair every labelled (``train``+``val``) frame.

        Walks ``rgb_anon/<condition>/<split in {train,val}>/<sequence>/`` in a
        deterministic, sorted order and pairs each ``*_rgb_anon.png`` with its
        ``*_gt_labelTrainIds.png`` sibling under ``gt/``. Ignores the
        unlabelled official ``test`` split and the ``*_ref`` reference
        directories (see module docstring §1).

        Raises:
            SerializationError: If a discovered RGB frame has no paired label
                file (a corrupt/incomplete archive).
            ValueError: If two frames yield the same id (should never happen
                for a well-formed archive — ACDC stems are globally unique,
                module docstring §1 — but rejected explicitly rather than
                silently collapsed).
        """
        frames: dict[str, _ACDCFrame] = {}
        for condition in sorted(p.name for p in self._rgb_root.iterdir() if p.is_dir()):
            if condition not in _CONDITIONS:
                continue
            for split in _LABELLED_SPLITS:
                split_dir = self._rgb_root / condition / split
                if not split_dir.is_dir():
                    continue
                for sequence in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                    for rgb_path in sorted(sequence.glob(f"*{_RGB_SUFFIX}")):
                        stem = rgb_path.name[: -len(_RGB_SUFFIX)]
                        label_path = (
                            self._gt_root
                            / condition
                            / split
                            / sequence.name
                            / f"{stem}{_LABEL_SUFFIX}"
                        )
                        if not label_path.is_file():
                            raise SerializationError(
                                f"ACDC frame {rgb_path} has no paired label at {label_path}"
                            )
                        if stem in frames:
                            raise ValueError(f"duplicate ACDC frame id discovered: {stem!r}")
                        frames[stem] = _ACDCFrame(
                            id_=stem,
                            image_path=rgb_path,
                            label_path=label_path,
                            condition=condition,
                            official_split=split,
                        )
        if not frames:
            raise SerializationError(
                f"no labelled ACDC frames discovered under {self._rgb_root} — "
                "is this an extracted ACDC archive? (expected "
                "rgb_anon/<condition>/<train|val>/<sequence>/*_rgb_anon.png)"
            )
        return frames

    def load(self, split_name: str) -> Dataset:
        """Load the named split (``"train"``, ``"calibration"``, or ``"test"``).

        Args:
            split_name: Name of the split to load.

        Returns:
            An :class:`ACDCDataset` over that split's assigned frames.

        Raises:
            ValueError: If ``split_name`` is not a recognized split name.
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
        ids = getattr(self._manifest, MANIFEST_FIELD[split_name])
        frames = tuple(self._frames[i] for i in ids)
        return ACDCDataset(
            frames,
            label_map=self._label_map,
            foreground_class=self._foreground_class,
        )
