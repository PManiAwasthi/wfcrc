"""``CifarLoader`` / ``CifarDataset`` — the CIFAR-10 / CIFAR-10.1 loader (DI-2, §3.3).

One loader family, two registered datasets — CIFAR-10 (Krizhevsky 2009) and
CIFAR-10.1 (Recht et al. 2018) — sharing a single ``CifarDataset`` and a
single ``CifarLoader`` class, selected by the ``variant`` constructor
parameter, exactly as MS6 Architecture Specification §3.3 anticipates ("one
class, ... registry entries pointing at the same class with different
constructor params"). Only the on-disk *source format* differs (§1); the
canonical in-memory representation and the whole ``Dataset`` contract are
identical across the two.

**1 · Real on-disk structure (verified against the actual local archives in
DI-2).** The two variants ship in genuinely different formats:

- **CIFAR-10** — the classic Python pickle batches:
  ```
  <root>/data_batch_1 .. data_batch_5   # 10,000 rows each = 50,000 train
        /test_batch                     # 10,000 rows = official test
        /batches.meta                   # label_names, etc.
  ```
  Each batch unpickles (``encoding="latin1"``) to ``{"data": uint8
  (10000, 3072), "labels": list[int] 0-9, "filenames": [...],
  "batch_label": ...}``. A ``data`` row is a **flat** ``3072``-vector laid
  out as ``[R(1024) | G(1024) | B(1024)]``, each channel row-major
  ``32 x 32`` — reshaped here to canonical ``(32, 32, 3)`` HWC.
- **CIFAR-10.1** — NumPy ``.npy`` arrays under a ``datasets/`` subdir:
  ```
  <root>/datasets/cifar10.1_v6_data.npy    # uint8 (2000, 32, 32, 3), HWC already
        /datasets/cifar10.1_v6_labels.npy  # int   (2000,), 0-9
        /datasets/cifar10.1_v4_{data,labels}.npy   # the older 2021-image v4
  ```
  ``v6`` (2,000 images, class-balanced) is this project's frozen recommended
  variant (`DATASET_METADATA["cifar10_1"].extra["recommended_variant"]`;
  `docs/DATASET_SPLIT_POLICY.md` §3.6) and the default; ``v4`` (2,021
  images, the originally-tested set) is selectable via ``cifar10_1_version``.
  CIFAR-10.1's data is **already** ``(N, 32, 32, 3)`` HWC — no reshape.
- **Real-data validated (DI-2):** CIFAR-10 = 50,000 train + 10,000 test rows,
  labels ``0..9``; CIFAR-10.1 v6 = 2,000 images / v4 = 2,021 images, labels
  ``0..9``, ``uint8`` HWC.

**2 · Stable ids reflect the official partition (a real fact, not a WFCRC
policy).** Like `MSDNiftiLoader` discovering only MSD's official "training"
pool, CIFAR ids encode each example's **official** subset and index, so the
discoverable pool is transparent and a manifest cannot silently mix the
official train and test pools by accident:

- CIFAR-10: ``train_00000`` .. ``train_49999`` (batches 1-5 concatenated in
  order) and ``test_00000`` .. ``test_09999`` (``test_batch``).
- CIFAR-10.1: ``v6_0000`` .. (or ``v4_0000`` ..).

The **WFCRC role split** (which official-subset ids go to train/calibration/
test) is still the caller-supplied ``split_manifest``'s job, never invented
here (`docs/DATASET_INTEGRATION_GUIDE.md` §7). For CIFAR-10.1 specifically,
`docs/DATASET_SPLIT_POLICY.md` §3.6's categorical "100% test, 0% train/
calibration" policy is expressed by a manifest putting every id in ``test``
(empty ``train``/``calibration`` — an already-supported code path) — this
loader does not hard-code it.

**3 · Label representation — one-hot boolean, for the frozen classification
loss contract.** The frozen classification pipeline
(:class:`wfcrc.prediction_sets.classification.ThresholdSets` builds a per-class
boolean set ``{k : score_k ≥ 1-λ}`` of shape ``(K,)``;
:class:`wfcrc.losses.miscoverage.MiscoverageLoss` needs a matching boolean
``label``). :meth:`CifarDataset.labels` therefore returns the **one-hot**
boolean vector of length ``K = 10`` (``True`` at the true class index) — the
natural set-valued reading of a single-label classification target, the
direct analogue of MSD/ACDC returning a boolean segmentation mask. The raw
integer class index and its human-readable name remain available via
:meth:`CifarDataset.class_index` / :meth:`CifarDataset.class_name`.

**4 · No preprocessing baked in.** Images are returned as raw ``uint8``
``(32, 32, 3)``; normalization is applied downstream via the frozen
:func:`wfcrc.datasets.preprocessing.resize_and_normalize`
(`docs/MODEL_POLICY.md` §6), never here.

**5 · Failure modes — frozen/sanctioned exception types only, no new
exception.** :class:`~wfcrc.exceptions.SerializationError` for content
problems (a missing/undecodable batch or ``.npy`` file, a malformed batch
dict, a data/label length mismatch); ``ValueError`` for structural/config
problems (an unsupported ``variant``/``cifar10_1_version``, a nonexistent
``root_dir``, an unrecognized ``split_name``, or a ``split_manifest`` id
outside the discovered pool); :class:`~wfcrc.exceptions.SplitLeakageError`
(unchanged) for an overlapping ``split_manifest``.
"""

from __future__ import annotations

import pickle
from collections.abc import Hashable, Iterator, Mapping, Sequence
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
from wfcrc.datasets.loaders._split_support import (
    MANIFEST_FIELD,
    SPLIT_NAMES,
    build_manifest,
    read_split_manifest,
    validate_manifest_ids,
)
from wfcrc.datasets.metadata import DATASET_METADATA
from wfcrc.exceptions import SerializationError

__all__ = ["CifarDataset", "CifarLoader"]

#: ``variant`` -> `DATASET_METADATA` key.
_VARIANT_METADATA_KEYS: dict[str, str] = {
    "cifar10": "cifar10",
    "cifar10_1": "cifar10_1",
}

#: The 10 canonical CIFAR-10 class names, in label-index order (0-9). These
#: are a fixed property of the dataset (from CIFAR-10's own ``batches.meta``)
#: and are shared by CIFAR-10.1, whose integer labels use the same scheme.
_CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
#: Number of classes ``K`` (the one-hot label / per-class score length).
_NUM_CLASSES = len(_CIFAR10_CLASSES)

#: CIFAR-10 pickle batch files, in the fixed order defining ``train_*`` ids.
_CIFAR10_TRAIN_BATCHES: tuple[str, ...] = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
)
_CIFAR10_TEST_BATCH = "test_batch"

#: Supported CIFAR-10.1 versions (`docs/DATASET_SPLIT_POLICY.md` §3.6).
_CIFAR10_1_VERSIONS: tuple[str, ...] = ("v4", "v6")


def _canonical_uint8_images(images: NDArray[Any]) -> NDArray[np.uint8]:
    """Validate and return ``images`` as a ``(N, 32, 32, 3)`` ``uint8`` array."""
    arr = np.asarray(images)
    if arr.shape[1:] != (32, 32, 3):
        raise SerializationError(f"expected CIFAR images of shape (N, 32, 32, 3), got {arr.shape}")
    return arr.astype(np.uint8)


def _load_cifar10_batch(path: Path) -> tuple[NDArray[np.uint8], list[int]]:
    """Unpickle one CIFAR-10 batch, returning ``((n, 32, 32, 3) uint8, labels)``.

    Raises:
        SerializationError: If ``path`` is missing, unpicklable, or malformed.
    """
    if not path.is_file():
        raise SerializationError(f"CIFAR-10 batch file not found: {path}")
    try:
        with path.open("rb") as handle:
            batch: Any = pickle.load(handle, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
        raise SerializationError(f"could not read/unpickle CIFAR-10 batch {path}: {exc}") from exc
    if not isinstance(batch, Mapping) or "data" not in batch or "labels" not in batch:
        raise SerializationError(f"malformed CIFAR-10 batch {path} (expected 'data'/'labels' keys)")
    data = np.asarray(batch["data"])
    labels = [int(v) for v in batch["labels"]]
    if data.ndim != 2 or data.shape[1] != 3072:
        raise SerializationError(
            f"CIFAR-10 batch {path} 'data' must be (n, 3072), got {data.shape}"
        )
    if data.shape[0] != len(labels):
        raise SerializationError(
            f"CIFAR-10 batch {path} data/label length mismatch: "
            f"{data.shape[0]} vs {len(labels)}"
        )
    # [R(1024) | G(1024) | B(1024)] row-major 32x32 -> (n, 32, 32, 3) HWC.
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    return images, labels


class CifarDataset(Dataset):
    """A loaded CIFAR split: in-memory ``uint8`` images and integer class labels.

    Holds only its own split's rows (the loader slices the full pool once at
    construction). See :mod:`wfcrc.datasets.loaders.cifar`'s module docstring
    (§3) for why :meth:`labels` returns a one-hot boolean vector.
    """

    def __init__(
        self,
        ids: Sequence[str],
        images: NDArray[np.uint8],
        class_indices: NDArray[np.int64],
        *,
        metadata_key: str,
    ) -> None:
        """Construct a split from its already-selected rows.

        Args:
            ids: The split's example ids, in stable order.
            images: ``(len(ids), 32, 32, 3)`` ``uint8`` images, row-aligned
                with ``ids``.
            class_indices: ``(len(ids),)`` integer class labels (``0..9``),
                row-aligned with ``ids``.
            metadata_key: The :data:`~wfcrc.datasets.metadata.DATASET_METADATA` key.

        Raises:
            ValueError: If ``ids`` contains a duplicate, or ``ids``/``images``/
                ``class_indices`` lengths disagree.
        """
        self._ids: tuple[str, ...] = tuple(ids)
        if len(self._ids) != len(set(self._ids)):
            duplicates = sorted({i for i in self._ids if self._ids.count(i) > 1})
            raise ValueError(f"duplicate id(s) within this split: {duplicates}")
        if not (len(self._ids) == images.shape[0] == class_indices.shape[0]):
            raise ValueError(
                f"ids/images/class_indices length mismatch: {len(self._ids)}, "
                f"{images.shape[0]}, {class_indices.shape[0]}"
            )
        self._images = images
        self._class_indices = class_indices
        self._metadata_key = metadata_key
        self._index_of: dict[Hashable, int] = {id_: i for i, id_ in enumerate(self._ids)}

    def _row_for(self, id_: Hashable) -> int:
        """Return the row index for ``id_``, or raise if it is not in this split."""
        row = self._index_of.get(id_)
        if row is None:
            raise ValueError(f"unknown id {id_!r} for this split")
        return row

    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        """Yield ``(id, image, label)`` triples in stable id order."""
        for id_ in self._ids:
            yield id_, self.image(id_), self.labels(id_)

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> Sequence[Hashable]:
        return self._ids

    def image(self, id_: Hashable) -> NDArray[np.uint8]:
        """Return the ``(32, 32, 3)`` ``uint8`` RGB image for ``id_``."""
        # ndarray.__getitem__ is typed as returning Any; make the row slice's
        # element type explicit for mypy --strict.
        image: NDArray[np.uint8] = self._images[self._row_for(id_)]
        return image

    def class_index(self, id_: Hashable) -> int:
        """Return ``id_``'s integer class label (``0..9``).

        Additive accessor: the raw single-label target that :meth:`labels`
        one-hot-encodes (useful for per-class group masks / metadata).
        """
        return int(self._class_indices[self._row_for(id_)])

    def class_name(self, id_: Hashable) -> str:
        """Return ``id_``'s human-readable class name (e.g. ``"cat"``)."""
        return _CIFAR10_CLASSES[self.class_index(id_)]

    def labels(self, id_: Hashable) -> NDArray[np.bool_]:
        """Return the one-hot boolean class vector of length ``K = 10`` for ``id_``.

        ``True`` at the true class index, else ``False`` — the set-valued
        reading of the classification target matching
        :class:`~wfcrc.prediction_sets.classification.ThresholdSets`'s
        per-class boolean set (module docstring §3).
        """
        one_hot = np.zeros(_NUM_CLASSES, dtype=np.bool_)
        one_hot[self.class_index(id_)] = True
        return one_hot

    def verify_integrity(self) -> IntegrityReport:
        """Check every example's image shape and label range.

        Per example: image is ``(32, 32, 3)`` ``uint8``; class index is in
        ``[0, K)``. (Per-file readability was already enforced at load time,
        raising :class:`~wfcrc.exceptions.SerializationError`; ``uint8``
        images cannot hold NaN/Inf.) Collects every issue rather than raising
        on the first, mirroring :class:`~wfcrc.datasets.base.IntegrityReport`.

        Returns:
            An :class:`~wfcrc.datasets.base.IntegrityReport` (empty if intact).
        """
        issues: list[IntegrityIssue] = []
        for id_ in self._ids:
            row = self._index_of[id_]
            image = self._images[row]
            if image.shape != (32, 32, 3) or image.dtype != np.uint8:
                issues.append(
                    IntegrityIssue(
                        id_, f"image must be (32, 32, 3) uint8, got {image.shape} {image.dtype}"
                    )
                )
            class_index = int(self._class_indices[row])
            if not 0 <= class_index < _NUM_CLASSES:
                issues.append(
                    IntegrityIssue(id_, f"class index {class_index} outside [0, {_NUM_CLASSES})")
                )
        return IntegrityReport(issues=tuple(issues))

    def meta(self) -> dict[str, Any]:
        """Return :data:`~wfcrc.datasets.metadata.DATASET_METADATA`'s record, plus class info."""
        record = DATASET_METADATA[self._metadata_key].to_dict()
        record["num_classes"] = _NUM_CLASSES
        record["class_names"] = list(_CIFAR10_CLASSES)
        return record


class CifarLoader(DatasetLoader):
    """Concrete :class:`~wfcrc.datasets.base.DatasetLoader` for the CIFAR format.

    See :mod:`wfcrc.datasets.loaders.cifar`'s module docstring for the full
    design record (the two source formats, the official-partition id scheme,
    and the one-hot boolean label representation).
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split_manifest: Mapping[str, Sequence[Hashable]] | str | Path,
        variant: str = "cifar10",
        cifar10_1_version: str = "v6",
    ) -> None:
        """Load the full pool for ``variant`` and validate a caller-supplied split.

        Args:
            root_dir: For ``variant="cifar10"``, the directory containing the
                pickle batches (``data_batch_*``/``test_batch``). For
                ``variant="cifar10_1"``, the CIFAR-10.1 repo root (containing
                ``datasets/cifar10.1_<ver>_{data,labels}.npy``).
            split_manifest: Either a mapping ``{"train": [...],
                "calibration": [...], "test": [...]}`` of ids, or a path to a
                JSON file with that shape. Never defaulted — see module
                docstring §2.
            variant: ``"cifar10"`` or ``"cifar10_1"``.
            cifar10_1_version: ``"v6"`` (default, recommended) or ``"v4"`` —
                only consulted when ``variant="cifar10_1"``.

        Raises:
            ValueError: If ``variant``/``cifar10_1_version`` is unsupported,
                ``root_dir`` is missing, or ``split_manifest`` references an id
                outside the discovered pool.
            SerializationError: If a source data file is missing or malformed,
                or a ``split_manifest`` file is missing/unparsable.
            SplitLeakageError: If ``split_manifest``'s splits overlap.
        """
        if variant not in _VARIANT_METADATA_KEYS:
            raise ValueError(
                f"unsupported variant {variant!r}; supported: {sorted(_VARIANT_METADATA_KEYS)}"
            )
        if cifar10_1_version not in _CIFAR10_1_VERSIONS:
            raise ValueError(
                f"unsupported cifar10_1_version {cifar10_1_version!r}; "
                f"supported: {list(_CIFAR10_1_VERSIONS)}"
            )
        self._variant = variant
        self._metadata_key = _VARIANT_METADATA_KEYS[variant]
        root = Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"CIFAR root directory not found: {root}")

        if variant == "cifar10":
            self._ids, self._images, self._class_indices = self._load_cifar10_pool(root)
        else:
            self._ids, self._images, self._class_indices = self._load_cifar10_1_pool(
                root, cifar10_1_version
            )
        self._index_of = {id_: i for i, id_ in enumerate(self._ids)}

        manifest_dict = read_split_manifest(split_manifest)
        validate_manifest_ids(manifest_dict, self._ids, pool_description=f"{variant} example pool")
        self._manifest = build_manifest(manifest_dict)

    @staticmethod
    def _load_cifar10_pool(
        root: Path,
    ) -> tuple[list[str], NDArray[np.uint8], NDArray[np.int64]]:
        """Load CIFAR-10's 50,000 train + 10,000 test rows into one canonical pool."""
        ids: list[str] = []
        image_blocks: list[NDArray[np.uint8]] = []
        label_blocks: list[int] = []

        train_images: list[NDArray[np.uint8]] = []
        train_labels: list[int] = []
        for batch_name in _CIFAR10_TRAIN_BATCHES:
            images, labels = _load_cifar10_batch(root / batch_name)
            train_images.append(images)
            train_labels.extend(labels)
        for i in range(len(train_labels)):
            ids.append(f"train_{i:05d}")
        image_blocks.append(np.concatenate(train_images, axis=0))
        label_blocks.extend(train_labels)

        test_images, test_labels = _load_cifar10_batch(root / _CIFAR10_TEST_BATCH)
        for i in range(len(test_labels)):
            ids.append(f"test_{i:05d}")
        image_blocks.append(test_images)
        label_blocks.extend(test_labels)

        images_all = _canonical_uint8_images(np.concatenate(image_blocks, axis=0))
        labels_all = np.asarray(label_blocks, dtype=np.int64)
        return ids, images_all, labels_all

    @staticmethod
    def _load_cifar10_1_pool(
        root: Path, version: str
    ) -> tuple[list[str], NDArray[np.uint8], NDArray[np.int64]]:
        """Load a CIFAR-10.1 ``.npy`` pool (``version`` ``"v6"``/``"v4"``)."""
        data_path = root / "datasets" / f"cifar10.1_{version}_data.npy"
        labels_path = root / "datasets" / f"cifar10.1_{version}_labels.npy"
        for path in (data_path, labels_path):
            if not path.is_file():
                raise SerializationError(f"CIFAR-10.1 file not found: {path}")
        try:
            data = np.load(data_path)
            labels = np.load(labels_path)
        except (OSError, ValueError) as exc:
            raise SerializationError(
                f"could not load CIFAR-10.1 {version} arrays under {root}: {exc}"
            ) from exc
        images_all = _canonical_uint8_images(data)
        if labels.shape[0] != images_all.shape[0]:
            raise SerializationError(
                f"CIFAR-10.1 {version} data/label length mismatch: "
                f"{images_all.shape[0]} vs {labels.shape[0]}"
            )
        ids = [f"{version}_{i:04d}" for i in range(images_all.shape[0])]
        return ids, images_all, np.asarray(labels, dtype=np.int64)

    def load(self, split_name: str) -> Dataset:
        """Load the named split (``"train"``, ``"calibration"``, or ``"test"``).

        Args:
            split_name: Name of the split to load.

        Returns:
            A :class:`CifarDataset` holding only that split's rows.

        Raises:
            ValueError: If ``split_name`` is not a recognized split name.
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
        split_ids = list(getattr(self._manifest, MANIFEST_FIELD[split_name]))
        rows = [self._index_of[i] for i in split_ids]
        row_index = np.asarray(rows, dtype=np.intp)
        images = self._images[row_index] if rows else np.empty((0, 32, 32, 3), dtype=np.uint8)
        class_indices = self._class_indices[row_index] if rows else np.empty((0,), dtype=np.int64)
        return CifarDataset(
            split_ids,
            images,
            class_indices,
            metadata_key=self._metadata_key,
        )
