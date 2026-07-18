"""Concrete :class:`~wfcrc.datasets.base.DatasetLoader` implementations (MS6.3 / DI-2).

Per the MS6 Architecture Specification (§3.3), each Phase-A dataset *format*
gets one loader family here. MS6.3A / DI-1 built and froze the MSD/NIfTI
family (:mod:`wfcrc.datasets.loaders.msd`) as the reference implementation;
**DI-2** extends the standard to every remaining locally-available dataset:

- :mod:`wfcrc.datasets.loaders.msd` — MSD/NIfTI (Task04_Hippocampus,
  Task07_Pancreas).
- :mod:`wfcrc.datasets.loaders.acdc` — ACDC (Cityscapes-format PNG).
- :mod:`wfcrc.datasets.loaders.kvasir` — Kvasir-SEG (polyp JPG).
- :mod:`wfcrc.datasets.loaders.cifar` — CIFAR-10 / CIFAR-10.1.

Shared, private helper modules (DI-2 self-audit deduplication):
:mod:`wfcrc.datasets.loaders._split_support` (the caller-supplied
``split_manifest`` mechanism, used by all four families) and
:mod:`wfcrc.datasets.loaders._image_io` (the Pillow-backed 2-D image reader,
used by the ACDC and Kvasir families). Cityscapes itself is out of scope
(absent, registration-gated; see :mod:`wfcrc.datasets.loaders.acdc`).
"""

from __future__ import annotations

from wfcrc.datasets.loaders.acdc import ACDCDataset, ACDCLoader
from wfcrc.datasets.loaders.cifar import CifarDataset, CifarLoader
from wfcrc.datasets.loaders.kvasir import KvasirDataset, KvasirLoader
from wfcrc.datasets.loaders.msd import MSDDataset, MSDNiftiLoader

__all__ = [
    "ACDCDataset",
    "ACDCLoader",
    "CifarDataset",
    "CifarLoader",
    "KvasirDataset",
    "KvasirLoader",
    "MSDDataset",
    "MSDNiftiLoader",
]
