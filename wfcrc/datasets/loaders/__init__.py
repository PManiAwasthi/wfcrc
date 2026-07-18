"""Concrete :class:`~wfcrc.datasets.base.DatasetLoader` implementations (MS6.3).

Per the MS6 Architecture Specification (§3.3), each Phase-A dataset family
gets one loader module here. MS6.3A implements only the MSD/NIfTI family
(:mod:`wfcrc.datasets.loaders.msd`), starting with MSD Task04_Hippocampus —
see that module's docstring for the full design record. The remaining three
families (Cityscapes-format [+ ACDC + Cityscapes-C], CIFAR, Kvasir) are
explicitly out of scope for this pass.
"""

from __future__ import annotations

from wfcrc.datasets.loaders.msd import MSDDataset, MSDNiftiLoader

__all__ = ["MSDDataset", "MSDNiftiLoader"]
