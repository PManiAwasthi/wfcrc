"""Dataset Registry — name-keyed lookup from ``config.data.name`` to a concrete
``DatasetLoader`` subclass (MS6.1, populated from MS6.3).

Per the MS6 Architecture Specification (§3.1), this mirrors the
:data:`wfcrc.ambiguity.FAMILIES` registry pattern (MS2) exactly: a plain,
module-level ``dict``, no dynamic registration API, no metaclass magic.
MS6.1 established the (then-empty) pattern; MS6.3A registered the first
concrete loader; **DI-2** registers every remaining locally-available
Phase-A loader family, each keyed identically to
:data:`wfcrc.datasets.metadata.DATASET_METADATA` (§3.1's consistency
requirement). The Config Resolver (MS6.6) is the intended consumer, looking
up ``DATASETS[config.data.name]`` exactly as
``wfcrc.runner.runner._build_family`` looks up ``FAMILIES[cfg.type]`` today.

Registered families (DI-2):

- ``msd_hippocampus``, ``msd_pancreas`` -> :class:`~wfcrc.datasets.loaders.msd.MSDNiftiLoader`
  (one class, two MSD tasks — `docs/DATASET_INTEGRATION_GUIDE.md` §6's
  "zero architectural change" case; the concrete task is a constructor
  argument the Config Resolver supplies from ``config.data.params``).
- ``acdc`` -> :class:`~wfcrc.datasets.loaders.acdc.ACDCLoader`.
- ``kvasir_seg`` -> :class:`~wfcrc.datasets.loaders.kvasir.KvasirLoader`.
- ``cifar10``, ``cifar10_1`` -> :class:`~wfcrc.datasets.loaders.cifar.CifarLoader`
  (one class, two datasets — distinguished by the ``variant`` constructor
  argument, again supplied via ``config.data.params``).

**Not registered:** ``cityscapes`` / ``cityscapes_c`` — raw Cityscapes is
absent from this environment (registration-gated) and explicitly out of
DI-2 scope; a Cityscapes loader (and the Cityscapes-C corruption wrapper)
remains future work, so its :data:`DATASET_METADATA` key intentionally has
no ``DATASETS`` entry yet.
"""

from __future__ import annotations

from wfcrc.datasets.base import DatasetLoader
from wfcrc.datasets.loaders.acdc import ACDCLoader
from wfcrc.datasets.loaders.cifar import CifarLoader
from wfcrc.datasets.loaders.kvasir import KvasirLoader
from wfcrc.datasets.loaders.msd import MSDNiftiLoader

__all__ = ["DATASETS"]

#: Registry mapping a config ``data.name`` string to its concrete
#: ``DatasetLoader`` class. Several names share a class (MSD tasks; CIFAR
#: variants) — the concrete task/variant is a constructor argument, not a
#: separate class, per MS6 Architecture Specification §3.3.
DATASETS: dict[str, type[DatasetLoader]] = {
    "msd_hippocampus": MSDNiftiLoader,
    "msd_pancreas": MSDNiftiLoader,
    "acdc": ACDCLoader,
    "kvasir_seg": KvasirLoader,
    "cifar10": CifarLoader,
    "cifar10_1": CifarLoader,
}
