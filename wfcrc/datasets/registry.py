"""Dataset Registry — name-keyed lookup from ``config.data.name`` to a concrete
``DatasetLoader`` subclass (MS6.1, populated from MS6.3).

Per the MS6 Architecture Specification (§3.1), this mirrors the
:data:`wfcrc.ambiguity.FAMILIES` registry pattern (MS2) exactly: a plain,
module-level ``dict``, no dynamic registration API, no metaclass magic.
MS6.1 established the (then-empty) pattern; MS6.3A registers the first
concrete loader (``"msd_hippocampus"`` -> :class:`~wfcrc.datasets.loaders.msd.MSDNiftiLoader`,
keyed identically to :data:`wfcrc.datasets.metadata.DATASET_METADATA`, per
§3.1's own consistency requirement). The Cityscapes-format (+ ACDC/
Cityscapes-C), CIFAR, and Kvasir loader families are out of scope for this
pass and are not registered here yet. The Config Resolver (MS6.6) is the
intended consumer, looking up ``DATASETS[config.data.name]`` exactly as
``wfcrc.runner.runner._build_family`` looks up ``FAMILIES[cfg.type]`` today.
"""

from __future__ import annotations

from wfcrc.datasets.base import DatasetLoader
from wfcrc.datasets.loaders.msd import MSDNiftiLoader

__all__ = ["DATASETS"]

#: Registry mapping a config ``data.name`` string to its concrete
#: ``DatasetLoader`` class.
DATASETS: dict[str, type[DatasetLoader]] = {
    "msd_hippocampus": MSDNiftiLoader,
}
