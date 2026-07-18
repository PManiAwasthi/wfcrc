"""Dataset Registry — name-keyed lookup from ``config.data.name`` to a concrete
``DatasetLoader`` subclass (MS6.1).

Per the MS6 Architecture Specification (§3.1), this mirrors the
:data:`wfcrc.ambiguity.FAMILIES` registry pattern (MS2) exactly: a plain,
module-level ``dict``, no dynamic registration API, no metaclass magic. The
registry starts empty — MS6.1 only establishes the pattern; no concrete
``DatasetLoader`` for any named dataset exists yet (MS6.3 populates it,
loader by loader, starting with MSD Task04_Hippocampus). The Config Resolver
(MS6.6) is the intended consumer, looking up ``DATASETS[config.data.name]``
exactly as ``wfcrc.runner.runner._build_family`` looks up
``FAMILIES[cfg.type]`` today.
"""

from __future__ import annotations

from wfcrc.datasets.base import DatasetLoader

__all__ = ["DATASETS"]

#: Registry mapping a config ``data.name`` string to its concrete
#: ``DatasetLoader`` class. Empty until MS6.3 registers the first concrete
#: loader.
DATASETS: dict[str, type[DatasetLoader]] = {}
