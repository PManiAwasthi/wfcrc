"""wfcrc: Worst-case Family Conformal Risk Control.

Reproducible research package implementing the frozen WF-CRC specification.

Milestone status: MS1 provided engineering infrastructure (configuration,
logging, numerics, seeding, caching, I/O, exceptions — under
:mod:`wfcrc.utils` and :mod:`wfcrc.config`). MS2 added the algorithmic core —
:mod:`wfcrc.losses`, :mod:`wfcrc.ambiguity`, :mod:`wfcrc.calibration`. MS3
completes the executable calibration pipeline: :mod:`wfcrc.prediction_sets`
(nested `C_λ` constructors), :mod:`wfcrc.evaluation` (the deterministic
`Verifier`), and :mod:`wfcrc.calibration.pipeline` (thin orchestration of
`LossTable → WFCRCCalibrator → Verifier`). MS4 adds :mod:`wfcrc.datasets`
(data-loading/score-provider/loss-table-builder contracts),
:mod:`wfcrc.evaluation.metrics` (realized-risk and statistical utilities),
and :mod:`wfcrc.evaluation.experiment` (a reduced, dataset-free experiment
report combining calibration, verification, and metrics). Visualization and
the full sweep/checkpointing experiment runner remain out of scope and land
in later milestones. This top-level ``__init__`` re-exports only the
version and the shared exception hierarchy; import the modules above
directly for their APIs.
"""

from __future__ import annotations

from wfcrc._version import __version__
from wfcrc.exceptions import (
    CacheError,
    ConfigError,
    FamilyError,
    PreconditionError,
    ReproducibilityError,
    SerializationError,
    SetConstructionError,
    SplitLeakageError,
    VerificationError,
    WFCRCError,
)

__all__ = [
    "CacheError",
    "ConfigError",
    "FamilyError",
    "PreconditionError",
    "ReproducibilityError",
    "SerializationError",
    "SetConstructionError",
    "SplitLeakageError",
    "VerificationError",
    "WFCRCError",
    "__version__",
]
