"""wfcrc: Worst-case Family Conformal Risk Control.

Reproducible research package implementing the frozen WF-CRC specification.

MS1 scope note: this milestone provides only engineering infrastructure
(configuration, logging, numerics, seeding, caching, I/O, exceptions). The
mathematical modules (ambiguity families, losses, calibration, prediction
sets, datasets, evaluation, visualization) are implemented in later
milestones and are intentionally absent from the public API below.
"""

from __future__ import annotations

from wfcrc._version import __version__
from wfcrc.exceptions import (
    CacheError,
    ConfigError,
    ReproducibilityError,
    SerializationError,
    WFCRCError,
)

__all__ = [
    "CacheError",
    "ConfigError",
    "ReproducibilityError",
    "SerializationError",
    "WFCRCError",
    "__version__",
]
