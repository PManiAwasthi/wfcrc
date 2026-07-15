"""Project-wide constants shared across MS1 infrastructure modules.

Centralizing these values avoids magic numbers/strings drifting between
:mod:`wfcrc.utils.io`, :mod:`wfcrc.utils.cache`, :mod:`wfcrc.utils.numerics`,
:mod:`wfcrc.utils.logging`, and :mod:`wfcrc.config`. Values here are defaults
only; every one of them is overridable through the configuration system.
"""

from __future__ import annotations

from typing import Final

# --- utils.io / utils.cache ------------------------------------------------

#: Default content-hash algorithm (must be a name accepted by ``hashlib.new``).
DEFAULT_HASH_ALGO: Final[str] = "sha256"

#: Default number of leading hex characters kept from a full digest. Used for
#: short, human-facing hashes (e.g. :meth:`wfcrc.config.schema.Config.hash`)
#: where brevity matters more than astronomical collision resistance.
DEFAULT_HASH_WIDTH: Final[int] = 16

#: Hex digest width for cache keys (:func:`wfcrc.utils.cache.make_key`) — the
#: full, untruncated SHA-256 digest (64 hex chars = 256 bits). Cache keys
#: accumulate across many long-running research sweeps (scores, loss tables,
#: dual estimates), where a truncated width's collision probability is no
#: longer negligible; a key collision there would silently serve a wrong
#: cached value, so this stays untruncated rather than reusing
#: ``DEFAULT_HASH_WIDTH``.
CACHE_KEY_HASH_WIDTH: Final[int] = 64

#: Default directory (relative to a run/working directory) for cached artifacts.
DEFAULT_CACHE_DIR: Final[str] = "cache"

#: File suffix used for atomic-write temporary files before rename.
ATOMIC_TMP_SUFFIX: Final[str] = ".tmp"

# --- utils.numerics ----------------------------------------------------------

#: Canonical floating-point dtype for all numerical computation in wfcrc.
DEFAULT_FLOAT_DTYPE: Final[str] = "float64"

#: Default interpolation method for :func:`wfcrc.utils.numerics.quantile`.
DEFAULT_QUANTILE_METHOD: Final[str] = "linear"

#: Default lower clamp bound used as a stand-in for the algorithm's ``eta_min``.
DEFAULT_ETA_MIN: Final[float] = 1e-12

# --- utils.logging -------------------------------------------------------------

#: Default logging level name (see :mod:`logging`).
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

#: Default directory (relative to a run directory) for JSONL log files.
DEFAULT_LOG_DIR: Final[str] = "logs"

#: Filename for the per-run structured JSONL event log.
LOG_FILENAME: Final[str] = "run.jsonl"

# --- config --------------------------------------------------------------------

#: Filename convention for the base configuration layer.
DEFAULT_CONFIG_LAYER: Final[str] = "default.yaml"

#: Text encoding used for all config/log/text I/O in the project.
TEXT_ENCODING: Final[str] = "utf-8"

# --- reproducibility -------------------------------------------------------------

#: Minimum allowed global seed value (inclusive).
MIN_SEED: Final[int] = 0

#: Maximum allowed global seed value (inclusive); keeps derived seeds within
#: the 32-bit range expected by ``numpy.random.SeedSequence``.
MAX_SEED: Final[int] = 2**32 - 1
