"""Structured exception hierarchy for wfcrc.

Every exception raised deliberately by wfcrc code derives from
:class:`WFCRCError`, so callers can catch ``except WFCRCError`` to handle
any project-specific failure without swallowing unrelated bugs (e.g. a
genuine ``TypeError`` from misused third-party code).

MS1 defines the base class plus the concrete exceptions needed by the MS1
infrastructure modules (``utils.io``, ``utils.cache``, ``utils.seeds``,
``config``). Later milestones add further subclasses (e.g. for precondition
or ambiguity-family failures) rather than repurposing these.
"""

from __future__ import annotations


class WFCRCError(Exception):
    """Base class for all exceptions raised intentionally by wfcrc.

    Args:
        message: Human-readable description of the failure.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SerializationError(WFCRCError):
    """Raised when content cannot be canonically serialized, hashed, or
    round-tripped through :mod:`wfcrc.utils.io`.

    Args:
        message: Human-readable description of the failure.
    """


class CacheError(WFCRCError):
    """Raised on cache corruption or read/write failures in
    :mod:`wfcrc.utils.cache` that are not resolved by a recompute.

    Args:
        message: Human-readable description of the failure.
    """


class ReproducibilityError(WFCRCError):
    """Raised when a reproducibility invariant is violated, e.g. an
    out-of-range or non-integer seed passed to :mod:`wfcrc.utils.seeds`.

    Args:
        message: Human-readable description of the failure.
    """


class ConfigError(WFCRCError):
    """Raised when a configuration fails schema validation or range checks.

    Args:
        field: Dotted path of the offending configuration field.
        reason: Human-readable explanation of why the value is invalid.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid config field '{field}': {reason}")
