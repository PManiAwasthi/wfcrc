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


class PreconditionError(WFCRCError):
    """Raised when a required mathematical precondition is violated.

    Per the Implementation Blueprint's failure-mode table (§19), this covers
    ``NonMonotoneLoss`` (a loss that is not non-increasing in ``λ``, violating
    A2/P-2) and ``NonNestedSets`` (a prediction-set family that is not
    nested, violating P-1). These are abort conditions: the assumption the
    rest of WF-CRC depends on does not hold for the given inputs.

    Args:
        message: Human-readable description of the violated precondition.
    """


class FamilyError(WFCRCError):
    """Raised when an ambiguity family cannot be used as specified.

    Per the Implementation Blueprint's failure-mode table (§19), this covers
    ``UnboundedTransform`` (a dual-transformed loss with no finite upper
    bound, `B̃=∞`) and ``UnsupportedFamily`` (a family type outside the
    frozen supported set: CVaR, KL, finite-group, known-weight).

    Args:
        message: Human-readable description of the family failure.
    """


class SetConstructionError(WFCRCError):
    """Raised when a `PredictionSetConstructor` cannot build `C_λ` as specified.

    This is the `prediction_sets` analogue of :class:`FamilyError`: it
    covers a configuration that names a real, frozen-spec construction
    knob (e.g. `MorphologicalSets`' ``direction``) for which no concrete
    formula exists in the Research Vault, so honoring it would mean
    inventing behavior rather than implementing a frozen one (see
    :mod:`wfcrc.prediction_sets.segmentation`'s module docstring).

    Args:
        message: Human-readable description of the construction failure.
    """


class VerificationError(WFCRCError):
    """Raised by ``VerificationReport.assert_ok()`` when a check failed.

    Per the Implementation Blueprint's `verify.Verifier` (§6) and the MS4
    Implementation Specification (C1): the `Verifier`'s ``check_*`` methods
    always run every applicable check and collect the results
    (non-strict); calling :meth:`~wfcrc.evaluation.verifier.VerificationReport.assert_ok`
    on the resulting report is the strict gate that raises this error,
    naming every failing check, if any check did not pass.

    Args:
        message: Human-readable description naming the failing check(s).
    """


class SplitLeakageError(WFCRCError):
    """Raised when dataset splits (train/calibration/test) overlap.

    Per the MS2 Implementation Specification (`data.loaders`, C1 item 8)
    and the Algorithm Specification's A1 hygiene requirement: calibration
    data must never be used for training or model selection, and test data
    must never leak into calibration. This is the dataset-loading analogue
    of :class:`PreconditionError` — an abort condition, not a warning.

    Args:
        message: Human-readable description of the overlap detected.
    """


class PreprocessingError(WFCRCError):
    """Raised for invalid inputs to :mod:`wfcrc.datasets.preprocessing`/
    :mod:`wfcrc.datasets.corruptions` (MS6.2).

    Covers invalid resize/normalize/resample parameters (non-positive
    target sizes or spacings, mismatched mean/std channel counts, wrong
    array rank) and invalid corruption requests (an unknown corruption
    name, an out-of-range severity, or an image not in the raw
    ``uint8`` pixel format the corruption suite requires) — the
    ``wfcrc.datasets`` preprocessing analogue of :class:`SetConstructionError`.

    Args:
        message: Human-readable description of the invalid input.
    """


class RunnerError(WFCRCError):
    """Raised for `wfcrc.runner`-specific failures with no more specific exception.

    Per the MS5 Implementation Specification (C2 item 8): covers conditions
    that are specific to orchestration itself, not to any lower module —
    e.g. `resume()` called on a directory with no resumable run artifacts,
    or a sweep-grid parameter that is invalid before it ever reaches
    `wfcrc.config`/`wfcrc.calibration`. Distinct from
    :class:`VerificationError` (raised when the verify STOP-gate itself
    fails, propagated unchanged rather than wrapped, so a caller can still
    catch it specifically).

    Args:
        message: Human-readable description of the failure.
    """


class BaselineError(WFCRCError):
    """Raised for `wfcrc.baselines`-specific failures with no more specific exception.

    Per the MS9 milestone (comparator-baseline implementations behind the
    common :class:`~wfcrc.baselines.base.Calibrator` interface): covers
    conditions specific to a baseline's own construction or calibration
    procedure that are not already one of the frozen exceptions above (e.g.
    an unknown name looked up in :data:`wfcrc.baselines.base.BASELINES`, or
    a baseline-specific parameter precondition such as an out-of-range
    `k_folds` or a fixed dual parameter of the wrong shape for the paired
    family). Distinct from :class:`FamilyError` (raised by the frozen
    `wfcrc.ambiguity`/`wfcrc.calibration` machinery a baseline may compose)
    and from :class:`ConfigError` (frozen config-schema validation) — this
    is the `wfcrc.baselines` analogue of :class:`RunnerError`, added per
    the same sanctioned additive-append pattern (new subclass only, no
    existing exception modified).

    Args:
        message: Human-readable description of the failure.
    """
