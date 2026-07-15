"""Typed, immutable configuration schema for a wfcrc run.

Every section is a frozen dataclass so a constructed :class:`Config` cannot
be mutated after validation — the only way to get a different configuration
is to load a different one. ``data``/``model``/``sets``/``loss`` sections are
intentionally generic (``name`` + free-form ``params``) because MS1 does not
implement the concrete dataset/model/set/loss registries those names will
resolve against in later milestones; ``family``/``calibration``/``runner``
carry the concrete fields the frozen Mathematical/Algorithm specifications
require MS1 to validate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from wfcrc.exceptions import ConfigError
from wfcrc.utils.io import content_hash

__all__ = [
    "CalibrationConfig",
    "Config",
    "DataConfig",
    "FamilyConfig",
    "FamilyType",
    "LossConfig",
    "ModelConfig",
    "RunnerConfig",
    "SetsConfig",
]

FamilyType = Literal["cvar", "kl", "finite_group", "known_weight"]

#: Family types requiring `Config.family.beta` / `.rho` / `.masks` / `.weights`.
FAMILY_REQUIRED_FIELD: dict[FamilyType, str] = {
    "cvar": "beta",
    "kl": "rho",
    "finite_group": "masks",
    "known_weight": "weights",
}


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and its constructor parameters.

    Attributes:
        name: Registry key of the dataset loader (resolved in a later
            milestone; MS1 only validates that it is a non-empty string).
        params: Free-form, dataset-specific parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """Base model selection and its constructor parameters.

    Attributes:
        name: Registry key of the model/score provider (resolved later).
        params: Free-form, model-specific parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SetsConfig:
    """Prediction-set constructor selection and its parameters.

    Attributes:
        name: Registry key of the ``PredictionSetConstructor`` (resolved later).
        params: Free-form, constructor-specific parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossConfig:
    """Loss evaluator selection and its parameters.

    Attributes:
        name: Registry key of the ``LossEvaluator`` (resolved later).
        params: Free-form, loss-specific parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyConfig:
    """Ambiguity family selection and its parameters.

    Exactly one of ``beta``/``rho``/``masks``/``weights`` is required,
    determined by ``type`` (see :data:`FAMILY_REQUIRED_FIELD`); the others
    are left ``None``. MS1 validates presence and basic ranges only — the
    family's mathematical semantics are implemented in a later milestone.

    Attributes:
        type: Which ambiguity family this configuration selects.
        beta: CVaR tail parameter, required iff ``type == "cvar"``, in ``(0, 1)``.
        rho: KL radius, required iff ``type == "kl"``, must be ``> 0``.
        masks: Group membership masks, required iff ``type == "finite_group"``.
        weights: Known importance weights, required iff ``type == "known_weight"``.
    """

    type: FamilyType
    beta: float | None = None
    rho: float | None = None
    masks: tuple[tuple[int, ...], ...] | None = None
    weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    """Parameters governing the single-split calibration procedure.

    Attributes:
        alpha: Target risk level; must satisfy ``0 < alpha < B``.
        B: Upper bound on the loss; must be ``> 0``.
        pi: Fraction of calibration data assigned to split A (dual
            estimation); must be in ``(0, 1)``.
        lambda_grid: Strictly increasing threshold grid to search over.
    """

    alpha: float
    B: float
    pi: float
    lambda_grid: tuple[float, ...]


@dataclass(frozen=True)
class RunnerConfig:
    """Orchestration-level infrastructure parameters.

    Attributes:
        cache_dir: Directory for content-addressed caches
            (:class:`wfcrc.utils.cache.Cache`).
        log_level: Minimum severity recorded by the run logger
            (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
    """

    cache_dir: str
    log_level: str


@dataclass(frozen=True)
class Config:
    """A fully validated, immutable wfcrc run configuration.

    Constructed only by :func:`wfcrc.config.loader.load_config`, which
    performs layered merging and full schema validation before building
    this object.

    Attributes:
        data: Dataset configuration.
        model: Model configuration.
        sets: Prediction-set constructor configuration.
        loss: Loss evaluator configuration.
        family: Ambiguity family configuration.
        calibration: Calibration procedure configuration.
        runner: Orchestration/infrastructure configuration.
        seed: Global reproducibility seed.
    """

    data: DataConfig
    model: ModelConfig
    sets: SetsConfig
    loss: LossConfig
    family: FamilyConfig
    calibration: CalibrationConfig
    runner: RunnerConfig
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Render this configuration as a plain, JSON/YAML-serializable dict.

        Returns:
            A nested ``dict`` with tuples rendered as lists — suitable for
            :func:`wfcrc.utils.io.content_hash` or :func:`yaml.safe_dump`.
        """
        result = _to_plain(self)
        assert isinstance(result, dict)  # a dataclass always renders to a dict
        return result

    def hash(self) -> str:
        """Compute this configuration's provenance content hash.

        The hash is invariant to field ordering and identical across
        processes for an equal configuration, so it is safe to embed in run
        manifests as a reproducibility fingerprint.

        Returns:
            A stable hex digest string.
        """
        return content_hash(self.to_dict())

    def to_yaml(self) -> str:
        """Render this configuration as canonical (sorted-key) YAML.

        Returns:
            A YAML document that, when loaded via
            :func:`wfcrc.config.loader.load_config`, reproduces an equal
            :class:`Config` (and therefore an equal :meth:`hash`).
        """
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    def get(self, dotted_key: str) -> Any:
        """Look up a value by dotted path, e.g. ``"calibration.alpha"``.

        Args:
            dotted_key: A dot-separated attribute path rooted at this config.

        Returns:
            The value at that path.

        Raises:
            ConfigError: If any segment of ``dotted_key`` does not exist.
        """
        node: Any = self
        for part in dotted_key.split("."):
            if not dataclasses.is_dataclass(node) or part not in {
                f.name for f in dataclasses.fields(node)
            }:
                raise ConfigError(dotted_key, f"no such config path (failed at '{part}')")
            node = getattr(node, part)
        return node


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses/tuples into plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
