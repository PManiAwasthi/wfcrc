"""Layered YAML configuration loading, merging, and strict validation.

Layering precedence (later wins): ``default.yaml`` ← ``dataset_*.yaml`` ←
``family_*.yaml`` ← ``experiment_*.yaml`` ← CLI ``overrides``. Every field is
validated against the schema in :mod:`wfcrc.config.schema`; unknown keys,
missing required fields, out-of-range values, and numeric-looking strings
are all rejected rather than silently coerced.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from wfcrc.config.schema import (
    FAMILY_REQUIRED_FIELD,
    CalibrationConfig,
    Config,
    DataConfig,
    FamilyConfig,
    FamilyType,
    LossConfig,
    ModelConfig,
    RunnerConfig,
    SetsConfig,
)
from wfcrc.constants import TEXT_ENCODING
from wfcrc.exceptions import ConfigError

__all__ = ["load_config"]

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
_VALID_FAMILY_TYPES = frozenset(FAMILY_REQUIRED_FIELD)

_TOP_LEVEL_SECTIONS = frozenset(
    {"data", "model", "sets", "loss", "family", "calibration", "runner", "seed"}
)
_GENERIC_SECTION_KEYS = frozenset({"name", "params"})
_FAMILY_KEYS = frozenset({"type", "beta", "rho", "masks", "weights"})
_CALIBRATION_KEYS = frozenset({"alpha", "B", "pi", "lambda_grid"})
_RUNNER_KEYS = frozenset({"cache_dir", "log_level"})


def _is_number(value: Any) -> bool:
    """True iff ``value`` is a real ``int``/``float`` (never a ``bool`` or string).

    Rejecting strings here is what enforces the "string-numeric coercion
    rejected" safety rule: a YAML value like ``alpha: "0.1"`` must fail
    validation rather than being silently parsed as a float.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, overlay values winning.

    Args:
        base: The lower-precedence mapping (mutated and returned).
        overlay: The higher-precedence mapping.

    Returns:
        ``base``, updated in place.
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_dotted_overrides(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply flat dotted-key CLI overrides, e.g. ``{"calibration.alpha": 0.05}``.

    Args:
        base: The mapping to update (mutated and returned).
        overrides: Dotted-key -> value overrides, highest precedence.

    Returns:
        ``base``, updated in place.
    """
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = base
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(dotted_key, f"cannot override into non-mapping at '{part}'")
        node[parts[-1]] = value
    return base


def _require_keys_subset(mapping: Mapping[str, Any], allowed: frozenset[str], prefix: str) -> None:
    """Raise :class:`ConfigError` if ``mapping`` has any key outside ``allowed``."""
    for key in mapping:
        if key not in allowed:
            raise ConfigError(f"{prefix}.{key}", "unknown key")


def _require_present(mapping: Mapping[str, Any], key: str, prefix: str) -> Any:
    """Raise :class:`ConfigError` if ``key`` is missing from ``mapping``, else return it."""
    if key not in mapping:
        raise ConfigError(f"{prefix}.{key}", "missing required field")
    return mapping[key]


def _parse_generic_section(
    mapping: Any, prefix: str, cls: type[DataConfig | ModelConfig | SetsConfig | LossConfig]
) -> Any:
    """Parse a ``{name, params}`` section shared by data/model/sets/loss."""
    if not isinstance(mapping, Mapping):
        raise ConfigError(prefix, "must be a mapping")
    _require_keys_subset(mapping, _GENERIC_SECTION_KEYS, prefix)
    name = _require_present(mapping, "name", prefix)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{prefix}.name", "must be a non-empty string")
    params = mapping.get("params", {})
    if not isinstance(params, Mapping):
        raise ConfigError(f"{prefix}.params", "must be a mapping")
    return cls(name=name, params=dict(params))


def _parse_family(mapping: Any) -> FamilyConfig:
    """Parse and validate the ``family`` section."""
    prefix = "family"
    if not isinstance(mapping, Mapping):
        raise ConfigError(prefix, "must be a mapping")
    _require_keys_subset(mapping, _FAMILY_KEYS, prefix)

    family_type = _require_present(mapping, "type", prefix)
    if family_type not in _VALID_FAMILY_TYPES:
        raise ConfigError(f"{prefix}.type", f"must be one of {sorted(_VALID_FAMILY_TYPES)}")
    family_type_literal: FamilyType = family_type

    required_field = FAMILY_REQUIRED_FIELD[family_type_literal]
    value = _require_present(mapping, required_field, f"{prefix}.{family_type}")

    beta: float | None = None
    rho: float | None = None
    masks: tuple[tuple[int, ...], ...] | None = None
    weights: tuple[float, ...] | None = None

    if required_field == "beta":
        if not _is_number(value):
            raise ConfigError(f"{prefix}.beta", "must be a number")
        if not (0.0 < value < 1.0):
            raise ConfigError(f"{prefix}.beta", "must satisfy 0 < beta < 1")
        beta = float(value)
    elif required_field == "rho":
        if not _is_number(value):
            raise ConfigError(f"{prefix}.rho", "must be a number")
        if not (value > 0.0):
            raise ConfigError(f"{prefix}.rho", "must satisfy rho > 0")
        rho = float(value)
    elif required_field == "masks":
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{prefix}.masks", "must be a non-empty list of index lists")
        try:
            masks = tuple(tuple(int(i) for i in group) for group in value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{prefix}.masks", "must be a list of lists of ints") from exc
    else:  # required_field == "weights"
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{prefix}.weights", "must be a non-empty list of numbers")
        if not all(_is_number(w) for w in value):
            raise ConfigError(f"{prefix}.weights", "must be a list of numbers")
        weights = tuple(float(w) for w in value)

    return FamilyConfig(type=family_type_literal, beta=beta, rho=rho, masks=masks, weights=weights)


def _parse_calibration(mapping: Any) -> CalibrationConfig:
    """Parse and validate the ``calibration`` section."""
    prefix = "calibration"
    if not isinstance(mapping, Mapping):
        raise ConfigError(prefix, "must be a mapping")
    _require_keys_subset(mapping, _CALIBRATION_KEYS, prefix)

    b_value = _require_present(mapping, "B", prefix)
    if not _is_number(b_value):
        raise ConfigError(f"{prefix}.B", "must be a number")
    if not (b_value > 0.0):
        raise ConfigError(f"{prefix}.B", "must satisfy B > 0")

    alpha_value = _require_present(mapping, "alpha", prefix)
    if not _is_number(alpha_value):
        raise ConfigError(f"{prefix}.alpha", "must be a number")
    if not (0.0 < alpha_value < b_value):
        raise ConfigError(f"{prefix}.alpha", "must satisfy 0 < alpha < B")

    pi_value = _require_present(mapping, "pi", prefix)
    if not _is_number(pi_value):
        raise ConfigError(f"{prefix}.pi", "must be a number")
    if not (0.0 < pi_value < 1.0):
        raise ConfigError(f"{prefix}.pi", "must satisfy 0 < pi < 1")

    grid_value = _require_present(mapping, "lambda_grid", prefix)
    if not isinstance(grid_value, list) or not grid_value:
        raise ConfigError(f"{prefix}.lambda_grid", "must be a non-empty list of numbers")
    if not all(_is_number(v) for v in grid_value):
        raise ConfigError(f"{prefix}.lambda_grid", "must be a list of numbers")
    grid = tuple(float(v) for v in grid_value)
    if list(grid) != sorted(set(grid)):
        raise ConfigError(f"{prefix}.lambda_grid", "must be strictly increasing with no duplicates")

    return CalibrationConfig(
        alpha=float(alpha_value), B=float(b_value), pi=float(pi_value), lambda_grid=grid
    )


def _parse_runner(mapping: Any) -> RunnerConfig:
    """Parse and validate the ``runner`` section."""
    prefix = "runner"
    if not isinstance(mapping, Mapping):
        raise ConfigError(prefix, "must be a mapping")
    _require_keys_subset(mapping, _RUNNER_KEYS, prefix)

    cache_dir = _require_present(mapping, "cache_dir", prefix)
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ConfigError(f"{prefix}.cache_dir", "must be a non-empty string")

    log_level = _require_present(mapping, "log_level", prefix)
    if not isinstance(log_level, str) or log_level not in _VALID_LOG_LEVELS:
        raise ConfigError(f"{prefix}.log_level", f"must be one of {sorted(_VALID_LOG_LEVELS)}")

    return RunnerConfig(cache_dir=cache_dir, log_level=log_level)


def _parse_seed(mapping: Mapping[str, Any]) -> int:
    """Parse and validate the top-level ``seed`` field."""
    seed = _require_present(mapping, "seed", "")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("seed", "must be an int")
    if seed < 0:
        raise ConfigError("seed", "must be >= 0")
    return seed


def _parse_config(merged: Mapping[str, Any]) -> Config:
    """Validate a fully merged mapping and construct the immutable :class:`Config`."""
    _require_keys_subset(merged, _TOP_LEVEL_SECTIONS, "")

    return Config(
        data=_parse_generic_section(_require_present(merged, "data", ""), "data", DataConfig),
        model=_parse_generic_section(_require_present(merged, "model", ""), "model", ModelConfig),
        sets=_parse_generic_section(_require_present(merged, "sets", ""), "sets", SetsConfig),
        loss=_parse_generic_section(_require_present(merged, "loss", ""), "loss", LossConfig),
        family=_parse_family(_require_present(merged, "family", "")),
        calibration=_parse_calibration(_require_present(merged, "calibration", "")),
        runner=_parse_runner(_require_present(merged, "runner", "")),
        seed=_parse_seed(merged),
    )


def load_config(paths: list[str | Path], overrides: Mapping[str, Any] | None = None) -> Config:
    """Load, layer-merge, and validate a wfcrc configuration.

    Args:
        paths: YAML file paths in increasing precedence order (e.g.
            ``[default.yaml, dataset_x.yaml, family_kl.yaml, experiment_1.yaml]``);
            later files override earlier ones on conflicting keys.
        overrides: Optional flat dotted-key overrides applied last, e.g.
            ``{"calibration.alpha": 0.05}``.

    Returns:
        A fully validated, immutable :class:`~wfcrc.config.schema.Config`.

    Raises:
        ConfigError: If any layer fails schema validation (unknown key,
            missing required field, out-of-range value, or a numeric field
            given as a string).
    """
    merged: dict[str, Any] = {}
    for path in paths:
        text = Path(path).read_text(encoding=TEXT_ENCODING)
        layer = yaml.safe_load(text) or {}
        if not isinstance(layer, Mapping):
            raise ConfigError(str(path), "top-level YAML document must be a mapping")
        _deep_merge(merged, layer)

    if overrides:
        _apply_dotted_overrides(merged, overrides)

    return _parse_config(merged)
