"""Unit tests for :mod:`wfcrc.config.schema` (Config.hash/to_yaml/get)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from wfcrc.config.loader import load_config
from wfcrc.config.schema import (
    CalibrationConfig,
    Config,
    DataConfig,
    FamilyConfig,
    LossConfig,
    ModelConfig,
    RunnerConfig,
    SetsConfig,
)
from wfcrc.exceptions import ConfigError


def _make_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "data": DataConfig(name="toy_dataset", params={}),
        "model": ModelConfig(name="toy_model", params={}),
        "sets": SetsConfig(name="toy_sets", params={}),
        "loss": LossConfig(name="toy_loss", params={}),
        "family": FamilyConfig(type="cvar", beta=0.1),
        "calibration": CalibrationConfig(alpha=0.1, B=1.0, pi=0.5, lambda_grid=(0.0, 0.5, 1.0)),
        "runner": RunnerConfig(cache_dir="cache", log_level="INFO"),
        "seed": 0,
    }
    defaults.update(overrides)
    return Config(**defaults)


def test_hash_is_deterministic() -> None:
    config = _make_config()
    assert config.hash() == config.hash()


def test_hash_differs_for_different_configs() -> None:
    a = _make_config(seed=0)
    b = _make_config(seed=1)
    assert a.hash() != b.hash()


def test_hash_is_key_order_invariant_via_dict_construction() -> None:
    # Two independently constructed but semantically identical configs must
    # hash identically, regardless of the order fields were assembled in.
    config_a = _make_config(
        family=FamilyConfig(type="cvar", beta=0.1),
        calibration=CalibrationConfig(alpha=0.1, B=1.0, pi=0.5, lambda_grid=(0.0, 0.5, 1.0)),
    )
    config_b = _make_config(
        calibration=CalibrationConfig(alpha=0.1, B=1.0, pi=0.5, lambda_grid=(0.0, 0.5, 1.0)),
        family=FamilyConfig(type="cvar", beta=0.1),
    )
    assert config_a.hash() == config_b.hash()


def test_hash_stable_across_processes_is_pure_function_of_content() -> None:
    # Same content hashed twice from scratch (simulating two processes)
    # must match; this is the closest MS1-scope proxy for a cross-process
    # determinism gate without spawning a subprocess.
    dict_repr = _make_config().to_dict()
    from wfcrc.utils.io import content_hash

    assert content_hash(dict_repr) == _make_config().hash()


def test_to_yaml_round_trips_through_load_config(tmp_path: Path) -> None:
    config = _make_config()
    yaml_text = config.to_yaml()
    path = tmp_path / "roundtrip.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    reloaded = load_config([path])
    assert reloaded.hash() == config.hash()


def test_to_yaml_is_valid_yaml() -> None:
    config = _make_config()
    parsed = yaml.safe_load(config.to_yaml())
    assert parsed["seed"] == 0
    assert parsed["calibration"]["alpha"] == pytest.approx(0.1)


def test_get_dotted_key() -> None:
    config = _make_config()
    assert config.get("calibration.alpha") == pytest.approx(0.1)
    assert config.get("family.type") == "cvar"
    assert config.get("seed") == 0


def test_get_invalid_path_raises() -> None:
    config = _make_config()
    with pytest.raises(ConfigError):
        config.get("calibration.not_a_field")
    with pytest.raises(ConfigError):
        config.get("not_a_section.alpha")


def test_to_dict_renders_tuples_as_lists() -> None:
    config = _make_config()
    d = config.to_dict()
    assert isinstance(d["calibration"]["lambda_grid"], list)
    assert d["calibration"]["lambda_grid"] == [0.0, 0.5, 1.0]
