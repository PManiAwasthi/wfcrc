"""Unit tests for :mod:`wfcrc.config.loader`."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from wfcrc.config.loader import load_config
from wfcrc.exceptions import ConfigError


def _base_dict() -> dict[str, Any]:
    return {
        "seed": 0,
        "data": {"name": "toy_dataset", "params": {}},
        "model": {"name": "toy_model", "params": {}},
        "sets": {"name": "toy_sets", "params": {}},
        "loss": {"name": "toy_loss", "params": {}},
        "family": {"type": "cvar", "beta": 0.1},
        "calibration": {"alpha": 0.1, "B": 1.0, "pi": 0.5, "lambda_grid": [0.0, 0.5, 1.0]},
        "runner": {"cache_dir": "cache", "log_level": "INFO"},
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "config.yaml", _base_dict())
    config = load_config([path])
    assert config.seed == 0
    assert config.family.type == "cvar"
    assert config.family.beta == pytest.approx(0.1)
    assert config.calibration.alpha == pytest.approx(0.1)
    assert config.calibration.lambda_grid == (0.0, 0.5, 1.0)


@pytest.mark.parametrize(
    ("mutation", "expected_field_substr"),
    [
        (lambda d: d["calibration"].update(alpha=1.0, B=1.0), "calibration.alpha"),
        (lambda d: d["calibration"].update(pi=0.0), "calibration.pi"),
        (lambda d: d["calibration"].update(pi=1.0), "calibration.pi"),
        (lambda d: d["family"].update(beta=0.0), "family.beta"),
        (lambda d: d["family"].update(beta=1.0), "family.beta"),
        (lambda d: d.__setitem__("family", {"type": "kl", "rho": 0.0}), "family.rho"),
        (lambda d: d.__setitem__("family", {"type": "kl", "rho": -1.0}), "family.rho"),
        (lambda d: d["calibration"].update(unknown_field=1), "calibration.unknown_field"),
        (lambda d: d.__setitem__("bogus_top_level", 1), "bogus_top_level"),
    ],
)
def test_invalid_configs_are_rejected(
    tmp_path: Path, mutation: Any, expected_field_substr: str
) -> None:
    data = _base_dict()
    mutation(data)
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError) as exc_info:
        load_config([path])
    assert expected_field_substr in exc_info.value.field


def test_missing_required_field_raises(tmp_path: Path) -> None:
    data = _base_dict()
    del data["calibration"]["alpha"]
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError) as exc_info:
        load_config([path])
    assert exc_info.value.field == "calibration.alpha"
    assert "missing" in exc_info.value.reason


def test_missing_top_level_section_raises(tmp_path: Path) -> None:
    data = _base_dict()
    del data["runner"]
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_string_numeric_coercion_is_rejected(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["alpha"] = "0.1"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError) as exc_info:
        load_config([path])
    assert exc_info.value.field == "calibration.alpha"


def test_string_seed_is_rejected(tmp_path: Path) -> None:
    data = _base_dict()
    data["seed"] = "0"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_negative_seed_is_rejected(tmp_path: Path) -> None:
    data = _base_dict()
    data["seed"] = -1
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_non_monotone_lambda_grid_is_rejected(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["lambda_grid"] = [0.5, 0.1, 1.0]
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_unknown_family_type_is_rejected(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "not_a_family", "beta": 0.1}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError) as exc_info:
        load_config([path])
    assert exc_info.value.field == "family.type"


def test_layering_precedence_later_file_wins(tmp_path: Path) -> None:
    base = _write_yaml(tmp_path / "base.yaml", _base_dict())
    override = _write_yaml(tmp_path / "override.yaml", {"seed": 99})

    config = load_config([base, override])
    assert config.seed == 99


def test_layering_deep_merges_nested_sections(tmp_path: Path) -> None:
    base = _write_yaml(tmp_path / "base.yaml", _base_dict())
    override = _write_yaml(tmp_path / "override.yaml", {"calibration": {"alpha": 0.05}})

    config = load_config([base, override])
    assert config.calibration.alpha == pytest.approx(0.05)
    # Untouched calibration fields from the base layer must survive the merge.
    assert pytest.approx(1.0) == config.calibration.B
    assert config.calibration.pi == pytest.approx(0.5)


def test_cli_overrides_win_over_all_file_layers(tmp_path: Path) -> None:
    base = _write_yaml(tmp_path / "base.yaml", _base_dict())
    override = _write_yaml(tmp_path / "override.yaml", {"calibration": {"alpha": 0.05}})

    config = load_config([base, override], overrides={"calibration.alpha": 0.02})
    assert config.calibration.alpha == pytest.approx(0.02)


def test_load_config_does_not_mutate_caller_dicts(tmp_path: Path) -> None:
    data = _base_dict()
    frozen_copy = copy.deepcopy(data)
    path = _write_yaml(tmp_path / "config.yaml", data)
    load_config([path])
    assert data == frozen_copy


def test_kl_family_valid_rho_loads(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "kl", "rho": 0.5}
    path = _write_yaml(tmp_path / "config.yaml", data)
    config = load_config([path])
    assert config.family.type == "kl"
    assert config.family.rho == pytest.approx(0.5)


def test_finite_group_family_requires_masks(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "finite_group", "masks": [[0, 1], [2, 3]]}
    path = _write_yaml(tmp_path / "config.yaml", data)
    config = load_config([path])
    assert config.family.masks == ((0, 1), (2, 3))


def test_known_weight_family_requires_weights(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "known_weight", "weights": [0.5, 1.5, 2.0]}
    path = _write_yaml(tmp_path / "config.yaml", data)
    config = load_config([path])
    assert config.family.weights == (0.5, 1.5, 2.0)


def test_top_level_document_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config([path])


def test_dotted_override_into_non_mapping_raises(tmp_path: Path) -> None:
    data = _base_dict()
    data["seed"] = 0
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path], overrides={"seed.nested": 1})


@pytest.mark.parametrize(
    ("section", "bad_value"),
    [
        ("data", "not_a_mapping"),
        ("family", "not_a_mapping"),
        ("calibration", "not_a_mapping"),
        ("runner", "not_a_mapping"),
    ],
)
def test_section_must_be_a_mapping(tmp_path: Path, section: str, bad_value: Any) -> None:
    data = _base_dict()
    data[section] = bad_value
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_generic_section_name_must_be_non_empty_string(tmp_path: Path) -> None:
    data = _base_dict()
    data["data"]["name"] = ""
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_generic_section_params_must_be_a_mapping(tmp_path: Path) -> None:
    data = _base_dict()
    data["data"]["params"] = "not_a_mapping"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_family_beta_wrong_type_raises(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "cvar", "beta": "0.1"}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_family_rho_wrong_type_raises(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "kl", "rho": "1.0"}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_family_masks_must_be_list_of_lists_of_ints(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "finite_group", "masks": [["a", "b"]]}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_family_masks_must_be_non_empty(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "finite_group", "masks": []}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_family_weights_must_be_non_empty(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "known_weight", "weights": []}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_family_weights_must_be_numbers(tmp_path: Path) -> None:
    data = _base_dict()
    data["family"] = {"type": "known_weight", "weights": ["a", "b"]}
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_calibration_b_wrong_type_raises(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["B"] = "1.0"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_calibration_b_must_be_positive(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["B"] = 0.0
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_calibration_pi_wrong_type_raises(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["pi"] = "0.5"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_calibration_lambda_grid_must_be_a_list(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["lambda_grid"] = "not_a_list"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_calibration_lambda_grid_must_be_non_empty(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["lambda_grid"] = []
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_calibration_lambda_grid_must_contain_numbers(tmp_path: Path) -> None:
    data = _base_dict()
    data["calibration"]["lambda_grid"] = ["a", "b"]
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_runner_cache_dir_must_be_non_empty_string(tmp_path: Path) -> None:
    data = _base_dict()
    data["runner"]["cache_dir"] = ""
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])


def test_runner_log_level_must_be_valid(tmp_path: Path) -> None:
    data = _base_dict()
    data["runner"]["log_level"] = "TRACE"
    path = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError):
        load_config([path])
