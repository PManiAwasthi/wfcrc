"""Tests for `wfcrc.baselines.group_conditional.GroupConditionalCRC`."""

from __future__ import annotations

from tests.unit.baselines._helpers import cfg, loss_table, population_losses
from wfcrc.ambiguity.finite_group import FiniteGroupFamily
from wfcrc.baselines.group_conditional import GroupConditionalCRC
from wfcrc.calibration.calibrator import WFCRCCalibrator


def test_group_conditional_reports_a_lambda_per_group_in_diagnostics() -> None:
    values = population_losses(seed=5, n=40)
    table = loss_table(values)
    groups = (tuple(range(0, 20)), tuple(range(20, 40)))
    family = FiniteGroupFamily(masks=groups)

    result = GroupConditionalCRC(family).calibrate(table, cfg(), seed=0)

    per_group = result.diagnostics["per_group"]
    assert len(per_group) == 2
    assert {item["group"] for item in per_group} == {0, 1}
    assert result.lambda_hat == max(item["lambda_hat"] for item in per_group)


def test_group_conditional_per_group_lambda_matches_frozen_finite_group_branch() -> None:
    """Each group's own lambda_hat must match WF-CRC's own per-group computation exactly.

    `WFCRCCalibrator`'s finite-group branch computes the identical
    per-group criterion internally before taking the max; this asserts
    `GroupConditionalCRC` reproduces the same per-group numbers, not just
    a plausible-looking max.
    """
    values = population_losses(seed=5, n=40)
    table = loss_table(values)
    groups = (tuple(range(0, 15)), tuple(range(15, 40)))
    family = FiniteGroupFamily(masks=groups)
    config = cfg()

    frozen_result = WFCRCCalibrator().calibrate(table, family, config, seed=0)
    disclosed_result = GroupConditionalCRC(family).calibrate(table, config, seed=0)

    frozen_per_group = {
        item["group"]: item["lambda_hat"] for item in frozen_result.diagnostics["per_group"]
    }
    disclosed_per_group = {
        item["group"]: item["lambda_hat"] for item in disclosed_result.diagnostics["per_group"]
    }
    assert frozen_per_group == disclosed_per_group
    # WF-CRC's own deployed lambda_hat is the max, matching this baseline's summary value.
    assert frozen_result.lambda_hat == disclosed_result.lambda_hat


def test_group_conditional_is_deterministic() -> None:
    values = population_losses(seed=5, n=30)
    table = loss_table(values)
    family = FiniteGroupFamily(masks=(tuple(range(0, 15)), tuple(range(15, 30))))
    a = GroupConditionalCRC(family).calibrate(table, cfg(), seed=0)
    b = GroupConditionalCRC(family).calibrate(table, cfg(), seed=123)
    assert a.lambda_hat == b.lambda_hat


def test_baseline_name_is_group_conditional() -> None:
    family = FiniteGroupFamily(masks=((0, 1),))
    assert GroupConditionalCRC(family).baseline_name == "group_conditional"
