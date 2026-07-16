"""Unit tests for :mod:`wfcrc.visualization.plots`."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wfcrc.evaluation.metrics import CI
from wfcrc.visualization import plots
from wfcrc.visualization.base import FigureFile, FigureSpec


def _read_sidecar(path: Path) -> tuple[str, list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    provenance = lines[0]
    reader = csv.reader(lines[1:])
    rows = list(reader)
    return provenance, rows[0], rows[1:]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


class TestSplitOutPath:
    def test_bare_stem(self) -> None:
        out_dir, stem = plots._split_out_path("myfig")
        assert out_dir == Path(".")
        assert stem == "myfig"

    def test_with_directory(self, tmp_path: Path) -> None:
        out_dir, stem = plots._split_out_path(tmp_path / "sub" / "myfig")
        assert out_dir == tmp_path / "sub"
        assert stem == "myfig"


class TestValidateSeries:
    def test_empty_x_raises(self) -> None:
        with pytest.raises(ValueError, match="alphas must be non-empty"):
            plots._validate_series([], {"a": []}, x_name="alphas")

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one data series"):
            plots._validate_series([1.0], {}, x_name="x")

    def test_mismatched_series_length_raises(self) -> None:
        with pytest.raises(ValueError, match="has 1 points but x has 2"):
            plots._validate_series([1.0, 2.0], {"a": [1.0]}, x_name="x")

    def test_ok(self) -> None:
        plots._validate_series([1.0, 2.0], {"a": [1.0, 2.0]}, x_name="x")


class TestValidateCiSeries:
    def test_none_is_noop(self) -> None:
        plots._validate_ci_series({"a": [1.0]}, None)

    def test_unknown_series_raises(self) -> None:
        with pytest.raises(ValueError, match="no matching data series"):
            plots._validate_ci_series({"a": [1.0]}, {"b": [CI(0.0, 1.0, 0.95)]})

    def test_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError, match="has 1 entries but its series has 2"):
            plots._validate_ci_series({"a": [1.0, 2.0]}, {"a": [CI(0.0, 1.0, 0.95)]})

    def test_ok(self) -> None:
        plots._validate_ci_series({"a": [1.0]}, {"a": [CI(0.0, 1.0, 0.95)]})


class TestDrawBand:
    def test_skips_when_cis_none(self) -> None:
        fig, ax = plt.subplots()
        plots._draw_band(ax, [1.0, 2.0], None, "blue")
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_skips_single_point(self) -> None:
        fig, ax = plt.subplots()
        plots._draw_band(ax, [1.0], [CI(0.0, 1.0, 0.95)], "blue")
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_draws_band_for_multi_point(self) -> None:
        fig, ax = plt.subplots()
        cis = [CI(0.0, 1.0, 0.95), CI(0.5, 1.5, 0.95)]
        plots._draw_band(ax, [1.0, 2.0], cis, "blue")
        assert len(ax.collections) == 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_g_curve
# ---------------------------------------------------------------------------


class TestGCurve:
    def test_build_labels_and_lines(self) -> None:
        fig, ax, data, header, rows = plots._build_g_curve(
            [0.0, 1.0, 2.0], [0.3, 0.2, 0.1], 0.15, 2.0
        )
        assert ax.get_xlabel() == "λ"
        assert ax.get_ylabel() == "g(λ)"
        assert ax.get_legend() is not None
        assert header == ["lambda", "g"]
        assert rows == [[0.0, 0.3], [1.0, 0.2], [2.0, 0.1]]
        assert data["alpha"] == 0.15
        plt.close(fig)

    def test_empty_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_grid must be non-empty"):
            plots._build_g_curve([], [], 0.1, 0.0)

    def test_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError, match="g_values has 1 points"):
            plots._build_g_curve([0.0, 1.0], [0.3], 0.1, 0.0)

    def test_end_to_end_writes_files(self, tmp_path: Path) -> None:
        result = plots.plot_g_curve(
            [0.0, 1.0, 2.0], [0.3, 0.2, 0.1], 0.15, 2.0, tmp_path / "g_curve"
        )
        assert isinstance(result, FigureFile)
        assert result.path.exists()
        _, header, rows = _read_sidecar(result.sidecar_path)
        assert header == ["lambda", "g"]
        assert rows == [["0.0", "0.3"], ["1.0", "0.2"], ["2.0", "0.1"]]


# ---------------------------------------------------------------------------
# plot_risk_vs_alpha (F1)
# ---------------------------------------------------------------------------


class TestRiskVsAlpha:
    def test_build_multi_family_with_ci(self) -> None:
        alphas = [0.05, 0.1]
        risks = {"cvar": [0.04, 0.09], "kl": [0.03, 0.08]}
        cis = {"cvar": [CI(0.01, 0.07, 0.95), CI(0.05, 0.13, 0.95)]}
        fig, ax, _, header, rows = plots._build_risk_vs_alpha(alphas, risks, cis)
        assert ax.get_xlabel() == "target alpha"
        assert ax.get_legend() is not None
        # target line + 2 family lines = 3 Line2D artists; cvar's band => 1 collection
        assert len(ax.lines) == 3
        assert len(ax.collections) == 1
        assert header == ["family", "alpha", "risk", "ci_lo", "ci_hi"]
        assert len(rows) == 4
        plt.close(fig)

    def test_no_cis(self) -> None:
        fig, _, _, _, rows = plots._build_risk_vs_alpha([0.05], {"cvar": [0.04]}, None)
        assert rows == [["cvar", 0.05, 0.04, "", ""]]
        plt.close(fig)

    def test_empty_alphas_raises(self) -> None:
        with pytest.raises(ValueError, match="alphas must be non-empty"):
            plots._build_risk_vs_alpha([], {"cvar": []}, None)

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_risk_vs_alpha(
            [0.05, 0.1], {"cvar": [0.04, 0.09]}, tmp_path / "f1", spec=FigureSpec(format="svg")
        )
        assert result.path.suffix == ".svg"


# ---------------------------------------------------------------------------
# plot_group_risk (F2)
# ---------------------------------------------------------------------------


class TestGroupRisk:
    def test_build(self) -> None:
        fig, ax, _, header, rows = plots._build_group_risk(
            ["road", "person"], [0.05, 0.2], [0.08, 0.35], 0.1
        )
        assert ax.get_ylabel() == "realized risk"
        assert header == ["group", "wfcrc_risk", "baseline_risk"]
        assert rows == [["road", 0.05, 0.08], ["person", 0.2, 0.35]]
        plt.close(fig)

    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="group_labels must be non-empty"):
            plots._build_group_risk([], [], [], 0.1)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="must match group_labels"):
            plots._build_group_risk(["a", "b"], [0.1], [0.1, 0.2], 0.1)

    def test_many_groups_layout(self) -> None:
        labels = [f"group_{i}" for i in range(20)]
        fig, _, _, _, rows = plots._build_group_risk(labels, [0.1] * 20, [0.2] * 20, 0.15)
        assert len(rows) == 20
        plt.close(fig)

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_group_risk(["a", "b"], [0.1, 0.2], [0.2, 0.3], 0.15, tmp_path / "f2")
        assert result.path.exists()


# ---------------------------------------------------------------------------
# plot_risk_vs_shift (F3)
# ---------------------------------------------------------------------------


class TestRiskVsShift:
    def test_build(self) -> None:
        severities = [1, 2, 3]
        risks = {"wfcrc": [0.05, 0.06, 0.07], "vanilla": [0.05, 0.12, 0.2]}
        fig, ax, _, header, rows = plots._build_risk_vs_shift(severities, risks, 0.1, None)
        assert ax.get_xlabel() == "corruption severity"
        assert header == ["method", "severity", "risk", "ci_lo", "ci_hi"]
        assert len(rows) == 6
        plt.close(fig)

    def test_with_cis(self) -> None:
        severities = [1, 2]
        risks = {"wfcrc": [0.05, 0.06]}
        cis = {"wfcrc": [CI(0.02, 0.08, 0.95), CI(0.03, 0.09, 0.95)]}
        fig, _, _, _, rows = plots._build_risk_vs_shift(severities, risks, 0.1, cis)
        assert rows[0][3] == 0.02
        plt.close(fig)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="severities must be non-empty"):
            plots._build_risk_vs_shift([], {}, 0.1, None)

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_risk_vs_shift(
            [1, 2, 3], {"wfcrc": [0.05, 0.06, 0.07]}, 0.1, tmp_path / "f3"
        )
        assert result.path.exists()


# ---------------------------------------------------------------------------
# plot_set_size (F4)
# ---------------------------------------------------------------------------


class TestSetSize:
    def test_build(self) -> None:
        alphas = [0.05, 0.1]
        sizes = {"marginal": [2.0, 1.5], "wfcrc": [3.0, 2.2], "worst_case_ball": [4.0, 3.0]}
        fig, ax, _, header, rows = plots._build_set_size(alphas, sizes)
        assert ax.get_ylabel() == "mean prediction-set size"
        assert header == ["method", "alpha", "size"]
        assert len(rows) == 6
        plt.close(fig)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="alphas must be non-empty"):
            plots._build_set_size([], {})

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_set_size([0.05], {"wfcrc": [2.0]}, tmp_path / "f4")
        assert result.path.exists()


# ---------------------------------------------------------------------------
# plot_nesting (F5)
# ---------------------------------------------------------------------------


class TestNesting:
    def test_build(self) -> None:
        fig, ax, _, header, rows = plots._build_nesting([0.01, 0.3, 0.5], [0.4, 0.05, 0.0], 0.1)
        assert ax.get_xlabel() == "group mass P(G)"
        assert header == ["group_mass", "divergence"]
        assert rows == [[0.01, 0.4], [0.3, 0.05], [0.5, 0.0]]
        plt.close(fig)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="group_masses must be non-empty"):
            plots._build_nesting([], [], 0.1)

    def test_mismatched_raises(self) -> None:
        with pytest.raises(ValueError, match="divergences has 1 points"):
            plots._build_nesting([0.1, 0.2], [0.1], 0.1)

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_nesting([0.01, 0.5], [0.4, 0.0], 0.1, tmp_path / "f5")
        assert result.path.exists()


# ---------------------------------------------------------------------------
# plot_qualitative (F6)
# ---------------------------------------------------------------------------


class TestQualitative:
    def test_build_without_mask(self) -> None:
        heatmap = np.array([[0.1, 0.2], [0.3, 0.4]])
        fig, ax, data, header, rows = plots._build_qualitative(heatmap, None, "")
        assert ax.get_title() == "F6: qualitative risk heatmap"
        assert header == ["row", "col", "value"]
        assert len(rows) == 4
        assert "overlay_mask" not in data
        plt.close(fig)

    def test_build_with_mask(self) -> None:
        heatmap = np.array([[0.1, 0.2], [0.3, 0.4]])
        mask = np.array([[True, False], [False, True]])
        fig, ax, data, header, rows = plots._build_qualitative(heatmap, mask, "custom title")
        assert ax.get_title() == "custom title"
        assert header == ["row", "col", "value", "mask"]
        assert data["overlay_mask"] == mask.tolist()
        # row 0 col 0 -> mask True
        assert next(r for r in rows if r[0] == 0 and r[1] == 0)[3] is True
        plt.close(fig)

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D array"):
            plots._build_qualitative(np.array([1.0, 2.0]), None, "")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D array"):
            plots._build_qualitative(np.zeros((0, 0)), None, "")

    def test_mask_shape_mismatch_raises(self) -> None:
        heatmap = np.zeros((2, 2))
        mask = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="does not match heatmap shape"):
            plots._build_qualitative(heatmap, mask, "")

    def test_end_to_end(self, tmp_path: Path) -> None:
        heatmap = np.array([[0.1, 0.9], [0.5, 0.2]])
        result = plots.plot_qualitative(heatmap, tmp_path / "f6")
        assert result.path.exists()


# ---------------------------------------------------------------------------
# plot_runtime (F7)
# ---------------------------------------------------------------------------


class TestRuntime:
    def test_build_default_log_scales(self) -> None:
        sizes = [10, 100, 1000]
        values = {"wfcrc": [0.01, 0.1, 1.0]}
        fig, ax, _, header, rows = plots._build_runtime(sizes, values, "wall-clock (s)", True, True)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        assert ax.get_ylabel() == "wall-clock (s)"
        assert header == ["method", "size", "value"]
        assert len(rows) == 3
        plt.close(fig)

    def test_build_linear_scales(self) -> None:
        fig, ax, _, _, _ = plots._build_runtime([1, 2], {"a": [1.0, 2.0]}, "y", False, False)
        assert ax.get_xscale() == "linear"
        assert ax.get_yscale() == "linear"
        plt.close(fig)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="sizes must be non-empty"):
            plots._build_runtime([], {}, "y", True, True)

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_runtime([10, 100], {"wfcrc": [0.01, 0.1]}, tmp_path / "f7")
        assert result.path.exists()


# ---------------------------------------------------------------------------
# plot_reliability (F8)
# ---------------------------------------------------------------------------


class TestReliability:
    def test_build_with_cis(self) -> None:
        targets = [0.05, 0.1]
        observed = [0.04, 0.09]
        cis = [CI(0.01, 0.07, 0.95), CI(0.05, 0.13, 0.95)]
        fig, ax, _, header, rows = plots._build_reliability(targets, observed, cis)
        assert ax.get_xlabel() == "target risk"
        assert header == ["target", "observed", "ci_lo", "ci_hi"]
        assert rows[0][2] == 0.01
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_build_without_cis(self) -> None:
        fig, _, _, _, rows = plots._build_reliability([0.05], [0.04], None)
        assert rows == [[0.05, 0.04, "", ""]]
        plt.close(fig)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="target_risks must be non-empty"):
            plots._build_reliability([], [], None)

    def test_mismatched_observed_raises(self) -> None:
        with pytest.raises(ValueError, match="observed_risks has 1 points"):
            plots._build_reliability([0.05, 0.1], [0.04], None)

    def test_mismatched_cis_raises(self) -> None:
        with pytest.raises(ValueError, match="cis has 1 entries"):
            plots._build_reliability([0.05, 0.1], [0.04, 0.09], [CI(0.0, 0.1, 0.95)])

    def test_end_to_end(self, tmp_path: Path) -> None:
        result = plots.plot_reliability([0.05, 0.1], [0.04, 0.09], tmp_path / "f8")
        assert result.path.exists()
