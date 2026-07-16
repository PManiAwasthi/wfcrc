"""Unit tests for :mod:`wfcrc.visualization.base`."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wfcrc.visualization.base import FigureFile, FigureSpec, new_figure, render_figure


class TestFigureSpec:
    def test_defaults(self) -> None:
        spec = FigureSpec()
        assert spec.format == "pdf"
        assert spec.dpi == 150
        assert spec.figsize == (6.0, 4.0)

    def test_rejects_bad_format(self) -> None:
        with pytest.raises(ValueError, match="format must be"):
            FigureSpec(format="png")  # type: ignore[arg-type]

    def test_rejects_nonpositive_dpi(self) -> None:
        with pytest.raises(ValueError, match="dpi must be > 0"):
            FigureSpec(dpi=0)


def _draw_minimal(spec: FigureSpec) -> tuple[Figure, Axes]:
    fig, ax = new_figure(spec)
    ax.plot([0, 1], [0, 1])
    return fig, ax


class TestRenderFigure:
    def test_writes_figure_and_sidecar(self, tmp_path: Path) -> None:
        fig, _ = _draw_minimal(FigureSpec())
        result = render_figure(
            fig,
            data={"x": [0, 1], "y": [0, 1]},
            out_dir=tmp_path,
            stem="smoke",
            spec=FigureSpec(),
            sidecar_header=["x", "y"],
            sidecar_rows=[[0, 0], [1, 1]],
        )
        assert isinstance(result, FigureFile)
        assert result.path == tmp_path / "smoke.pdf"
        assert result.sidecar_path == tmp_path / "smoke.csv"
        assert result.path.exists()
        assert result.sidecar_path.exists()
        assert result.path.stat().st_size > 0

    def test_sidecar_contains_header_provenance_and_rows(self, tmp_path: Path) -> None:
        fig, _ = _draw_minimal(FigureSpec())
        result = render_figure(
            fig,
            data={"x": [1, 2], "y": [3, 4]},
            out_dir=tmp_path,
            stem="sidecar_check",
            spec=FigureSpec(),
            sidecar_header=["x", "y"],
            sidecar_rows=[[1, 3], [2, 4]],
        )
        lines = result.sidecar_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == f"# source_hash={result.source_hash}"
        reader = csv.reader(lines[1:])
        rows = list(reader)
        assert rows[0] == ["x", "y"]
        assert rows[1:] == [["1", "3"], ["2", "4"]]

    def test_svg_format_renders(self, tmp_path: Path) -> None:
        fig, _ = _draw_minimal(FigureSpec(format="svg"))
        result = render_figure(
            fig,
            data={"x": [0]},
            out_dir=tmp_path,
            stem="svg_smoke",
            spec=FigureSpec(format="svg"),
            sidecar_header=["x"],
            sidecar_rows=[[0]],
        )
        assert result.path.suffix == ".svg"
        assert result.path.read_bytes().startswith(b"<?xml")

    @pytest.mark.parametrize("fmt", ["pdf", "svg"])
    def test_byte_deterministic_across_calls(self, tmp_path: Path, fmt: str) -> None:
        spec = FigureSpec(format=fmt)  # type: ignore[arg-type]

        def make(stem: str) -> FigureFile:
            fig, ax = new_figure(spec)
            ax.plot([1, 2, 3], [1, 4, 9], label="series")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            return render_figure(
                fig,
                data={"x": [1, 2, 3], "y": [1, 4, 9]},
                out_dir=tmp_path,
                stem=stem,
                spec=spec,
                sidecar_header=["x", "y"],
                sidecar_rows=[[1, 1], [2, 4], [3, 9]],
            )

        a = make(f"det_a_{fmt}")
        b = make(f"det_b_{fmt}")
        assert a.path.read_bytes() == b.path.read_bytes()
        assert a.source_hash == b.source_hash

    def test_creates_missing_out_dir(self, tmp_path: Path) -> None:
        fig, _ = _draw_minimal(FigureSpec())
        nested = tmp_path / "a" / "b" / "c"
        result = render_figure(
            fig,
            data={"x": [0]},
            out_dir=nested,
            stem="nested",
            spec=FigureSpec(),
            sidecar_header=["x"],
            sidecar_rows=[[0]],
        )
        assert result.path.exists()

    def test_rejects_non_serializable_data(self, tmp_path: Path) -> None:
        fig, _ = _draw_minimal(FigureSpec())
        with pytest.raises(TypeError):
            render_figure(
                fig,
                data=object(),
                out_dir=tmp_path,
                stem="bad_data",
                spec=FigureSpec(),
                sidecar_header=["x"],
                sidecar_rows=[[0]],
            )
