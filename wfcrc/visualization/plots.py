"""``Plotter`` free functions — the paper's figures F1-F8 (+ `g`-curve) (M14).

Per the Implementation Blueprint (§6, §14) and the MS5 Implementation
Specification (C1): each function renders one figure **deterministically
from a metrics manifest** (already-computed numbers — means, CIs, per-group
values — never raw datasets/models) to a vector file (`.pdf`/`.svg`) plus a
CSV data sidecar that is exactly the data the figure was drawn from (see
:mod:`wfcrc.visualization.base`). No function in this module loads a
dataset, runs a model, or re-executes any part of the calibration
procedure — that dependency-free, metrics-only design is what lets a
camera-ready figure regenerate byte-identically without an algorithm re-run
(MS5 spec C1 item 1, item 12).

Each public `plot_*` function is a thin wrapper around a private
`_build_*` function that constructs and returns the `(Figure, Axes, ...)`
before saving — the private builders are what the unit tests exercise to
assert on axis labels/legends/reference lines/CI bands without needing to
parse a rendered PDF/SVG back apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wfcrc.evaluation.metrics import CI
from wfcrc.visualization.base import FigureFile, FigureSpec, new_figure, render_figure

__all__ = [
    "plot_g_curve",
    "plot_group_risk",
    "plot_nesting",
    "plot_qualitative",
    "plot_reliability",
    "plot_risk_vs_alpha",
    "plot_risk_vs_shift",
    "plot_runtime",
    "plot_set_size",
]


def _resolve_spec(spec: FigureSpec | None) -> FigureSpec:
    return spec if spec is not None else FigureSpec()


def _split_out_path(out_path: str | Path) -> tuple[Path, str]:
    path = Path(out_path)
    return path.parent if str(path.parent) not in ("", ".") else Path("."), path.name


def _validate_series(
    x: Sequence[float], series: Mapping[str, Sequence[float]], *, x_name: str = "x"
) -> None:
    """Raise if `x` is empty or any named series does not match its length.

    Raises:
        ValueError: If `x` is empty, `series` is empty, or a series length
            does not match `len(x)`.
    """
    if len(x) == 0:
        raise ValueError(f"{x_name} must be non-empty")
    if len(series) == 0:
        raise ValueError("at least one data series is required")
    for name, y in series.items():
        if len(y) != len(x):
            raise ValueError(f"series {name!r} has {len(y)} points but {x_name} has {len(x)}")


def _validate_ci_series(
    series: Mapping[str, Sequence[float]], cis: Mapping[str, Sequence[CI]] | None
) -> None:
    """Raise if `cis` names a series that does not exist or mismatches its length.

    Raises:
        ValueError: If a key in `cis` is not in `series`, or its length does
            not match that series' length.
    """
    if cis is None:
        return
    for name, ci_seq in cis.items():
        if name not in series:
            raise ValueError(f"cis has an entry for {name!r} but no matching data series")
        if len(ci_seq) != len(series[name]):
            raise ValueError(
                f"cis[{name!r}] has {len(ci_seq)} entries but its series has {len(series[name])}"
            )


def _draw_band(ax: Axes, x: Sequence[float], cis: Sequence[CI] | None, color: Any) -> None:
    """Draw a CI band via `fill_between`, skipping single-point series (no band)."""
    if cis is None or len(x) < 2:
        return
    lo = [c.lo for c in cis]
    hi = [c.hi for c in cis]
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def _draw_multi_series(
    ax: Axes,
    x: Sequence[float],
    series: Mapping[str, Sequence[float]],
    cis: Mapping[str, Sequence[CI]] | None,
) -> None:
    """Plot every named series as a line with markers, plus its CI band if given."""
    for name, y in series.items():
        (line,) = ax.plot(x, y, marker="o", label=name)
        _draw_band(ax, x, None if cis is None else cis.get(name), line.get_color())


def _build_g_curve(
    lambda_grid: Sequence[float], g_values: Sequence[float], alpha: float, lambda_hat: float
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must be non-empty")
    if len(g_values) != len(lambda_grid):
        raise ValueError(
            f"g_values has {len(g_values)} points but lambda_grid has {len(lambda_grid)}"
        )
    fig, ax = new_figure(FigureSpec())
    ax.plot(lambda_grid, g_values, marker="o", label="g(λ)")
    ax.axhline(alpha, linestyle="--", color="tab:red", label="alpha")
    ax.axvline(lambda_hat, linestyle=":", color="tab:green", label="λ̂")
    ax.set_xlabel("λ")
    ax.set_ylabel("g(λ)")
    ax.set_title("Inflated criterion g(λ) vs λ")
    ax.legend()
    fig.tight_layout()
    data = {"lambda_grid": list(lambda_grid), "g_values": list(g_values), "alpha": alpha}
    header = ["lambda", "g"]
    rows: list[list[Any]] = [
        [float(lam), float(g)] for lam, g in zip(lambda_grid, g_values, strict=True)
    ]
    return fig, ax, data, header, rows


def plot_g_curve(
    lambda_grid: Sequence[float],
    g_values: Sequence[float],
    alpha: float,
    lambda_hat: float,
    out_path: str | Path,
    *,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """`g(λ)` vs `λ`, with the `alpha` reference line and the selected `λ̂`.

    Args:
        lambda_grid: The `λ`-grid `g` was evaluated on.
        g_values: `g(λ)` at each grid point, same length as `lambda_grid`.
        alpha: The target risk level (drawn as a horizontal reference line).
        lambda_hat: The selected threshold (drawn as a vertical marker line).
        out_path: Destination path stem (no extension); `.{spec.format}`
            and `.csv` are appended.
        spec: Format/dpi/figsize; defaults to `FigureSpec()`.

    Returns:
        The rendered :class:`~wfcrc.visualization.base.FigureFile`.

    Raises:
        ValueError: If `lambda_grid` is empty or `g_values` mismatches its length.
    """
    fig, _, data, header, rows = _build_g_curve(lambda_grid, g_values, alpha, lambda_hat)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_risk_vs_alpha(
    alphas: Sequence[float],
    risks_by_family: Mapping[str, Sequence[float]],
    cis_by_family: Mapping[str, Sequence[CI]] | None,
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    _validate_series(alphas, risks_by_family, x_name="alphas")
    _validate_ci_series(risks_by_family, cis_by_family)
    fig, ax = new_figure(FigureSpec())
    lo, hi = min(alphas), max(alphas)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="target (y=x)")
    _draw_multi_series(ax, alphas, risks_by_family, cis_by_family)
    ax.set_xlabel("target alpha")
    ax.set_ylabel("realized worst-case risk")
    ax.set_title("F1: realized worst-case risk vs alpha")
    ax.legend()
    fig.tight_layout()
    data = {
        "alphas": list(alphas),
        "risks_by_family": {k: list(v) for k, v in risks_by_family.items()},
    }
    header = ["family", "alpha", "risk", "ci_lo", "ci_hi"]
    rows: list[list[Any]] = []
    for name, risks in risks_by_family.items():
        cis = cis_by_family.get(name) if cis_by_family else None
        for i, a in enumerate(alphas):
            ci = cis[i] if cis is not None else None
            rows.append([name, float(a), float(risks[i]), ci.lo if ci else "", ci.hi if ci else ""])
    return fig, ax, data, header, rows


def plot_risk_vs_alpha(
    alphas: Sequence[float],
    risks_by_family: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    cis_by_family: Mapping[str, Sequence[CI]] | None = None,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F1: realized worst-case risk vs `alpha`, one series per family, with the `y=x` target line.

    Args:
        alphas: Target risk levels evaluated.
        risks_by_family: `{family_name: realized risks}`, each aligned to `alphas`.
        out_path: Destination path stem.
        cis_by_family: Optional `{family_name: CIs}`, each aligned to `alphas`.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `alphas`/`risks_by_family` is empty, a series length
            mismatches `alphas`, or `cis_by_family` names an unknown series
            or mismatched length.
    """
    fig, _, data, header, rows = _build_risk_vs_alpha(alphas, risks_by_family, cis_by_family)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_group_risk(
    group_labels: Sequence[str],
    wfcrc_risks: Sequence[float],
    baseline_risks: Sequence[float],
    alpha: float,
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    if len(group_labels) == 0:
        raise ValueError("group_labels must be non-empty")
    if len(wfcrc_risks) != len(group_labels) or len(baseline_risks) != len(group_labels):
        raise ValueError("wfcrc_risks/baseline_risks must match group_labels in length")
    fig, ax = new_figure(FigureSpec())
    x = np.arange(len(group_labels))
    width = 0.35
    ax.bar(x - width / 2, wfcrc_risks, width, label="WF-CRC")
    ax.bar(x + width / 2, baseline_risks, width, label="baseline")
    ax.axhline(alpha, linestyle="--", color="tab:red", label="alpha")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha="right")
    ax.set_ylabel("realized risk")
    ax.set_title("F2: per-group realized risk, WF-CRC vs baseline")
    ax.legend()
    fig.tight_layout()
    data = {
        "group_labels": list(group_labels),
        "wfcrc_risks": list(wfcrc_risks),
        "baseline_risks": list(baseline_risks),
        "alpha": alpha,
    }
    header = ["group", "wfcrc_risk", "baseline_risk"]
    rows: list[list[Any]] = [
        [g, float(w), float(b)]
        for g, w, b in zip(group_labels, wfcrc_risks, baseline_risks, strict=True)
    ]
    return fig, ax, data, header, rows


def plot_group_risk(
    group_labels: Sequence[str],
    wfcrc_risks: Sequence[float],
    baseline_risks: Sequence[float],
    alpha: float,
    out_path: str | Path,
    *,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F2: per-region/group realized risk, WF-CRC vs a baseline (bars), with the `alpha` line.

    Args:
        group_labels: One label per group/region.
        wfcrc_risks: WF-CRC's realized risk per group, aligned to `group_labels`.
        baseline_risks: The baseline's realized risk per group, aligned to `group_labels`.
        alpha: The target risk level (horizontal reference line).
        out_path: Destination path stem.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `group_labels` is empty or the risk sequences mismatch its length.
    """
    fig, _, data, header, rows = _build_group_risk(group_labels, wfcrc_risks, baseline_risks, alpha)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_risk_vs_shift(
    severities: Sequence[float],
    risks_by_method: Mapping[str, Sequence[float]],
    alpha: float,
    cis_by_method: Mapping[str, Sequence[CI]] | None,
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    _validate_series(severities, risks_by_method, x_name="severities")
    _validate_ci_series(risks_by_method, cis_by_method)
    fig, ax = new_figure(FigureSpec())
    _draw_multi_series(ax, severities, risks_by_method, cis_by_method)
    ax.axhline(alpha, linestyle="--", color="tab:red", label="alpha")
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("realized risk")
    ax.set_title("F3: realized risk vs corruption severity")
    ax.legend()
    fig.tight_layout()
    data = {
        "severities": list(severities),
        "risks_by_method": {k: list(v) for k, v in risks_by_method.items()},
        "alpha": alpha,
    }
    header = ["method", "severity", "risk", "ci_lo", "ci_hi"]
    rows: list[list[Any]] = []
    for name, risks in risks_by_method.items():
        cis = cis_by_method.get(name) if cis_by_method else None
        for i, s in enumerate(severities):
            ci = cis[i] if cis is not None else None
            rows.append([name, float(s), float(risks[i]), ci.lo if ci else "", ci.hi if ci else ""])
    return fig, ax, data, header, rows


def plot_risk_vs_shift(
    severities: Sequence[float],
    risks_by_method: Mapping[str, Sequence[float]],
    alpha: float,
    out_path: str | Path,
    *,
    cis_by_method: Mapping[str, Sequence[CI]] | None = None,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F3: realized risk vs corruption severity, one series per method, with the `alpha` line.

    Args:
        severities: Corruption severity levels (e.g. `1..5`).
        risks_by_method: `{method_name: realized risks}`, each aligned to `severities`.
        alpha: The target risk level (horizontal reference line).
        out_path: Destination path stem.
        cis_by_method: Optional `{method_name: CIs}`, each aligned to `severities`.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `severities`/`risks_by_method` is empty, a series
            length mismatches `severities`, or `cis_by_method` names an
            unknown series or mismatched length.
    """
    fig, _, data, header, rows = _build_risk_vs_shift(
        severities, risks_by_method, alpha, cis_by_method
    )
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_set_size(
    alphas: Sequence[float], sizes_by_method: Mapping[str, Sequence[float]]
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    _validate_series(alphas, sizes_by_method, x_name="alphas")
    fig, ax = new_figure(FigureSpec())
    _draw_multi_series(ax, alphas, sizes_by_method, None)
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean prediction-set size")
    ax.set_title("F4: set-size vs alpha efficiency frontier")
    ax.legend()
    fig.tight_layout()
    data = {
        "alphas": list(alphas),
        "sizes_by_method": {k: list(v) for k, v in sizes_by_method.items()},
    }
    header = ["method", "alpha", "size"]
    rows: list[list[Any]] = [
        [name, float(a), float(s)]
        for name, sizes in sizes_by_method.items()
        for a, s in zip(alphas, sizes, strict=True)
    ]
    return fig, ax, data, header, rows


def plot_set_size(
    alphas: Sequence[float],
    sizes_by_method: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F4: mean prediction-set size vs `alpha`, one series per method (the efficiency frontier).

    Args:
        alphas: Target risk levels evaluated.
        sizes_by_method: `{method_name: mean set sizes}`, each aligned to `alphas`.
        out_path: Destination path stem.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `alphas`/`sizes_by_method` is empty or a series length mismatches `alphas`.
    """
    fig, _, data, header, rows = _build_set_size(alphas, sizes_by_method)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_nesting(
    group_masses: Sequence[float], divergences: Sequence[float], threshold: float
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    if len(group_masses) == 0:
        raise ValueError("group_masses must be non-empty")
    if len(divergences) != len(group_masses):
        raise ValueError(
            f"divergences has {len(divergences)} points but group_masses has {len(group_masses)}"
        )
    fig, ax = new_figure(FigureSpec())
    ax.scatter(group_masses, divergences, label="groups")
    ax.axvline(threshold, linestyle="--", color="gray", label="P(G) threshold")
    ax.set_xlabel("group mass P(G)")
    ax.set_ylabel("divergence |lambda_hat_G - lambda_hat|")
    ax.set_title("F5: coincide/diverge nesting")
    ax.legend()
    fig.tight_layout()
    data = {
        "group_masses": list(group_masses),
        "divergences": list(divergences),
        "threshold": threshold,
    }
    header = ["group_mass", "divergence"]
    rows: list[list[Any]] = [
        [float(m), float(d)] for m, d in zip(group_masses, divergences, strict=True)
    ]
    return fig, ax, data, header, rows


def plot_nesting(
    group_masses: Sequence[float],
    divergences: Sequence[float],
    threshold: float,
    out_path: str | Path,
    *,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F5: per-group `λ̂` divergence from the worst-case `λ̂`, vs group mass `P(G)`.

    Illustrates the coincide/diverge nesting boundary (Cor iii, L4): groups
    with mass at or above `threshold` are expected to coincide with the
    worst-case-ball threshold; rare groups (mass below `threshold`) diverge.

    Args:
        group_masses: `P(G)` for each group.
        divergences: `|lambda_hat_G - lambda_hat|` for each group, aligned to `group_masses`.
        threshold: The `P(G)` value at which coincidence is predicted to break down.
        out_path: Destination path stem.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `group_masses` is empty or `divergences` mismatches its length.
    """
    fig, _, data, header, rows = _build_nesting(group_masses, divergences, threshold)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_qualitative(
    heatmap: np.ndarray, overlay_mask: np.ndarray | None, title: str
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    arr = np.asarray(heatmap, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"heatmap must be a non-empty 2-D array, got shape {arr.shape}")
    mask_arr: np.ndarray | None = None
    if overlay_mask is not None:
        mask_arr = np.asarray(overlay_mask, dtype=np.bool_)
        if mask_arr.shape != arr.shape:
            raise ValueError(
                f"overlay_mask shape {mask_arr.shape} does not match heatmap shape {arr.shape}"
            )
    fig, ax = new_figure(FigureSpec())
    image = ax.imshow(arr, cmap="viridis")
    fig.colorbar(image, ax=ax)
    if mask_arr is not None:
        ax.contour(mask_arr, levels=[0.5], colors="red", linewidths=1.0)
    ax.set_title(title or "F6: qualitative risk heatmap")
    fig.tight_layout()
    data: dict[str, Any] = {"heatmap": arr.tolist()}
    header = ["row", "col", "value"]
    rows: list[list[Any]] = [[int(i), int(j), float(v)] for (i, j), v in np.ndenumerate(arr)]
    if mask_arr is not None:
        data["overlay_mask"] = mask_arr.tolist()
        header.append("mask")
        mask_by_pos = {(int(i), int(j)): bool(v) for (i, j), v in np.ndenumerate(mask_arr)}
        rows = [[*row, mask_by_pos[(row[0], row[1])]] for row in rows]
    return fig, ax, data, header, rows


def plot_qualitative(
    heatmap: np.ndarray,
    out_path: str | Path,
    *,
    overlay_mask: np.ndarray | None = None,
    title: str = "",
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F6: a qualitative per-pixel risk/uncertainty heatmap, with an optional boundary overlay.

    Args:
        heatmap: A 2-D array of per-pixel values (e.g. risk or uncertainty).
        out_path: Destination path stem.
        overlay_mask: Optional 2-D boolean array, same shape as `heatmap`,
            drawn as a contour outline (e.g. a ground-truth boundary).
        title: Figure title; defaults to a generic F6 caption if empty.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `heatmap` is not a non-empty 2-D array, or
            `overlay_mask`'s shape does not match `heatmap`'s.
    """
    fig, _, data, header, rows = _build_qualitative(heatmap, overlay_mask, title)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_runtime(
    sizes: Sequence[float],
    values_by_method: Mapping[str, Sequence[float]],
    ylabel: str,
    log_x: bool,
    log_y: bool,
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    _validate_series(sizes, values_by_method, x_name="sizes")
    fig, ax = new_figure(FigureSpec())
    _draw_multi_series(ax, sizes, values_by_method, None)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("n (or T)")
    ax.set_ylabel(ylabel)
    ax.set_title("F7: runtime/memory scaling")
    ax.legend()
    fig.tight_layout()
    data = {
        "sizes": list(sizes),
        "values_by_method": {k: list(v) for k, v in values_by_method.items()},
        "log_x": log_x,
        "log_y": log_y,
    }
    header = ["method", "size", "value"]
    rows: list[list[Any]] = [
        [name, float(n), float(v)]
        for name, values in values_by_method.items()
        for n, v in zip(sizes, values, strict=True)
    ]
    return fig, ax, data, header, rows


def plot_runtime(
    sizes: Sequence[float],
    values_by_method: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    ylabel: str = "wall-clock (s)",
    log_x: bool = True,
    log_y: bool = True,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F7: wall-clock or memory scaling vs input size `n`/`T`, one series per method.

    Args:
        sizes: The `n`/`T` values measured.
        values_by_method: `{method_name: measured values}` (wall-clock or
            memory; controlled by `ylabel`), each aligned to `sizes`.
        out_path: Destination path stem.
        ylabel: Y-axis label (e.g. `"wall-clock (s)"` or `"peak memory (MB)"`).
        log_x: Use a log-scaled x-axis (the usual scaling-plot convention).
        log_y: Use a log-scaled y-axis.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `sizes`/`values_by_method` is empty or a series length mismatches `sizes`.
    """
    fig, _, data, header, rows = _build_runtime(sizes, values_by_method, ylabel, log_x, log_y)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )


def _build_reliability(
    target_risks: Sequence[float], observed_risks: Sequence[float], cis: Sequence[CI] | None
) -> tuple[Figure, Axes, dict[str, Any], list[str], list[list[Any]]]:
    if len(target_risks) == 0:
        raise ValueError("target_risks must be non-empty")
    if len(observed_risks) != len(target_risks):
        raise ValueError(
            f"observed_risks has {len(observed_risks)} points but "
            f"target_risks has {len(target_risks)}"
        )
    if cis is not None and len(cis) != len(target_risks):
        raise ValueError(f"cis has {len(cis)} entries but target_risks has {len(target_risks)}")
    fig, ax = new_figure(FigureSpec())
    lo, hi = min(target_risks), max(target_risks)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="ideal (y=x)")
    (line,) = ax.plot(target_risks, observed_risks, marker="o", label="observed")
    _draw_band(ax, target_risks, cis, line.get_color())
    ax.set_xlabel("target risk")
    ax.set_ylabel("observed (realized) risk")
    ax.set_title("F8: reliability curve")
    ax.legend()
    fig.tight_layout()
    data = {"target_risks": list(target_risks), "observed_risks": list(observed_risks)}
    header = ["target", "observed", "ci_lo", "ci_hi"]
    rows: list[list[Any]] = [
        [
            float(t),
            float(o),
            cis[i].lo if cis is not None else "",
            cis[i].hi if cis is not None else "",
        ]
        for i, (t, o) in enumerate(zip(target_risks, observed_risks, strict=True))
    ]
    return fig, ax, data, header, rows


def plot_reliability(
    target_risks: Sequence[float],
    observed_risks: Sequence[float],
    out_path: str | Path,
    *,
    cis: Sequence[CI] | None = None,
    spec: FigureSpec | None = None,
) -> FigureFile:
    """F8: reliability curve — observed vs target risk, with the ideal `y=x` reference.

    Args:
        target_risks: The target risk levels evaluated (e.g. a grid of `alpha`).
        observed_risks: The realized risks at each target, aligned to `target_risks`.
        out_path: Destination path stem.
        cis: Optional CIs on `observed_risks`, aligned to `target_risks`.
        spec: Format/dpi/figsize.

    Returns:
        The rendered `FigureFile`.

    Raises:
        ValueError: If `target_risks` is empty, `observed_risks`/`cis` mismatch its length.
    """
    fig, _, data, header, rows = _build_reliability(target_risks, observed_risks, cis)
    out_dir, stem = _split_out_path(out_path)
    return render_figure(
        fig,
        data=data,
        out_dir=out_dir,
        stem=stem,
        spec=_resolve_spec(spec),
        sidecar_header=header,
        sidecar_rows=rows,
    )
