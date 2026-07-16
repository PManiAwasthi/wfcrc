"""Deterministic paper figures rendered from metrics manifests (Implementation Blueprint §14, M14).

Public API: :class:`~wfcrc.visualization.base.FigureSpec` /
:class:`~wfcrc.visualization.base.FigureFile`, and the figure functions in
:mod:`wfcrc.visualization.plots` — :func:`~wfcrc.visualization.plots.plot_g_curve`,
:func:`~wfcrc.visualization.plots.plot_risk_vs_alpha` (F1),
:func:`~wfcrc.visualization.plots.plot_group_risk` (F2),
:func:`~wfcrc.visualization.plots.plot_risk_vs_shift` (F3),
:func:`~wfcrc.visualization.plots.plot_set_size` (F4),
:func:`~wfcrc.visualization.plots.plot_nesting` (F5),
:func:`~wfcrc.visualization.plots.plot_qualitative` (F6),
:func:`~wfcrc.visualization.plots.plot_runtime` (F7),
:func:`~wfcrc.visualization.plots.plot_reliability` (F8).
"""

from __future__ import annotations

from wfcrc.visualization.base import FigureFile, FigureSpec
from wfcrc.visualization.plots import (
    plot_g_curve,
    plot_group_risk,
    plot_nesting,
    plot_qualitative,
    plot_reliability,
    plot_risk_vs_alpha,
    plot_risk_vs_shift,
    plot_runtime,
    plot_set_size,
)

__all__ = [
    "FigureFile",
    "FigureSpec",
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
