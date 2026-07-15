"""Numerically stable scalar/array primitives used by later math modules.

All functions operate in ``float64`` (:data:`wfcrc.constants.DEFAULT_FLOAT_DTYPE`)
and are pure (no hidden state, no randomness) so they are trivially
bit-deterministic given identical inputs. This module intentionally contains
no domain logic (no KL dual, no CVaR, no calibration) — only the generic
numerical building blocks those future modules will call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wfcrc.constants import DEFAULT_ETA_MIN

__all__ = [
    "clamp",
    "logsumexp",
    "quantile",
    "safe_div",
    "weighted_logsumexp",
]

_logger = logging.getLogger(__name__)

#: Minimum seconds between consecutive clamp-saturation DEBUG log records.
_CLAMP_LOG_MIN_INTERVAL_S = 1.0
_last_clamp_log_time = 0.0


def _to_float64(x: ArrayLike) -> NDArray[np.float64]:
    """Coerce ``x`` to a ``float64`` numpy array without copying if avoidable."""
    return np.asarray(x, dtype=np.float64)


def _require_nonempty(x: NDArray[np.float64]) -> None:
    """Raise :class:`ValueError` if ``x`` has zero elements.

    Args:
        x: Array to check.

    Raises:
        ValueError: If ``x.size == 0``.
    """
    if x.size == 0:
        raise ValueError("expected a non-empty array")


def _require_finite(x: NDArray[np.float64], *, allow_neg_inf: bool = False) -> None:
    """Raise :class:`ValueError` on NaN or (optionally) any infinite value.

    Args:
        x: Array to check.
        allow_neg_inf: If ``True``, ``-inf`` entries are permitted (they
            represent zero-probability mass and are handled correctly by
            :func:`logsumexp`); ``+inf`` and ``NaN`` are always rejected.

    Raises:
        ValueError: If ``x`` contains NaN, or contains an infinite value not
            permitted by ``allow_neg_inf``.
    """
    if np.any(np.isnan(x)):
        raise ValueError("input contains NaN")
    if allow_neg_inf:
        if np.any(np.isposinf(x)):
            raise ValueError("input contains +inf, which is not supported")
    elif np.any(np.isinf(x)):
        raise ValueError("input contains inf/-inf")


def logsumexp(x: ArrayLike, axis: int | None = None) -> np.float64 | NDArray[np.float64]:
    """Compute ``log(sum(exp(x)))`` with max-subtraction for numerical stability.

    ``-inf`` entries are supported (they contribute zero mass, matching the
    convention ``exp(-inf) == 0``); this is required for the KL dual's use of
    this function on log-weights that may be exactly zero probability.

    Args:
        x: Input array or scalar (coerced to ``float64``).
        axis: Axis (or ``None`` for the flattened array) to reduce over.

    Returns:
        The log-sum-exp reduction, as a ``float64`` scalar if ``axis`` is
        ``None``, else an array with that axis removed.

    Raises:
        ValueError: If ``x`` is empty, contains NaN, or contains ``+inf``.
    """
    arr = _to_float64(x)
    _require_nonempty(arr)
    _require_finite(arr, allow_neg_inf=True)

    x_max = np.max(arr, axis=axis, keepdims=True)
    # Where the max is -inf, every element on that reduction slice is -inf;
    # substitute 0 to avoid `-inf - (-inf) = nan`, and let log(sum(exp(0*...)))
    # collapse back to -inf via log(0) below.
    x_max_safe = np.where(np.isneginf(x_max), 0.0, x_max)
    summed = np.sum(np.exp(arr - x_max_safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        result = np.log(summed) + x_max_safe

    if axis is None:
        return np.float64(result.reshape(()))
    return np.asarray(np.squeeze(result, axis=axis), dtype=np.float64)


def weighted_logsumexp(
    x: ArrayLike, w: ArrayLike, axis: int | None = None
) -> np.float64 | NDArray[np.float64]:
    """Compute ``log(sum(w * exp(x)))`` stably.

    Equivalent to ``logsumexp(x + log(w), axis=axis)`` but avoids taking
    ``log`` of a zero weight directly.

    Args:
        x: Input array or scalar (coerced to ``float64``).
        w: Nonnegative weights, broadcastable against ``x``.
        axis: Axis (or ``None`` for the flattened array) to reduce over.

    Returns:
        The weighted log-sum-exp reduction.

    Raises:
        ValueError: If ``x``/``w`` are empty, contain NaN, ``+inf``, or if
            any weight is negative.
    """
    x_arr = _to_float64(x)
    w_arr = _to_float64(w)
    _require_nonempty(x_arr)
    _require_nonempty(w_arr)
    _require_finite(x_arr, allow_neg_inf=True)
    _require_finite(w_arr, allow_neg_inf=False)
    if np.any(w_arr < 0):
        raise ValueError("weights must be nonnegative")

    with np.errstate(divide="ignore"):
        log_w = np.where(w_arr > 0, np.log(np.where(w_arr > 0, w_arr, 1.0)), -np.inf)
    return logsumexp(x_arr + log_w, axis=axis)


def clamp(x: ArrayLike, lo: float, hi: float) -> np.float64 | NDArray[np.float64]:
    """Clip ``x`` into ``[lo, hi]``, logging (rate-limited) on saturation.

    Args:
        x: Input array or scalar (coerced to ``float64``).
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.

    Returns:
        ``x`` element-wise clipped to ``[lo, hi]``.

    Raises:
        ValueError: If ``x`` contains NaN/inf, or if ``lo > hi``.
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
    arr = _to_float64(x)
    _require_finite(arr, allow_neg_inf=False)

    result = np.clip(arr, lo, hi)
    if bool(np.any((arr < lo) | (arr > hi))):
        _log_clamp_saturation(lo, hi)

    if np.isscalar(x) or (isinstance(x, np.generic)):
        return np.float64(result)
    return result


def _log_clamp_saturation(lo: float, hi: float) -> None:
    """Emit a rate-limited DEBUG log when :func:`clamp` saturates a value."""
    global _last_clamp_log_time
    now = time.monotonic()
    if now - _last_clamp_log_time >= _CLAMP_LOG_MIN_INTERVAL_S:
        _last_clamp_log_time = now
        _logger.debug("clamp saturated at least one value to [%s, %s]", lo, hi)


def safe_div(a: ArrayLike, b: ArrayLike, eps: float = DEFAULT_ETA_MIN) -> NDArray[np.float64]:
    """Divide ``a / b``, guarding against division by (near-)zero ``b``.

    The magnitude of ``b`` is floored at ``eps`` while preserving its sign
    (zero is treated as positive), so the result is always finite for finite
    ``a`` and ``eps > 0``.

    Args:
        a: Numerator array or scalar.
        b: Denominator array or scalar.
        eps: Minimum allowed magnitude for the denominator.

    Returns:
        ``a / b`` with ``|b|`` floored at ``eps``.

    Raises:
        ValueError: If ``a``/``b`` contain NaN/inf, or ``eps <= 0``.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    a_arr = _to_float64(a)
    b_arr = _to_float64(b)
    _require_finite(a_arr, allow_neg_inf=False)
    _require_finite(b_arr, allow_neg_inf=False)

    sign = np.where(b_arr >= 0, 1.0, -1.0)
    safe_b = np.where(np.abs(b_arr) < eps, sign * eps, b_arr)
    return np.asarray(a_arr / safe_b, dtype=np.float64)


def quantile(
    x: ArrayLike,
    q: float,
    method: Literal["linear", "lower", "higher", "midpoint", "nearest"] = "linear",
) -> np.float64:
    """Compute the ``q``-th quantile of ``x`` with a fixed interpolation method.

    A thin, validated wrapper over :func:`numpy.quantile` so every caller in
    wfcrc uses the same (deterministic) interpolation convention.

    Args:
        x: Input array (coerced to ``float64``).
        q: Quantile to compute, in ``[0, 1]``.
        method: Interpolation method, forwarded to :func:`numpy.quantile`.

    Returns:
        The ``q``-th quantile as a ``float64`` scalar.

    Raises:
        ValueError: If ``x`` is empty, contains NaN/inf, or ``q`` is not in
            ``[0, 1]``.
    """
    arr = _to_float64(x)
    _require_nonempty(arr)
    _require_finite(arr, allow_neg_inf=False)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")

    result: Any = np.quantile(arr, q, method=method)
    return np.float64(result)
