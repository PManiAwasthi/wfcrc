"""Dataset Preprocessing — pure 2-D resize/normalize and 3-D resampling (MS6.2, §3.7).

Every function here operates only on already-loaded ``numpy`` arrays — no
file I/O, no image codec, no dataset-specific knowledge. This is a
deliberate scope boundary (MS6 Architecture Specification §3.7:
"never touches raw file I/O — that's the loader's job"), and it is why
these two functions need **no new dependency**: separable 1-D linear
interpolation (:func:`numpy.interp`, applied axis by axis) reproduces
standard bilinear/trilinear resize without Pillow, SciPy, or a NIfTI
reader. Per the MS6.2 task instructions, a NIfTI reader belongs to the
Dataset Loader milestone (MS6.3), not here — this module resamples an
already-loaded 3-D array, it does not read a ``.nii``/``.nii.gz`` file.

Both functions are deterministic (no randomness) given their inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wfcrc.exceptions import PreprocessingError

__all__ = ["resample_volume", "resize_and_normalize"]


def _resize_axis(array: NDArray[np.float64], axis: int, new_size: int) -> NDArray[np.float64]:
    """Resize ``array`` along one ``axis`` via separable 1-D linear interpolation.

    Args:
        array: Array to resize.
        axis: Axis to resize along.
        new_size: Target length of ``axis`` (must be positive).

    Returns:
        A new array, identical to ``array`` except ``axis`` now has length
        ``new_size``. Degenerate (length-1) source axes broadcast to a
        constant value, which ``numpy.interp`` already does correctly via
        its default out-of-range fill behavior.
    """
    old_size = array.shape[axis]
    old_coords = np.linspace(0.0, 1.0, old_size)
    new_coords = np.linspace(0.0, 1.0, new_size)
    return np.apply_along_axis(lambda v: np.interp(new_coords, old_coords, v), axis, array)


def resize_and_normalize(
    image: NDArray[Any],
    target_size: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
) -> NDArray[np.float64]:
    """Resize a 2-D (or multi-channel 2-D) image and apply per-channel normalization.

    Resize is separable bilinear interpolation (:func:`numpy.interp` along
    each of the two spatial axes); normalization is ``(resized - mean) /
    std``, applied per channel. This function is scale-agnostic: it does
    not assume ``image`` is in ``[0, 255]`` or ``[0, 1]`` — callers pass
    ``mean``/``std`` in whatever numeric scale ``image`` is already in.

    Args:
        image: A ``(H, W)`` or ``(H, W, C)`` array.
        target_size: ``(target_height, target_width)``, both positive.
        mean: Per-channel mean, length ``1`` for a 2-D ``image`` or length
            ``C`` for a ``(H, W, C)`` image.
        std: Per-channel standard deviation, same length convention as
            ``mean``; every entry must be nonzero.

    Returns:
        A ``float64`` array of shape ``(target_height, target_width)`` or
        ``(target_height, target_width, C)``, matching ``image``'s rank.

    Raises:
        PreprocessingError: If ``image`` is not 2-D or 3-D, ``target_size``
            has a non-positive entry, ``mean``/``std`` do not each have one
            entry per channel, or any ``std`` entry is zero.
    """
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise PreprocessingError(f"image must be 2-D or 3-D, got ndim={arr.ndim}")

    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise PreprocessingError(f"target_size entries must be positive, got {target_size}")

    channels = 1 if arr.ndim == 2 else arr.shape[2]
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    if mean_arr.shape != (channels,) or std_arr.shape != (channels,):
        raise PreprocessingError(
            f"mean/std must each have exactly {channels} entries (one per channel), "
            f"got mean={mean_arr.shape}, std={std_arr.shape}"
        )
    if np.any(std_arr == 0.0):
        raise PreprocessingError("every std entry must be nonzero")

    resized = _resize_axis(arr.astype(np.float64), axis=0, new_size=target_h)
    resized = _resize_axis(resized, axis=1, new_size=target_w)

    if resized.ndim == 2:
        return (resized - float(mean_arr[0])) / float(std_arr[0])
    return (resized - mean_arr) / std_arr


def resample_volume(
    volume: NDArray[Any],
    spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Resample a 3-D volume from ``spacing`` to ``target_spacing`` via separable linear interp.

    Output shape along each axis is ``round(volume.shape[axis] *
    spacing[axis] / target_spacing[axis])``, clamped to at least 1.
    Resampling is separable 1-D linear interpolation
    (:func:`numpy.interp`), applied axis by axis — the same technique
    :func:`resize_and_normalize` uses in 2-D, extended to 3 axes.

    Args:
        volume: A ``(D, H, W)`` array (e.g. an already-loaded NIfTI volume).
        spacing: The volume's current voxel spacing, ``(sz, sy, sx)``,
            every entry positive.
        target_spacing: The desired voxel spacing, same convention,
            every entry positive.

    Returns:
        A ``float64`` array, resampled to the shape implied by
        ``target_spacing``.

    Raises:
        PreprocessingError: If ``volume`` is not 3-D, ``spacing``/
            ``target_spacing`` do not each have exactly 3 entries, or any
            entry of either is non-positive.
    """
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise PreprocessingError(f"volume must be 3-D, got ndim={arr.ndim}")
    if len(spacing) != 3 or len(target_spacing) != 3:
        raise PreprocessingError("spacing and target_spacing must each have exactly 3 entries")
    if any(s <= 0 for s in spacing) or any(s <= 0 for s in target_spacing):
        raise PreprocessingError("spacing and target_spacing entries must all be positive")

    resampled = arr.astype(np.float64)
    for axis in range(3):
        new_size = max(1, round(arr.shape[axis] * spacing[axis] / target_spacing[axis]))
        resampled = _resize_axis(resampled, axis=axis, new_size=new_size)
    return resampled
