"""Shared 2-D image reading for the PNG/JPG loader families (DI-2).

The ACDC (PNG) and Kvasir (JPG) loaders both read ordinary 2-D raster
images and their per-pixel label images from disk. This module holds the
one Pillow-backed reader they share, so neither loader reimplements
codec-level file reading or its error handling.

Pillow (``pillow``) is already a project dependency (pulled in by
``matplotlib`` at MS5 / RC1, pinned in ``requirements/lock.txt``), so this
adds **no new dependency** — the DI-2 self-audit's "no new dependency
unless genuinely required" rule (`PROJECT_CONTEXT.md` §9) holds.

Reads return raw ``uint8`` arrays at the file's native resolution — no
resize, normalize, or colour conversion beyond selecting a channel layout,
per the same "loaders do I/O, not preprocessing" boundary the MSD loader
observes (`wfcrc/datasets/loaders/msd.py` §5;
:mod:`wfcrc.datasets.preprocessing`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from wfcrc.exceptions import SerializationError

__all__ = ["read_image_rgb", "read_label_image"]


def _open(path: Path) -> Image.Image:
    """Open an image file, raising :class:`SerializationError` on any failure."""
    if not path.is_file():
        raise SerializationError(f"referenced image file does not exist: {path}")
    try:
        image = Image.open(path)
        image.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise SerializationError(f"could not read image {path}: {exc}") from exc
    return image


def read_image_rgb(path: Path) -> NDArray[np.uint8]:
    """Read a raster image as a native-resolution ``(H, W, 3)`` ``uint8`` RGB array.

    Any source mode (grayscale, palette, RGBA, ...) is converted to 3-channel
    RGB so downstream consumers see a single, predictable layout.

    Args:
        path: Path to a PNG/JPG (or any Pillow-readable) image file.

    Returns:
        An ``(H, W, 3)`` ``uint8`` array.

    Raises:
        SerializationError: If ``path`` does not exist or cannot be decoded.
    """
    image = _open(path)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array


def read_label_image(path: Path) -> NDArray[np.uint8]:
    """Read a per-pixel label/mask image as a native-resolution 2-D ``uint8`` array.

    The image is read as a single 8-bit channel (Pillow mode ``"L"``): for a
    class-index label map (e.g. Cityscapes ``*_labelTrainIds.png``) this is
    the class index per pixel; for a binary mask stored as an image (e.g. a
    Kvasir polyp mask) this is the per-pixel intensity a caller then
    thresholds. Reading as ``"L"`` collapses an accidentally-RGB-encoded
    grayscale label (Kvasir's masks are stored RGB with ``R==G==B``) to its
    single meaningful channel via Pillow's standard luma conversion.

    Args:
        path: Path to a label/mask image file.

    Returns:
        A 2-D ``(H, W)`` ``uint8`` array.

    Raises:
        SerializationError: If ``path`` does not exist or cannot be decoded.
    """
    image = _open(path)
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array
