"""Unit tests for :mod:`wfcrc.datasets.loaders._image_io` (DI-2).

The shared Pillow-backed 2-D image reader used by the ACDC and Kvasir
loaders. Fixtures are tiny images written with Pillow, never real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wfcrc.datasets.loaders._image_io import read_image_rgb, read_label_image
from wfcrc.exceptions import SerializationError


def _write(path: Path, array: np.ndarray, mode: str) -> None:
    Image.fromarray(array, mode=mode).save(path)


def test_read_image_rgb_returns_hwc_uint8(tmp_path: Path) -> None:
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    path = tmp_path / "img.png"
    _write(path, arr, "RGB")
    out = read_image_rgb(path)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_read_image_rgb_converts_grayscale_to_three_channels(tmp_path: Path) -> None:
    arr = np.array([[10, 200], [30, 40]], dtype=np.uint8)
    path = tmp_path / "gray.png"
    _write(path, arr, "L")
    out = read_image_rgb(path)
    assert out.shape == (2, 2, 3)
    # grayscale replicated across channels
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_read_image_rgb_converts_rgba_to_rgb(tmp_path: Path) -> None:
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 3] = 128
    path = tmp_path / "rgba.png"
    _write(path, arr, "RGBA")
    out = read_image_rgb(path)
    assert out.shape == (2, 2, 3)


def test_read_label_image_returns_single_channel(tmp_path: Path) -> None:
    arr = np.array([[0, 1, 18], [255, 2, 3]], dtype=np.uint8)
    path = tmp_path / "label.png"
    _write(path, arr, "L")
    out = read_label_image(path)
    assert out.shape == (2, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_read_label_image_collapses_equal_channel_rgb_mask(tmp_path: Path) -> None:
    # A binary mask stored RGB with R==G==B (Kvasir's real format) collapses
    # to its single meaningful channel.
    gray = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    path = tmp_path / "mask.png"
    _write(path, rgb, "RGB")
    out = read_label_image(path)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, gray)


def test_read_image_missing_file_raises_serialization_error(tmp_path: Path) -> None:
    with pytest.raises(SerializationError, match="does not exist"):
        read_image_rgb(tmp_path / "nope.png")
    with pytest.raises(SerializationError, match="does not exist"):
        read_label_image(tmp_path / "nope.png")


def test_read_image_corrupt_file_raises_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not a real png file")
    with pytest.raises(SerializationError, match="could not read image"):
        read_image_rgb(path)
    with pytest.raises(SerializationError, match="could not read image"):
        read_label_image(path)
