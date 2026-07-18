"""Unit tests for :mod:`wfcrc.datasets.preprocessing` (MS6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from wfcrc.datasets.preprocessing import resample_volume, resize_and_normalize
from wfcrc.exceptions import PreprocessingError

# --- resize_and_normalize ---------------------------------------------------


def test_resize_preserves_size_and_is_near_identity_before_normalization() -> None:
    image = np.arange(16.0).reshape(4, 4)
    out = resize_and_normalize(image, target_size=(4, 4), mean=[0.0], std=[1.0])
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, image)


def test_resize_changes_output_shape_grayscale() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    out = resize_and_normalize(image, target_size=(4, 2), mean=[0.0], std=[1.0])
    assert out.shape == (4, 2)


def test_resize_changes_output_shape_multichannel() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    out = resize_and_normalize(image, target_size=(4, 2), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    assert out.shape == (4, 2, 3)


def test_resize_output_dtype_is_float64() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    out = resize_and_normalize(image, target_size=(4, 4), mean=[0.0], std=[1.0])
    assert out.dtype == np.float64


def test_normalization_applies_per_channel_mean_and_std() -> None:
    image = np.full((2, 2, 2), fill_value=10.0)
    out = resize_and_normalize(image, target_size=(2, 2), mean=[0.0, 5.0], std=[2.0, 1.0])
    np.testing.assert_allclose(out[:, :, 0], np.full((2, 2), 5.0))
    np.testing.assert_allclose(out[:, :, 1], np.full((2, 2), 5.0))


def test_resize_corner_values_are_exact_under_upscaling() -> None:
    image = np.array([[0.0, 10.0], [20.0, 30.0]])
    out = resize_and_normalize(image, target_size=(4, 4), mean=[0.0], std=[1.0])
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, -1] == pytest.approx(10.0)
    assert out[-1, 0] == pytest.approx(20.0)
    assert out[-1, -1] == pytest.approx(30.0)


def test_resize_degenerate_single_pixel_input_broadcasts() -> None:
    image = np.array([[5.0]])
    out = resize_and_normalize(image, target_size=(3, 3), mean=[0.0], std=[1.0])
    np.testing.assert_allclose(out, np.full((3, 3), 5.0))


def test_resize_all_zero_input() -> None:
    image = np.zeros((5, 5))
    out = resize_and_normalize(image, target_size=(3, 3), mean=[0.0], std=[1.0])
    np.testing.assert_allclose(out, np.zeros((3, 3)))


def test_resize_all_zero_input_with_nonzero_mean() -> None:
    image = np.zeros((5, 5))
    out = resize_and_normalize(image, target_size=(3, 3), mean=[2.0], std=[1.0])
    np.testing.assert_allclose(out, np.full((3, 3), -2.0))


def test_resize_rejects_1d_input() -> None:
    with pytest.raises(PreprocessingError, match="2-D or 3-D"):
        resize_and_normalize(np.zeros(4), target_size=(2, 2), mean=[0.0], std=[1.0])


@pytest.mark.parametrize("target_size", [(0, 4), (4, 0), (-1, 4)])
def test_resize_rejects_non_positive_target_size(target_size: tuple[int, int]) -> None:
    with pytest.raises(PreprocessingError, match="target_size"):
        resize_and_normalize(np.zeros((4, 4)), target_size=target_size, mean=[0.0], std=[1.0])


def test_resize_rejects_mismatched_mean_length() -> None:
    with pytest.raises(PreprocessingError, match="mean/std"):
        resize_and_normalize(
            np.zeros((4, 4, 3)), target_size=(2, 2), mean=[0.0, 0.0], std=[1.0, 1.0, 1.0]
        )


def test_resize_rejects_zero_std() -> None:
    with pytest.raises(PreprocessingError, match="std"):
        resize_and_normalize(np.zeros((4, 4)), target_size=(2, 2), mean=[0.0], std=[0.0])


# --- resample_volume ---------------------------------------------------------


def test_resample_volume_preserves_shape_under_equal_spacing() -> None:
    volume = np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2)
    out = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out, volume)


def test_resample_volume_expected_output_dimensions_on_downsampling() -> None:
    volume = np.zeros((8, 8, 8))
    # Doubling the target spacing halves the voxel count per axis.
    out = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(2.0, 2.0, 2.0))
    assert out.shape == (4, 4, 4)


def test_resample_volume_expected_output_dimensions_on_upsampling() -> None:
    volume = np.zeros((4, 4, 4))
    out = resample_volume(volume, spacing=(2.0, 2.0, 2.0), target_spacing=(1.0, 1.0, 1.0))
    assert out.shape == (8, 8, 8)


def test_resample_volume_anisotropic_spacing() -> None:
    volume = np.zeros((10, 20, 30))
    out = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(2.0, 1.0, 5.0))
    assert out.shape == (5, 20, 6)


def test_resample_volume_output_dtype_is_float64() -> None:
    volume = np.zeros((2, 2, 2), dtype=np.int16)
    out = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))
    assert out.dtype == np.float64


def test_resample_volume_is_deterministic() -> None:
    volume = np.random.default_rng(0).random((5, 6, 7))
    a = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(0.7, 1.3, 2.0))
    b = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(0.7, 1.3, 2.0))
    np.testing.assert_array_equal(a, b)


def test_resample_volume_degenerate_single_voxel_input_broadcasts() -> None:
    volume = np.array([[[7.0]]])
    out = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(0.5, 0.5, 0.5))
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out, np.full((2, 2, 2), 7.0))


def test_resample_volume_never_produces_a_zero_length_axis() -> None:
    volume = np.zeros((2, 2, 2))
    out = resample_volume(volume, spacing=(1.0, 1.0, 1.0), target_spacing=(1000.0, 1.0, 1.0))
    assert out.shape[0] == 1


def test_resample_volume_rejects_non_3d_input() -> None:
    with pytest.raises(PreprocessingError, match="3-D"):
        resample_volume(np.zeros((4, 4)), spacing=(1.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))


def test_resample_volume_rejects_wrong_length_spacing() -> None:
    with pytest.raises(PreprocessingError, match="exactly 3 entries"):
        resample_volume(np.zeros((2, 2, 2)), spacing=(1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0)])
def test_resample_volume_rejects_non_positive_spacing(spacing: tuple[float, float, float]) -> None:
    with pytest.raises(PreprocessingError, match="positive"):
        resample_volume(np.zeros((2, 2, 2)), spacing=spacing, target_spacing=(1.0, 1.0, 1.0))
