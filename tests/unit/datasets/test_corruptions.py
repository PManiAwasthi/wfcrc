"""Unit tests for :mod:`wfcrc.datasets.corruptions` (MS6.2 Resolution Pass, frozen Q2).

All corruption calls run against a tiny (32x32, the library's own
documented minimum) synthetic uint8 image — no real dataset, no network
access, no GPU. Per the MS6.2 task instructions, "different seed ->
different output" is asserted only for corruptions verified (module
docstring, `RANDOMIZED_CORRUPTIONS`) to actually consume randomness; the
essential, always-asserted invariant is "same input + same corruption +
same severity + same seed -> reproducible output".

Note on scope: the full cross-environment bit-identity / statistical
equivalence validation for the three compatibility-path corruptions
(`glass_blur`, `fog`, `impulse_noise`) was performed once, manually,
during the MS6.2 Resolution Pass against a dedicated reference conda
environment (Python 3.9, numpy==1.23.5, scikit-image==0.18.3/0.19.3) —
see `wfcrc/datasets/corruptions.py`'s module docstring for the full
numeric results. That reference environment is not a project dependency
and is not available at test time, so it cannot be re-run here; these
tests instead lock in the *observable contract* the investigation
established (determinism, shape/dtype, no exceptions, and — for
`impulse_noise` — that the realized proportion of changed pixels tracks
its `amount` parameter) as an ongoing regression guard.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

# Must import wfcrc.datasets.corruptions before anything imports
# `imagecorruptions` directly: the former installs a stdlib-only
# pkg_resources shim (MS6.2 Resolution Pass, Task 4) that `imagecorruptions`
# needs at its own module-load time on a modern setuptools with no
# pkg_resources at all. `get_corruption_names` is therefore imported
# locally (inside the one test that needs it) rather than at module level,
# so isort can never reorder it ahead of this import block.
from wfcrc.datasets.corruptions import (
    _IMPULSE_NOISE_SEVERITY_TABLE,
    COMPAT_PATH_CORRUPTIONS,
    CORRUPTION_NAMES,
    DIRECT_CORRUPTIONS,
    RANDOMIZED_CORRUPTIONS,
    _channel_axis_gaussian,
    _fog_compat,
    apply_corruption,
    ensure_pkg_resources_shim,
)
from wfcrc.exceptions import PreprocessingError

_DETERMINISTIC_CORRUPTIONS = tuple(
    name for name in CORRUPTION_NAMES if name not in RANDOMIZED_CORRUPTIONS
)


def _sample_image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(32, 32, 3)).astype(np.uint8)


# --- canonical name / severity validation -----------------------------------


def test_corruption_names_has_exactly_15_entries() -> None:
    assert len(CORRUPTION_NAMES) == 15


def test_corruption_names_matches_the_pinned_librarys_own_common_subset() -> None:
    # Guards against a future imagecorruptions version silently reordering,
    # renaming, adding, or removing a "common" corruption (frozen Q2: "do
    # not silently change corruption parameters"). Imported locally: see
    # the note on this module's import block.
    from imagecorruptions import get_corruption_names

    assert tuple(get_corruption_names("common")) == CORRUPTION_NAMES


def test_randomized_corruptions_is_a_subset_of_corruption_names() -> None:
    assert set(CORRUPTION_NAMES) >= RANDOMIZED_CORRUPTIONS


def test_apply_corruption_rejects_unknown_name() -> None:
    with pytest.raises(PreprocessingError, match="unknown corruption"):
        apply_corruption(_sample_image(), "not_a_real_corruption", severity=1, seed=0)


@pytest.mark.parametrize("severity", [0, 6, -1])
def test_apply_corruption_rejects_invalid_severity(severity: int) -> None:
    with pytest.raises(PreprocessingError, match="severity"):
        apply_corruption(_sample_image(), "contrast", severity=severity, seed=0)


# --- direct vs compatibility-path bookkeeping --------------------------------


def test_direct_and_compat_path_partition_all_15_corruptions() -> None:
    assert set(CORRUPTION_NAMES) == DIRECT_CORRUPTIONS | COMPAT_PATH_CORRUPTIONS
    assert DIRECT_CORRUPTIONS.isdisjoint(COMPAT_PATH_CORRUPTIONS)


def test_compat_path_is_exactly_the_three_resolved_corruptions() -> None:
    assert {"glass_blur", "fog", "impulse_noise"} == COMPAT_PATH_CORRUPTIONS
    assert len(DIRECT_CORRUPTIONS) == 12


# --- image validation ---------------------------------------------------------


def test_apply_corruption_rejects_non_uint8_dtype() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float64)
    with pytest.raises(PreprocessingError, match="uint8"):
        apply_corruption(image, "contrast", severity=1, seed=0)


def test_apply_corruption_rejects_wrong_channel_count() -> None:
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    with pytest.raises(PreprocessingError, match="1 or 3 channels"):
        apply_corruption(image, "contrast", severity=1, seed=0)


def test_apply_corruption_rejects_too_small_image() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(PreprocessingError, match=">= 32"):
        apply_corruption(image, "contrast", severity=1, seed=0)


def test_apply_corruption_rejects_1d_image() -> None:
    with pytest.raises(PreprocessingError, match="2-D or 3-D"):
        apply_corruption(np.zeros(32, dtype=np.uint8), "contrast", severity=1, seed=0)


# --- the essential invariant: same input+seed -> reproducible output, for ALL 15


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_same_seed_reproduces_identical_output(name: str) -> None:
    image = _sample_image()
    a = apply_corruption(image.copy(), name, severity=3, seed=42)
    b = apply_corruption(image.copy(), name, severity=3, seed=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_output_shape_and_dtype_match_input(name: str) -> None:
    image = _sample_image()
    out = apply_corruption(image, name, severity=2, seed=0)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_every_corruption_executes_at_every_severity(name: str, severity: int) -> None:
    # Regression guard for the MS6.2 Resolution Pass: glass_blur and fog
    # used to raise (TypeError / AttributeError) at every severity before
    # their compatibility paths were added.
    out = apply_corruption(_sample_image(), name, severity=severity, seed=1)
    assert out.dtype == np.uint8


# --- seed sensitivity: only asserted where the library actually uses randomness


@pytest.mark.parametrize("name", sorted(RANDOMIZED_CORRUPTIONS))
def test_randomized_corruptions_differ_across_seeds(name: str) -> None:
    image = _sample_image()
    a = apply_corruption(image.copy(), name, severity=3, seed=1)
    b = apply_corruption(image.copy(), name, severity=3, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("name", _DETERMINISTIC_CORRUPTIONS)
def test_deterministic_corruptions_are_seed_independent(name: str) -> None:
    # The essential invariant never requires a different-seed-different-
    # output assertion for these; this positive check goes further and
    # confirms they are mathematically deterministic (per MS6.2's own
    # source-level investigation), so a future library update that
    # introduces randomness here is caught rather than silently accepted.
    image = _sample_image()
    a = apply_corruption(image.copy(), name, severity=3, seed=1)
    b = apply_corruption(image.copy(), name, severity=3, seed=2)
    np.testing.assert_array_equal(a, b)


# --- impulse_noise compatibility-path: algorithm/parameter-preservation guard


@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_impulse_noise_realized_fraction_tracks_its_amount_parameter(severity: int) -> None:
    # MS6.2 Resolution Pass found the compatibility path is not bit-
    # identical to the reference environment (different PRNG bit
    # generator) but *is* algorithmically equivalent: the realized
    # proportion of altered pixels tracks the severity table's `amount`
    # value. This is the ongoing, dependency-free regression guard for
    # that finding (a generous tolerance band, since this is a small
    # 32x32 sample, not the 64x64 sample used during manual validation).
    image = _sample_image()
    out = apply_corruption(image.copy(), "impulse_noise", severity=severity, seed=7)
    realized_fraction = float(np.mean(out != image))
    target = _IMPULSE_NOISE_SEVERITY_TABLE[severity - 1]
    assert realized_fraction == pytest.approx(target, abs=0.05)


def test_impulse_noise_severity_table_matches_reference_source() -> None:
    # Verbatim transcription of imagecorruptions.corruptions.impulse_noise's
    # own severity table (frozen Q2: parameters must not be silently changed).
    assert _IMPULSE_NOISE_SEVERITY_TABLE == (0.03, 0.06, 0.09, 0.17, 0.27)


# --- ensure_pkg_resources_shim (MS6.2 Resolution Pass, Task 4) --------------


def test_shim_is_a_no_op_when_pkg_resources_already_in_sys_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setitem(sys.modules, "pkg_resources", sentinel)  # type: ignore[misc]
    ensure_pkg_resources_shim()
    assert sys.modules["pkg_resources"] is sentinel


def test_shim_defers_to_a_real_pkg_resources_if_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "pkg_resources":
            return object()  # any non-None sentinel signals "importable"
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    ensure_pkg_resources_shim()
    assert "pkg_resources" not in sys.modules


def test_shim_resource_filename_raises_for_an_unfindable_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the shim-install path regardless of whether the ambient
    # environment happens to have a real pkg_resources, so this test is
    # deterministic on every machine.
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "pkg_resources":
            return None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    ensure_pkg_resources_shim()
    shim = sys.modules["pkg_resources"]
    with pytest.raises(ModuleNotFoundError):
        shim.resource_filename("not_a_real_package_xyz_ms62", "foo.png")  # type: ignore[attr-defined]


# --- internal compat-path helpers: direct branch coverage -------------------


def test_channel_axis_gaussian_passthrough_without_multichannel_kwarg() -> None:
    # Covers the branch where a caller doesn't pass multichannel= at all;
    # imagecorruptions.corruptions.glass_blur always does, but the
    # translator itself should behave as a transparent passthrough
    # otherwise, matching plain skimage.filters.gaussian exactly.
    from skimage.filters import gaussian

    x = np.random.default_rng(0).random((16, 16))
    out = _channel_axis_gaussian(x, sigma=1.0)
    expected = gaussian(x, sigma=1.0)
    np.testing.assert_array_equal(out, expected)


def test_fog_compat_restores_a_pre_existing_np_float_attribute() -> None:
    # Covers the branch where np.float_ already existed before the compat
    # context manager ran (e.g. an older numpy) -- must be restored to
    # its prior value on exit, not deleted.
    np.float_ = np.float64  # type: ignore[attr-defined]
    try:
        with _fog_compat():
            assert np.float_ is np.float64  # type: ignore[attr-defined]
        assert np.float_ is np.float64  # type: ignore[attr-defined]
    finally:
        del np.float_  # type: ignore[attr-defined]
