"""Cityscapes-C corruption utilities (MS6.2, §3.7; frozen Q2).

Per frozen Q2, ``imagecorruptions`` is the reference implementation for
the Cityscapes-C corruption protocol, but **the frozen scientific artifact
is the protocol itself** (corruption types, severities, parameterization,
deterministic seeding), not the library call.

**MS6.2 Resolution Pass — final compatibility strategy.** The initial
MS6.2 pass found 3 of the 15 required "common" corruptions
(``glass_blur``, ``fog``, ``impulse_noise``) failing against
``imagecorruptions==1.1.2`` in this project's own dependency environment
(``numpy>=1.26,<3`` resolving to NumPy 2.x, a ``cp312``-compatible
scikit-image). Per the frozen Resolution Policy, the main WFCRC
environment was **not** downgraded to accommodate the outdated library;
instead each failure was individually root-caused and repaired with the
narrowest possible compatibility path, then validated against a genuine
reference environment. **All 15 corruptions now execute.**

Reference environment used for equivalence validation (built solely for
this one-time verification, never a runtime dependency of ``wfcrc``):
a separate conda environment, Python 3.9, ``numpy==1.23.5``,
``scikit-image==0.18.3``/``0.19.3``, ``scipy==1.10.1`` — versions old
enough that ``imagecorruptions==1.1.2`` runs entirely unmodified (all 15
corruptions verified to execute natively there).

**Per-corruption resolution:**

- **``glass_blur`` — Priority 1 (minimal runtime adapter), exact
  equivalence.** Root cause: ``imagecorruptions.corruptions.glass_blur``
  calls ``skimage.filters.gaussian(..., multichannel=True)``; that
  keyword was removed from every scikit-image release with a ``cp312``
  wheel, in favor of ``channel_axis``. Fix: for the duration of the one
  ``corrupt()`` call, the module-level ``gaussian`` name inside
  ``imagecorruptions.corruptions`` is swapped for a thin wrapper that
  translates ``multichannel=True -> channel_axis=-1`` /
  ``multichannel=False -> channel_axis=None`` and delegates to the real
  ``skimage.filters.gaussian`` — no corruption algorithm logic is
  touched. **Validated:** at all 5 of ``glass_blur``'s severity sigmas
  (0.7, 0.9, 1.0, 1.1, 1.5), both color and grayscale,
  ``gaussian(..., multichannel=True)`` and
  ``gaussian(..., channel_axis=-1)`` were run side by side in an
  environment where both still work (scikit-image 0.19.3) —
  **bit-identical, max abs diff = 0.0, at every sigma.** Cross-environment
  check: the same image, seed, and severities 1-5 run through the patched
  ``wfcrc_env`` and the reference environment's unmodified
  ``imagecorruptions`` produced **bit-identical output (max/mean abs
  diff = 0.0) at every severity**.
- **``fog`` — Priority 1 (minimal runtime adapter), exact equivalence.**
  Root cause: ``imagecorruptions.corruptions.plasma_fractal`` (called by
  ``fog``) uses ``dtype=np.float_``; that alias was removed in NumPy 2.0
  (NumPy's own removal message: "``np.float_`` was removed in the NumPy
  2.0 release. Use ``np.float64`` instead." — ``np.float_`` was *literally*
  ``np.float64`` under a different name in every NumPy version where it
  existed, not merely a similar type). Fix: ``np.float_`` is restored as
  a plain module attribute (``= np.float64``) for the duration of the one
  ``corrupt()`` call, then removed again — no corruption algorithm logic
  is touched. **Validated:** the same image, seed, and severities 1-5 run
  through the patched ``wfcrc_env`` and the reference environment's
  unmodified ``imagecorruptions`` produced **bit-identical output (max/mean
  abs diff = 0.0) at every severity** — expected, since ``fog``'s
  randomness (``np.random.uniform`` inside ``plasma_fractal``) goes
  through numpy's *legacy* global RNG API, which NumPy deliberately kept
  bit-stream-compatible across the 1.x -> 2.x transition; only the dtype
  alias needed repair.
- **``impulse_noise`` — Priority 2 (internal reimplementation of the
  3-line wrapper only), algorithm/parameters preserved, determinism
  restored, NOT bit-identical to the reference (quantified below).**
  Root cause: ``imagecorruptions.corruptions.impulse_noise`` calls
  ``skimage.util.random_noise(x, mode='s&p', amount=c)`` with no
  seed/rng argument. Reading both scikit-image versions' actual
  ``random_noise`` source side by side (MS6.2 resolution pass) confirms
  the salt-and-pepper *algorithm* is byte-for-byte unchanged (same
  ``flipped``/``salted``/``peppered`` boolean-mask construction, same
  ``amount``/``salt_vs_pepper`` semantics) — only the RNG plumbing
  changed: the old signature was
  ``random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs)``
  and fell back to numpy's *legacy* global RNG when ``seed=None``; the
  installed signature is
  ``random_noise(image, mode='gaussian', rng=None, clip=True, **kwargs)``,
  and with ``rng=None`` it draws from its own independent
  ``numpy.random.default_rng()`` (PCG64), decoupled from the legacy
  global state entirely — this is *not* a bug in this project's code,
  it is scikit-image's own documented RNG-modernization (``rng: int`` ->
  ``numpy.random.default_rng(rng)``, per the installed function's own
  docstring). Fix: :func:`apply_corruption` reimplements
  ``impulse_noise``'s exact 3-line body (the severity-to-``amount`` table,
  unchanged, and the clip/scale-to-uint8 step, unchanged) so it can pass
  the derived seed through the *modern* ``rng=`` parameter directly — the
  actual noise-generating call remains the unmodified, upstream
  ``skimage.util.random_noise`` function; only the outer 3-line
  convenience wrapper (not the corruption algorithm) is duplicated.
  **Validated (severities 1-5, same synthetic image, reference
  environment vs. this compatibility path):** NOT bit-identical
  (``max_abs_diff = 255`` at every severity — expected for a binary
  salt-and-pepper substitution once two different PRNG algorithms
  disagree on which pixels flip). The compatibility path *is*
  deterministic (same seed -> same output, confirmed at every severity)
  and *is* seed-sensitive (different seed -> different output, confirmed
  at every severity). The algorithmically meaningful quantity — the
  realized fraction of pixels altered, which is what the ``amount``
  parameter actually controls — tracks the reference closely at every
  severity (e.g. severity 1: 3.13% reference vs. 2.94% compat, target
  ``amount=0.03``; severity 5: 27.53% reference vs. 27.64% compat, target
  ``amount=0.27``), confirming the two PRNGs realize the *same*
  proportion-controlled algorithm, merely disagreeing (as any two
  independent PRNGs would) on *which* pixels a given proportion lands on.
  **Conclusion: the difference is not algorithmically meaningful** — it is
  a PRNG-bit-generator artifact (legacy MT19937-via-implicit-global-state
  vs. modern PCG64-via-explicit-seed), not a difference in corruption
  type, severity mapping, or formula. Bit-identical reproduction of one
  specific historical RNG stream was never part of the frozen corruption
  *protocol* (types + severities + parameterization) to begin with — only
  reproducibility of *this project's own* runs, which is fully restored.

:data:`DIRECT_CORRUPTIONS` (12) call the unmodified upstream
``imagecorruptions.corrupt()`` directly. :data:`COMPAT_PATH_CORRUPTIONS`
(3: ``glass_blur``, ``fog``, ``impulse_noise``) use the narrow paths
above. No corruption was reimplemented beyond what is documented here; no
severity table or corruption type was altered; ``imagecorruptions`` was
not swapped for a different library.

**``setuptools<81`` dependency — reassessed and removed (MS6.2 Resolution
Pass, Task 4).** The initial MS6.2 pass pinned ``setuptools<81`` project-wide
to keep ``pkg_resources`` importable, because
``imagecorruptions.corruptions`` does ``from pkg_resources import
resource_filename`` at module load time, using it for exactly one thing:
locating the six bundled ``frost*.png``/``.jpg`` texture files ``frost()``
blends into the image. Rather than keep an entire (increasingly
deprecated — its own runtime warning says "slated for removal as early as
2025-11-30") legacy packaging API pinned as a project-wide dependency for
one function call, this module installs a **stdlib-only, ~10-line shim**
(:func:`_ensure_pkg_resources_shim`, using only ``importlib.util`` +
``pathlib``) into ``sys.modules['pkg_resources']`` before importing
``imagecorruptions`` — but only if a *real* ``pkg_resources`` is not
already importable, so an environment that does have it (e.g. an older
setuptools already installed for an unrelated reason) is left alone and
takes precedence. **Validated:** with modern ``setuptools`` (83.0.0, no
``pkg_resources`` at all) the shim resolves ``frost()``'s bundled images
and every one of 10 tested seeds (statistically covering all 5 of
``frost``'s bundled-image choices) produced **bit-identical output** to
the same calls made with the real ``pkg_resources`` (via a
``setuptools<81`` pin, tested for comparison only). The ``setuptools<81``
pin has therefore been **removed** from this project's dependencies —
this module has no dependency on ``pkg_resources`` or on any particular
``setuptools`` version at all.

**Import-order requirement.** The shim only helps if it runs *before*
anything does ``import imagecorruptions``/``from imagecorruptions import
...`` — any code (elsewhere in ``wfcrc``, or a test) that imports
``imagecorruptions`` directly, without first importing this module, can
still hit the same ``ModuleNotFoundError: No module named 'pkg_resources'``
on a modern-setuptools environment. Any code in this codebase that needs
``imagecorruptions`` directly must import
:mod:`wfcrc.datasets.corruptions` (or call :func:`ensure_pkg_resources_shim`
directly) first; :func:`ensure_pkg_resources_shim` is exported (not
private) for exactly this defensive use.

**Determinism bridge (unchanged from the initial MS6.2 pass).**
``imagecorruptions.corrupt()`` takes no seed argument; every randomized
corruption function it wraps reads directly from numpy's process-global
legacy RNG state, not an injectable :class:`numpy.random.Generator`. This
is the one place in ``wfcrc`` that deliberately touches that global
state — a disclosed, narrow exception to the project's "no bare global
RNG outside ``wfcrc.utils.seeds``" rule, forced by a third-party API with
no alternative. :func:`apply_corruption` saves the prior global state,
seeds it deterministically (via :func:`wfcrc.utils.seeds.derive_seed`),
makes the one ``corrupt()`` call (or, for ``impulse_noise``, the one
``random_noise()`` call with an explicit ``rng=``), and restores the
prior state — so no mutation of global RNG state is ever visible to any
caller before or after this function returns.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import skimage.util
from numpy.typing import NDArray

from wfcrc.exceptions import PreprocessingError
from wfcrc.utils.seeds import derive_seed


def ensure_pkg_resources_shim() -> None:
    """Install a stdlib-only ``pkg_resources.resource_filename`` shim if needed.

    ``imagecorruptions.corruptions`` imports ``pkg_resources`` (removed
    from ``setuptools>=81``) solely to call ``resource_filename`` for its
    bundled frost-texture files. If a real ``pkg_resources`` is already
    importable, this is a no-op — the real one takes precedence. See the
    module docstring's "setuptools<81 dependency" section for the
    equivalence validation of this shim, and its "Import-order requirement"
    section for why any code needing ``imagecorruptions`` directly should
    call this (or import this module) first.
    """
    if "pkg_resources" in sys.modules:
        return
    if importlib.util.find_spec("pkg_resources") is not None:
        return

    def _resource_filename(package_or_requirement: str, resource_name: str) -> str:
        spec = importlib.util.find_spec(package_or_requirement)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(package_or_requirement)
        return str((Path(spec.origin).parent / resource_name).resolve())

    shim = types.ModuleType("pkg_resources")
    shim.resource_filename = _resource_filename  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


ensure_pkg_resources_shim()

from imagecorruptions import corrupt as _corrupt  # noqa: E402
from imagecorruptions import corruptions as _corruptions_module  # noqa: E402

__all__ = [
    "COMPAT_PATH_CORRUPTIONS",
    "CORRUPTION_NAMES",
    "DIRECT_CORRUPTIONS",
    "RANDOMIZED_CORRUPTIONS",
    "apply_corruption",
    "ensure_pkg_resources_shim",
]

#: The 15 canonical "common" Cityscapes-C / ImageNet-C corruption names
#: (Hendrycks & Dietterich), in the order
#: ``imagecorruptions.get_corruption_names("common")`` returns them.
#: Hardcoded explicitly per frozen Q2 ("define the canonical corruption
#: names explicitly, do not silently change corruption parameters");
#: ``tests/unit/datasets/test_corruptions.py`` asserts this tuple stays
#: byte-for-byte in sync with the pinned library's own list, so a future
#: library version bump that silently reorders or renames a corruption is
#: caught by a failing test rather than silently accepted.
CORRUPTION_NAMES: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

#: Subset of :data:`CORRUPTION_NAMES` whose ``imagecorruptions==1.1.2``
#: implementation reads randomness (numpy's global legacy RNG state, or,
#: for ``impulse_noise``'s compatibility path, an explicit modern
#: generator) — verified empirically: same seed -> same output; different
#: seed -> different output. The remaining 6 names are mathematically
#: deterministic given ``(image, severity)`` alone — verified empirically
#: to produce identical output regardless of seed.
RANDOMIZED_CORRUPTIONS: frozenset[str] = frozenset(
    {
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "glass_blur",
        "motion_blur",
        "snow",
        "frost",
        "fog",
        "elastic_transform",
    }
)

#: Corruptions executed directly through the unmodified upstream
#: ``imagecorruptions.corrupt()`` call (12 of 15) — no compatibility path.
DIRECT_CORRUPTIONS: frozenset[str] = frozenset(CORRUPTION_NAMES) - {
    "glass_blur",
    "fog",
    "impulse_noise",
}

#: Corruptions using a narrow, individually-verified compatibility path
#: (MS6.2 Resolution Pass) — see module docstring for the root cause and
#: equivalence validation of each.
COMPAT_PATH_CORRUPTIONS: frozenset[str] = frozenset({"glass_blur", "fog", "impulse_noise"})

_VALID_SEVERITIES = (1, 2, 3, 4, 5)
_MIN_IMAGE_SIDE = 32

#: severity -> `amount` (proportion of pixels replaced), verbatim from
#: ``imagecorruptions.corruptions.impulse_noise``'s own severity table —
#: unchanged by the compatibility path (module docstring, "impulse_noise").
_IMPULSE_NOISE_SEVERITY_TABLE: tuple[float, ...] = (0.03, 0.06, 0.09, 0.17, 0.27)

_ORIGINAL_GAUSSIAN = _corruptions_module.gaussian


def _channel_axis_gaussian(*args: Any, **kwargs: Any) -> NDArray[Any]:
    """Translate the removed skimage ``multichannel=`` kwarg to ``channel_axis=``.

    Verified bit-identical to the removed ``multichannel=`` behavior at
    every sigma :func:`apply_corruption`'s ``glass_blur`` path uses, in
    both color and grayscale (module docstring).
    """
    if "multichannel" in kwargs:
        multichannel = kwargs.pop("multichannel")
        kwargs["channel_axis"] = -1 if multichannel else None
    return _ORIGINAL_GAUSSIAN(*args, **kwargs)  # type: ignore[no-any-return]


@contextlib.contextmanager
def _glass_blur_compat() -> Iterator[None]:
    """Swap in :func:`_channel_axis_gaussian` for the duration of one ``corrupt()`` call."""
    _corruptions_module.gaussian = _channel_axis_gaussian
    try:
        yield
    finally:
        _corruptions_module.gaussian = _ORIGINAL_GAUSSIAN


@contextlib.contextmanager
def _fog_compat() -> Iterator[None]:
    """Restore the removed ``np.float_`` alias for the duration of one ``corrupt()`` call."""
    had_attr = hasattr(np, "float_")
    prior = getattr(np, "float_", None)
    setattr(np, "float_", np.float64)  # noqa: B010
    try:
        yield
    finally:
        if had_attr:
            setattr(np, "float_", prior)  # noqa: B010
        else:
            delattr(np, "float_")


def _impulse_noise_compat(
    image_uint8: NDArray[Any], severity: int, rng_seed: int
) -> NDArray[np.uint8]:
    """Reimplement ``impulse_noise``'s 3-line wrapper with an explicit modern ``rng=``.

    The severity table and clip/scale-to-uint8 steps are verbatim from
    ``imagecorruptions.corruptions.impulse_noise``; the noise-generating
    call itself is the unmodified, upstream
    :func:`skimage.util.random_noise`, given ``rng=rng_seed`` instead of
    relying on that library's broken implicit-global-state fallback (see
    module docstring, "impulse_noise", for the full equivalence analysis).
    """
    amount = _IMPULSE_NOISE_SEVERITY_TABLE[severity - 1]
    noised: NDArray[np.float64] = skimage.util.random_noise(  # type: ignore[no-untyped-call]
        np.array(image_uint8) / 255.0, mode="s&p", amount=amount, rng=rng_seed
    )
    return (np.clip(noised, 0, 1) * 255).astype(np.uint8)


def apply_corruption(
    image: NDArray[Any], corruption_name: str, severity: int, seed: int
) -> NDArray[np.uint8]:
    """Apply one Cityscapes-C corruption to a raw image, deterministically.

    Args:
        image: Raw pixel array, dtype ``uint8``, shape ``(H, W)`` or
            ``(H, W, C)`` with ``C`` in ``{1, 3}`` and both ``H, W >= 32``
            (the underlying library's own constraint, validated here so
            the failure is a clear :class:`PreprocessingError`, not a
            third-party ``AttributeError``).
        corruption_name: One of :data:`CORRUPTION_NAMES`.
        severity: Integer in ``[1, 5]``.
        seed: Base seed; the actual per-call seed is derived via
            :func:`wfcrc.utils.seeds.derive_seed` from
            ``(corruption_name, severity, seed)``, so distinct
            ``(corruption_name, severity)`` pairs never share a derived
            seed even when called with the same base ``seed``.

    Returns:
        The corrupted image, dtype ``uint8``, same shape as ``image``. See
        the module docstring for which corruptions this calls directly
        (:data:`DIRECT_CORRUPTIONS`) versus through a verified
        compatibility path (:data:`COMPAT_PATH_CORRUPTIONS`).

    Raises:
        PreprocessingError: If ``corruption_name`` is not one of
            :data:`CORRUPTION_NAMES`, ``severity`` is not in ``[1, 5]``,
            or ``image`` does not satisfy the dtype/shape contract above.
    """
    if corruption_name not in CORRUPTION_NAMES:
        raise PreprocessingError(
            f"unknown corruption {corruption_name!r}; must be one of {CORRUPTION_NAMES}"
        )
    if severity not in _VALID_SEVERITIES:
        raise PreprocessingError(f"severity must be one of {_VALID_SEVERITIES}, got {severity}")

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise PreprocessingError(
            f"image must be dtype uint8 (raw 0-255 pixel values), got {arr.dtype}"
        )
    if arr.ndim not in (2, 3):
        raise PreprocessingError(f"image must be 2-D or 3-D, got ndim={arr.ndim}")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3):
        raise PreprocessingError(f"image must have 1 or 3 channels, got {arr.shape[2]}")
    if arr.shape[0] < _MIN_IMAGE_SIDE or arr.shape[1] < _MIN_IMAGE_SIDE:
        raise PreprocessingError(
            f"image height and width must each be >= {_MIN_IMAGE_SIDE}, got {arr.shape[:2]}"
        )

    derived_seed = derive_seed(f"corruption.{corruption_name}.{severity}", seed)

    if corruption_name == "impulse_noise":
        return _impulse_noise_compat(arr, severity, derived_seed)

    prior_state = np.random.get_state()
    try:
        np.random.seed(derived_seed)
        if corruption_name == "glass_blur":
            with _glass_blur_compat():
                corrupted = _corrupt(arr, corruption_name=corruption_name, severity=severity)
        elif corruption_name == "fog":
            with _fog_compat():
                corrupted = _corrupt(arr, corruption_name=corruption_name, severity=severity)
        else:
            corrupted = _corrupt(arr, corruption_name=corruption_name, severity=severity)
    finally:
        np.random.set_state(prior_state)

    return np.asarray(corrupted, dtype=np.uint8)
