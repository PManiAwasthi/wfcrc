"""Checkpoint Management — minimal, inference-only checkpoint handling (MS7, reduced §3.5 scope).

Per the MS6 Architecture Specification §3.5, full Checkpoint Management also
includes a `CheckpointProvenance`/`assert_no_checkpoint_leakage` pair
guarding against R-CKPT1 ("a public pretrained checkpoint whose training
split overlaps the calibration pool would leak"). That check is deliberately
**not implemented here**: it exists to protect against leakage from a
checkpoint's own (external, undisclosed) *training* history, and the MS7
checkpoint this module loads was never trained on anything at all (module
docstring of :mod:`wfcrc.models.scores.hippocampus_segmenter` — a
deterministically seeded, randomly initialized network, framework-validation
only) — there is no training-time data exposure for the check to guard
against. Implementing that dataclass/function against a checkpoint that
structurally cannot leak would be exercising an as-yet-untestable code path
rather than closing a real gap; it is deferred to whichever future milestone
first loads a genuinely externally-*trained* checkpoint, consistent with the
MS7 task brief's own "no training" scope boundary.

This module implements exactly the four items the MS7 task brief's Task 3
asks for: checkpoint discovery, loading, inference mode, device selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from wfcrc.utils.io import content_hash

__all__ = ["checkpoint_fingerprint", "load_checkpoint"]


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Discover and load a PyTorch checkpoint's state dict from disk.

    Args:
        path: Path to a checkpoint file previously written by
            :func:`torch.save` (e.g. ``model.state_dict()``).
        device: Device to map loaded tensors onto (device selection);
            defaults to ``"cpu"`` — this project has no GPU dependency
            anywhere (Q1, `MS6_ARCHITECTURE_SPEC.md` §8.1).

    Returns:
        The loaded state dict, ready to pass to
        ``torch.nn.Module.load_state_dict``.

    Raises:
        FileNotFoundError: If ``path`` does not exist (checkpoint
            discovery failure) — a clear, explicit failure rather than a
            silent empty score, per `MS6_ARCHITECTURE_SPEC.md` §6's own
            MS6.4 failure-mode table.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    state_dict: dict[str, torch.Tensor] = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    return state_dict


def checkpoint_fingerprint(path: str | Path) -> str:
    """Return a stable content-hash fingerprint of a checkpoint file.

    Reuses the frozen :func:`wfcrc.utils.io.content_hash` (per
    `MS6_ARCHITECTURE_SPEC.md` §3.5), so that scores from two different
    checkpoint files never collide under the same cache key even if a
    caller reuses the same dataset ids.

    **Fingerprints the file, not "the weights."** ``torch.save``'s own
    container format is not guaranteed byte-identical across independent
    calls even given identical tensor content (empirically verified,
    MS7: two checkpoints written from the same seed produce the same
    tensor values but different file bytes) — this function therefore
    distinguishes *which literal checkpoint file* produced a given cached
    score, which is exactly its documented cache-collision-avoidance
    purpose, but it is **not** a semantic "same weights" equality check.

    Args:
        path: Path to the checkpoint file.

    Returns:
        A stable hex digest string identifying this exact checkpoint
        file's byte content.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    raw: Any = checkpoint_path.read_bytes().hex()
    return content_hash(raw)
