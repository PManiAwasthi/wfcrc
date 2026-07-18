"""``HippocampusScoreProvider`` — the MS7 concrete ``ScoreProvider`` for MSD Task04_Hippocampus.

**Model selection (MS7 Task, "choose ONE lightweight, well-established
segmentation baseline").** A compact 3-D U-Net (Çiçek et al. 2016, "3D
U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" — the
direct volumetric generalization of Ronneberger et al.'s 2015 U-Net) is
implemented here as :class:`_TinyUNet3D`, scored against the task's own
criteria:

- **Reproducible.** No stochastic layers (no dropout, no batch
  normalization — the latter is deliberately omitted because inference here
  always runs at batch size 1, where batchnorm's running statistics are
  degenerate); given a fixed initialization seed, every forward pass is a
  pure function of its input.
- **Stable / widely used.** The 3-D U-Net encoder-decoder-with-skip-
  connections family is the standard architecture for volumetric medical
  segmentation; nnU-Net (built on exactly this family) is the dominant
  approach across Medical Segmentation Decathlon leaderboards, and
  `MS6_ARCHITECTURE_SPEC.md` §3.4 already anticipates "an nnU-Net-style
  architecture for MSD" as the expected model family for this exact
  dataset.
- **Lightweight / easy to verify.** Two downsampling levels, 8/16/32
  channels — on the order of 40,000 parameters, trivial CPU inference on
  volumes this small (MS6.3A real-data validation: shapes in the
  neighborhood of `30-45` voxels per axis), and simple enough that its
  encoder/decoder/skip-connection shape flow is straightforward to unit
  test directly.
- **Compatible with future Pancreas experiments.** The identical
  architecture family scales to Task07_Pancreas (more channels/levels for
  the harder task) without a different architecture class — matching the
  Dataset Selection Audit's own framing of the two MSD tasks as
  "cheap-statistics vs hard-conditional" duals within one modality.

No PyTorch pretrained/public checkpoint for Hippocampus segmentation was
available in this environment (`MS7` architecture-audit step; a repo-wide
search for `*.pt`/`*.pth`/`*.ckpt`/`*.h5`/`*.onnx`/`*.safetensors` found
none). Per the MS7 task brief's own explicit constraints — "Do not
implement training. No optimizer. No scheduler. No fine-tuning. Inference
only" (Task 3) and "The objective is NOT accuracy... proving that the
complete WFCRC pipeline executes successfully" (Task 6) — the only
coherent resolution is an **inference-only run on a deterministically,
randomly initialized (never trained) network**. :func:`create_untrained_checkpoint`
does exactly this and nothing else (no forward+backward step, no
optimizer, ever constructed); its own docstring repeats this disclosure so
it cannot be mistaken for a clinically meaningful model. This is a
framework-validation smoke test, not a segmentation-accuracy claim.

**Tensor/interface contracts (MS7 Task 1/2, fully documented).**

- **Input.** One MRI volume, `(D, H, W)` `float64` (as
  :class:`~wfcrc.datasets.loaders.msd.MSDDataset` yields via `__iter__`),
  no channel axis (single modality). Internally cast to `float32` and
  reshaped to `(1, 1, D, H, W)` (batch=1, channels=1) for
  :class:`torch.nn.Conv3d`.
- **Padding.** MSD Hippocampus volumes have no fixed shape (236 distinct
  shapes across the 260 real cases, MS6.3A validation). `_TinyUNet3D` has
  two `MaxPool3d(2)` stages, so each spatial dimension must be a multiple
  of 4 for the encoder/decoder skip-connections to align exactly;
  :func:`_pad_to_multiple` zero-pads up to the next multiple of 4 per axis
  before the forward pass, and the output is cropped back to the original
  shape afterward — the padding is purely a shape-alignment mechanism, not
  a preprocessing policy, and is invisible to the returned score's shape.
- **Output — the frozen-interface adaptation this module exists to
  disclose.** `wfcrc.datasets.score_provider.ScoreArray` is declared as
  `NDArray[np.float64]`, but the frozen
  :meth:`wfcrc.prediction_sets.segmentation.MorphologicalSets.construct`
  (MS3, unmodified) requires its `score` argument to have **literal**
  `dtype == numpy.bool_` — a plain `float64` array of `0.0`/`1.0` values
  does not satisfy `mask.dtype != np.bool_` and would be rejected at
  runtime. `PredictionSetConstructor.construct`'s own ABC-level parameter
  type is `ArrayLike` (`wfcrc/prediction_sets/base.py`), which is
  permissive enough to accept either representation — the friction is
  specifically between `ScoreProvider.scores_for`'s declared *float64*
  return alias and `MorphologicalSets`'s actual *bool* runtime requirement,
  for this one (segmentation-with-dilation-margin-sets) constructor
  pairing. Per the MS7 task brief's own instruction ("Do not modify
  LossTableBuilder. Instead adapt the ScoreProvider to the frozen
  contract"), :meth:`HippocampusScoreProvider.scores_for` returns a
  genuine `NDArray[np.bool_]` — the network's raw single-channel logit is
  passed through a sigmoid to a foreground probability, then thresholded
  at a fixed, documented operating point (`threshold`, default `0.5`) into
  a boolean seed mask `M₀`, exactly what `MorphologicalSets.construct(score,
  lam)` needs (`C_λ = dilate(M₀, ⌊λ⌋)`). No frozen file (`ScoreProvider`,
  `MorphologicalSets`, `LossTableBuilder`) is modified; this is a disclosed,
  narrow typing accommodation in one concrete subclass, not a redesign.
- **Batching.** :meth:`scores_batch` is a thin per-id loop over
  :meth:`scores_for` (no batched tensor inference — MSD Hippocampus
  volumes have no common shape to batch without padding every volume to
  the maximum shape in the batch, which this module does not attempt,
  since nothing in MS7's scope requires throughput optimization).

**Where the raw image comes from — a second, disclosed interface
adaptation.** `ScoreProvider.scores_for(self, id_)` takes only an `id_`
(`wfcrc/datasets/score_provider.py`, frozen) — it has no path to a
`Dataset` instance, and `Dataset` itself (`wfcrc/datasets/base.py`, frozen)
exposes no per-id, image-only random-access method (`labels(id_)` is
label-only; `__iter__` yields every `(id, image, label)` triple but not on
demand for one id). `MS6_ARCHITECTURE_SPEC.md` §3.4's own constructor
sketch (`TorchScoreProvider(checkpoint_path, cache_dir, device)`) likewise
carries no `dataset` parameter, implying a concrete `ScoreProvider` is
expected to resolve `id_ -> image` independently. Rather than duplicate
:mod:`wfcrc.datasets.loaders.msd`'s private file-discovery/NIfTI-reading
internals (which would violate MS6.3A's own "no preprocessing logic
duplicated" discipline) or reach into that module's private attributes,
:class:`HippocampusScoreProvider` takes one or more already-constructed
:class:`~wfcrc.datasets.loaders.msd.MSDDataset` splits at construction
(typically calibration + test together, since `LossTableBuilder.build()`
is called once per split against the *same* `ScoreProvider` instance/
`model_fingerprint` — `MS6_ARCHITECTURE_SPEC.md` §3.10's own
`build_loss_tables` sketch shares one `score_provider` across both calls)
and consumes **only their frozen, public** `__iter__` contract, once per
dataset, to build an internal `id -> image` lookup table spanning all of
them. This is a disclosed, additive constructor parameter beyond §3.4's
literal sketch (the same pattern already used, and already disclosed, for
`MSDNiftiLoader`'s own `split_manifest` parameter in MS6.3A) — no frozen
file is touched.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as torch_functional

from wfcrc.datasets.loaders.msd import MSDDataset
from wfcrc.datasets.score_provider import ScoreArray, ScoreBatch, ScoreProvider
from wfcrc.models.checkpoint import checkpoint_fingerprint, load_checkpoint
from wfcrc.utils.cache import Cache, make_key

__all__ = ["HippocampusScoreProvider", "create_untrained_checkpoint"]

#: Number of `MaxPool3d(2)` stages in `_TinyUNet3D`; every spatial axis fed
#: to the network must be a multiple of `2 ** _NUM_POOL_STAGES` for the
#: encoder/decoder skip connections to align exactly after cropping.
_NUM_POOL_STAGES = 2
_SHAPE_MULTIPLE = 2**_NUM_POOL_STAGES


class _ConvBlock(nn.Module):
    """Two `Conv3d(k=3) + ReLU` layers, channel count `in_c -> out_c -> out_c`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class _TinyUNet3D(nn.Module):
    """A compact 2-level 3-D U-Net; single input channel, single output logit channel.

    See the module docstring for the full architecture justification.
    """

    def __init__(self, *, base_channels: int = 8) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(1, c)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = _ConvBlock(c * 2, c * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = _ConvBlock(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = _ConvBlock(c * 2 + c, c)
        self.out_conv = nn.Conv3d(c, 1, kernel_size=1)
        # Deterministic (not random) init choice, standard practice for a
        # final logit layer: PyTorch's default Conv3d bias init is a small
        # *random* value; left random, it dominates the (weak, untrained)
        # upstream feature signal entirely, collapsing the output to a
        # near-constant sigmoid value for every input regardless of image
        # content (empirically confirmed on real Hippocampus volumes,
        # module docstring). Zeroing it is not a training step (no
        # gradient, no data seen) — it only removes a spurious random
        # offset so the untrained network's small spatial signal is
        # actually visible in its output, which is what makes the MS7
        # smoke experiment a meaningful exercise of `MorphologicalSets`'
        # dilation-growth behavior rather than an always-empty/always-full
        # degenerate mask.
        assert self.out_conv.bias is not None  # always True: bias=True is Conv3d's own default
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(N, 1, D, H, W)` (D/H/W each a multiple of 4) to `(N, 1, D, H, W)` raw logits."""
        skip1 = self.enc1(x)
        x = self.pool1(skip1)
        skip2 = self.enc2(x)
        x = self.pool2(skip2)
        x = self.bottleneck(x)
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x)
        logits: torch.Tensor = self.out_conv(x)
        return logits


def _pad_to_multiple(
    volume: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Zero-pad a `(N, C, D, H, W)` tensor so `D`/`H`/`W` are each a multiple of `multiple`.

    Args:
        volume: A `(N, C, D, H, W)` tensor.
        multiple: The required divisor for each spatial dimension.

    Returns:
        `(padded, original_shape)`: the padded tensor, and the original
        `(D, H, W)` to crop back to after the forward pass.
    """
    _, _, d, h, w = volume.shape
    pad_d = (-d) % multiple
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    # F.pad takes padding in reverse-dim order: (W_left,W_right,H_left,H_right,D_left,D_right).
    padded = torch_functional.pad(
        volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0
    )
    return padded, (d, h, w)


def _zscore_normalize(volume: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Per-volume z-score normalize: `(volume - mean) / (std + eps)`.

    Real MSD Hippocampus MRI intensities are raw scanner units, not scaled
    to `[0, 1]` or `[-1, 1]` (real-data validation, MS7: one real case
    ranged `0`-`1932`, mean `~596`, std `~273`). Feeding that directly into
    a network initialized under the standard assumption of roughly
    unit-scale input saturates every activation, collapsing the sigmoid
    output to a near-constant value regardless of image content — verified
    directly: without this normalization, several `torch.manual_seed`
    values produced an entirely empty or entirely full seed mask for every
    real volume tested. Per-volume (not per-dataset-constant) z-score
    normalization is the standard, uncontroversial preprocessing step for
    CT/MRI CNN input (matching `DATASET_METADATA["msd_hippocampus"]`'s own
    documented `"preprocessing": "Resample/normalize (nnU-Net)"` note) and
    is intentionally **not** implemented via the frozen
    :func:`wfcrc.datasets.preprocessing.resize_and_normalize` (MS6.2): that
    function is 2-D-only and requires an externally-supplied, fixed
    per-channel `mean`/`std` constant — the opposite of what per-volume
    normalization needs (a statistic computed from each volume itself).
    This is model-input preparation, squarely a `ScoreProvider`
    responsibility, not dataset preprocessing; it never touches
    `MSDDataset`'s own returned arrays.

    Args:
        volume: A tensor of any shape (used here on `(D, H, W)` volumes).
        eps: Numerical floor added to the standard deviation.

    Returns:
        The normalized tensor, same shape as `volume`.
    """
    return (volume - volume.mean()) / (volume.std() + eps)


def create_untrained_checkpoint(path: str | Path, *, seed: int, base_channels: int = 8) -> None:
    """Save a deterministically, randomly initialized `_TinyUNet3D` checkpoint.

    **This is not a trained model.** No forward pass, loss, optimizer, or
    backward step ever runs here — this function only constructs the
    network under a fixed PyTorch RNG seed (making its initial weights
    reproducible) and writes `state_dict()` to disk, exactly as if it were
    any other checkpoint file. It exists solely so
    :class:`HippocampusScoreProvider` (MS7's inference-only, "no training"
    scope) has a real checkpoint file to discover/load/fingerprint,
    matching Checkpoint Management's expected shape (§3.5) without
    claiming any clinical validity — see the module docstring for the full
    "no training" rationale (MS7 Task 3/6).

    Args:
        path: Destination checkpoint file path.
        seed: PyTorch RNG seed for weight initialization (deterministic
            and reproducible for a fixed value — pass a value derived via
            :func:`wfcrc.utils.seeds.derive_seed` to keep this within the
            project's own reproducibility-provenance framework, since
            `torch`'s RNG is a separate system from
            :mod:`wfcrc.utils.seeds`'s own numpy-based one, confined here
            to `wfcrc/models/` per Q1).
        base_channels: Passed through to `_TinyUNet3D` — must match what
            :class:`HippocampusScoreProvider` is constructed with, or
            `load_state_dict` will fail with a shape mismatch (an
            intentional, explicit failure, not a silent one).
    """
    torch.manual_seed(seed)
    model = _TinyUNet3D(base_channels=base_channels)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


class HippocampusScoreProvider(ScoreProvider):
    """Inference-only `_TinyUNet3D` `ScoreProvider` for MSD Task04_Hippocampus (MS7).

    See the module docstring for the full model-selection justification,
    tensor/shape contracts, and the two disclosed frozen-interface
    adaptations (bool-dtype `ScoreArray`; dataset-`__iter__`-sourced image
    lookup).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        datasets: Iterable[MSDDataset],
        *,
        cache_dir: str | Path | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        base_channels: int = 8,
    ) -> None:
        """Load the checkpoint and prepare for inference.

        Args:
            checkpoint_path: Path to a checkpoint written by
                :func:`create_untrained_checkpoint` (or any future
                `_TinyUNet3D`-compatible checkpoint).
            datasets: One or more `MSDDataset` splits whose ids this
                provider may be asked to score — typically the
                calibration and test splits together, since one
                `ScoreProvider` instance (one `model_fingerprint`) is
                expected to serve `LossTableBuilder.build()` calls across
                both (`MS6_ARCHITECTURE_SPEC.md` §3.10's own
                `build_loss_tables` sketch takes a single shared
                `score_provider` for the whole pipeline). Each dataset's
                frozen, public `__iter__` is consumed exactly once, at
                construction, to build an internal `id -> image` lookup
                spanning all of them (module docstring: "Where the raw
                image comes from"). Passing only one split's `Dataset`
                (e.g. in a single-split unit test) is also valid.
            cache_dir: Optional read-through score cache directory
                (`wfcrc.utils.cache.Cache`, keyed on
                `(model_fingerprint, id_)`); if `None`, no caching occurs
                and every `scores_for` call re-runs inference.
            device: Inference device; `"cpu"` only is exercised/verified
                in this environment (no GPU dependency anywhere, Q1).
            threshold: Fixed foreground-probability operating point (in
                `[0, 1]`) used to binarize the network's sigmoid output
                into the boolean seed mask `MorphologicalSets` requires.
            base_channels: Must match the checkpoint's own
                `base_channels` (see :func:`create_untrained_checkpoint`).

        Raises:
            FileNotFoundError: If `checkpoint_path` does not exist
                (propagated from :func:`wfcrc.models.checkpoint.load_checkpoint`).
            ValueError: If `threshold` is outside `[0, 1]`.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._device = torch.device(device)
        self._threshold = threshold
        self._model = _TinyUNet3D(base_channels=base_channels)
        state_dict = load_checkpoint(checkpoint_path, device=str(self._device))
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self._fingerprint = checkpoint_fingerprint(checkpoint_path)
        self._images: dict[Hashable, NDArray[np.float64]] = {
            id_: image for dataset in datasets for id_, image, _ in dataset
        }
        self._cache = Cache(cache_dir) if cache_dir is not None else None

    def scores_for(self, id_: Hashable) -> ScoreArray:
        """Return the (cached, or freshly computed) boolean seed mask for `id_`.

        Args:
            id_: An example id, as returned by the `MSDDataset.ids()` this
                provider was constructed against.

        Returns:
            A boolean array (see module docstring, "Output"), same shape
            as the source image, `True` where the sigmoid foreground
            probability is `>= threshold`.

        Raises:
            KeyError: If `id_` was not present in the `dataset` this
                provider was constructed from.
        """
        if id_ not in self._images:
            raise KeyError(f"unknown id for this ScoreProvider: {id_!r}")
        if self._cache is None:
            # ScoreArray is declared NDArray[float64] (wfcrc/datasets/score_provider.py,
            # frozen), but MorphologicalSets.construct() (wfcrc/prediction_sets/segmentation.py,
            # frozen) requires a literal bool-dtype seed mask -- see module
            # docstring, "Output", for the full disclosure of this narrow,
            # necessary adaptation.
            return self._infer(id_)  # type: ignore[return-value]
        key = make_key(self._fingerprint, str(id_))
        cached = self._cache.get_or_compute(key, lambda: self._infer(id_))
        return np.asarray(cached, dtype=np.bool_)

    def scores_batch(self, ids: Sequence[Hashable]) -> ScoreBatch:
        """Return scores for a batch of ids, in order (no batched tensor inference).

        See the module docstring's "Batching" paragraph.
        """
        return [self.scores_for(id_) for id_ in ids]

    def model_fingerprint(self) -> str:
        """Return the checkpoint file's stable content-hash fingerprint."""
        return self._fingerprint

    def _infer(self, id_: Hashable) -> NDArray[np.bool_]:
        """Run one forward pass for `id_` and threshold it into a boolean seed mask."""
        image = self._images[id_]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
            dtype=torch.float32, device=self._device
        )
        tensor = _zscore_normalize(tensor)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        padded, original_shape = _pad_to_multiple(tensor, _SHAPE_MULTIPLE)
        with torch.no_grad():
            logits = self._model(padded)
        d, h, w = original_shape
        logits = logits[..., :d, :h, :w]
        probability = torch.sigmoid(logits)
        mask = (probability >= self._threshold).squeeze(0).squeeze(0)
        return mask.cpu().numpy().astype(np.bool_)
