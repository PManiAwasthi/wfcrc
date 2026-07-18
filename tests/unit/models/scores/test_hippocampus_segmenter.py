"""Unit tests for :mod:`wfcrc.models.scores.hippocampus_segmenter` (MS7).

All tests run against tiny **synthetic** volumes — never real downloaded
data — retaining the opt-in real-data philosophy established during
MS6.3A. See `test_hippocampus_pipeline_real_data.py` for the opt-in,
marker-gated end-to-end pipeline test against real MSD Task04_Hippocampus
data.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from wfcrc.datasets.base import Dataset
from wfcrc.models.checkpoint import checkpoint_fingerprint, load_checkpoint
from wfcrc.models.scores.hippocampus_segmenter import (
    _SHAPE_MULTIPLE,
    HippocampusScoreProvider,
    _pad_to_multiple,
    _TinyUNet3D,
    _zscore_normalize,
    create_untrained_checkpoint,
)

SHAPE = (9, 11, 13)


class _FakeVolumeDataset(Dataset):
    """A minimal synthetic 3-D `Dataset` double: `n` small volumes."""

    def __init__(self, ids: Sequence[str], *, shape: tuple[int, int, int] = SHAPE) -> None:
        self._ids = tuple(ids)
        rng = np.random.default_rng(0)
        self._images = {i: rng.uniform(0.0, 1900.0, size=shape) for i in self._ids}
        self._labels = {i: rng.uniform(0.0, 1.0, size=shape) > 0.8 for i in self._ids}

    def __iter__(self) -> Iterator[tuple[Hashable, Any, Any]]:
        for i in self._ids:
            yield i, self._images[i], self._labels[i]

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> Sequence[Hashable]:
        return self._ids

    def labels(self, id_: Hashable) -> Any:
        return self._labels[id_]

    def meta(self) -> dict[str, Any]:
        return {"version": "fake", "license": "fake"}


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / "ckpt.pt"
    create_untrained_checkpoint(path, seed=0)
    return path


@pytest.fixture
def fake_dataset() -> _FakeVolumeDataset:
    return _FakeVolumeDataset(["a", "b", "c"])


# --- wfcrc.models.checkpoint --------------------------------------------------


def test_create_untrained_checkpoint_writes_a_loadable_file(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    create_untrained_checkpoint(path, seed=1)
    assert path.is_file()
    state_dict = load_checkpoint(path)
    assert "out_conv.weight" in state_dict


def test_load_checkpoint_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_checkpoint_fingerprint_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint_fingerprint(tmp_path / "does_not_exist.pt")


def test_checkpoint_fingerprint_stable_across_loads(checkpoint_path: Path) -> None:
    assert checkpoint_fingerprint(checkpoint_path) == checkpoint_fingerprint(checkpoint_path)


def test_checkpoint_fingerprint_differs_across_checkpoints(tmp_path: Path) -> None:
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    create_untrained_checkpoint(path_a, seed=1)
    create_untrained_checkpoint(path_b, seed=2)
    assert checkpoint_fingerprint(path_a) != checkpoint_fingerprint(path_b)


def test_same_seed_produces_identical_weights(tmp_path: Path) -> None:
    # torch.save's own container format is not guaranteed byte-identical
    # across independent calls even for identical tensor content (verified:
    # same length, different bytes) -- checkpoint_fingerprint therefore
    # fingerprints *the file*, not "any checkpoint from this seed", matching
    # its documented purpose (distinguishing which literal checkpoint file
    # produced a given cached score). What create_untrained_checkpoint
    # actually promises to be reproducible is the *tensor values*.
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    create_untrained_checkpoint(path_a, seed=7)
    create_untrained_checkpoint(path_b, seed=7)
    state_a = load_checkpoint(path_a)
    state_b = load_checkpoint(path_b)
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)


# --- private helpers (_pad_to_multiple, _zscore_normalize) --------------------


def test_pad_to_multiple_pads_and_reports_original_shape() -> None:
    x = torch.zeros(1, 1, 9, 11, 13)
    padded, original = _pad_to_multiple(x, _SHAPE_MULTIPLE)
    assert original == (9, 11, 13)
    assert padded.shape[2] % _SHAPE_MULTIPLE == 0
    assert padded.shape[3] % _SHAPE_MULTIPLE == 0
    assert padded.shape[4] % _SHAPE_MULTIPLE == 0
    assert padded.shape[2] >= 9 and padded.shape[3] >= 11 and padded.shape[4] >= 13


def test_pad_to_multiple_noop_when_already_aligned() -> None:
    x = torch.zeros(1, 1, 8, 8, 8)
    padded, original = _pad_to_multiple(x, _SHAPE_MULTIPLE)
    assert padded.shape == x.shape
    assert original == (8, 8, 8)


def test_zscore_normalize_zero_mean_unit_std() -> None:
    x = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
    normalized = _zscore_normalize(x)
    assert normalized.mean().abs().item() < 1e-5
    assert abs(normalized.std().item() - 1.0) < 1e-3


def test_zscore_normalize_constant_input_does_not_divide_by_zero() -> None:
    x = torch.full((5,), 42.0)
    normalized = _zscore_normalize(x)
    assert torch.isfinite(normalized).all()


# --- HippocampusScoreProvider ---------------------------------------------------


def test_scores_for_returns_bool_array_matching_image_shape(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    provider = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    score = provider.scores_for("a")
    assert score.dtype == np.bool_
    assert score.shape == SHAPE


def test_scores_for_unknown_id_raises_key_error(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    provider = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    with pytest.raises(KeyError):
        provider.scores_for("not-a-real-id")


def test_scores_batch_matches_scores_for(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    provider = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    batch = provider.scores_batch(fake_dataset.ids())
    individual = [provider.scores_for(id_) for id_ in fake_dataset.ids()]
    assert all(np.array_equal(a, b) for a, b in zip(batch, individual, strict=True))


def test_inference_is_deterministic_across_calls(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    provider = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    first = provider.scores_for("a")
    second = provider.scores_for("a")
    np.testing.assert_array_equal(first, second)


def test_two_providers_from_same_checkpoint_agree(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    provider_1 = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    provider_2 = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    np.testing.assert_array_equal(provider_1.scores_for("a"), provider_2.scores_for("a"))


def test_model_fingerprint_matches_checkpoint_fingerprint(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    provider = HippocampusScoreProvider(checkpoint_path, [fake_dataset])
    assert provider.model_fingerprint() == checkpoint_fingerprint(checkpoint_path)


def test_threshold_out_of_range_raises_value_error(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        HippocampusScoreProvider(checkpoint_path, [fake_dataset], threshold=1.5)


def test_multiple_datasets_combine_into_one_lookup(checkpoint_path: Path) -> None:
    ds_a = _FakeVolumeDataset(["a1", "a2"])
    ds_b = _FakeVolumeDataset(["b1", "b2"])
    provider = HippocampusScoreProvider(checkpoint_path, [ds_a, ds_b])
    for id_ in ("a1", "a2", "b1", "b2"):
        score = provider.scores_for(id_)
        assert score.shape == SHAPE


def test_cache_hit_avoids_second_inference_call(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset, tmp_path: Path
) -> None:
    provider = HippocampusScoreProvider(
        checkpoint_path, [fake_dataset], cache_dir=tmp_path / "cache"
    )
    with patch.object(provider, "_infer", wraps=provider._infer) as mocked_infer:
        provider.scores_for("a")
        provider.scores_for("a")
        assert mocked_infer.call_count == 1


def test_cache_round_trip_preserves_dtype_and_values(
    checkpoint_path: Path, fake_dataset: _FakeVolumeDataset, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    provider_1 = HippocampusScoreProvider(checkpoint_path, [fake_dataset], cache_dir=cache_dir)
    uncached = provider_1.scores_for("a")

    provider_2 = HippocampusScoreProvider(checkpoint_path, [fake_dataset], cache_dir=cache_dir)
    cached = provider_2.scores_for("a")

    assert cached.dtype == np.bool_
    np.testing.assert_array_equal(uncached, cached)


def test_odd_shaped_volume_is_handled_via_padding(checkpoint_path: Path) -> None:
    # A shape not divisible by _SHAPE_MULTIPLE, exercising the padding path
    # with a nonzero remainder on every axis.
    dataset = _FakeVolumeDataset(["odd"], shape=(7, 9, 5))
    provider = HippocampusScoreProvider(checkpoint_path, [dataset])
    score = provider.scores_for("odd")
    assert score.shape == (7, 9, 5)


# --- _TinyUNet3D architecture ---------------------------------------------------


def test_tiny_unet_output_shape_matches_input() -> None:
    model = _TinyUNet3D(base_channels=4)
    model.eval()
    x = torch.zeros(1, 1, 8, 8, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 8, 8, 8)


def test_tiny_unet_final_bias_is_zero_initialized() -> None:
    model = _TinyUNet3D()
    assert model.out_conv.bias is not None
    assert torch.all(model.out_conv.bias == 0.0)


def test_tiny_unet_parameter_count_is_small() -> None:
    model = _TinyUNet3D(base_channels=8)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000
