"""Unit tests for :mod:`wfcrc.datasets.metadata` (MS6.2 Resolution Pass, Task 5)."""

from __future__ import annotations

import dataclasses

import pytest

from wfcrc.datasets.metadata import DATASET_METADATA, DatasetMetadata

#: The eight Phase-A dataset keys, per the Dataset Selection Audit §5
#: Acquisition Plan table and Experiment Environment Audit §6's
#: `configs/dataset_*.yaml` naming convention.
_EXPECTED_PHASE_A_KEYS = frozenset(
    {
        "cityscapes",
        "cityscapes_c",
        "acdc",
        "msd_hippocampus",
        "msd_pancreas",
        "cifar10",
        "cifar10_1",
        "kvasir_seg",
    }
)


def test_to_dict_round_trips_every_field() -> None:
    meta = DatasetMetadata(
        name="example",
        version="v1",
        license="MIT",
        source_url="example.com",
        extra={"note": "x"},
    )
    assert meta.to_dict() == {
        "name": "example",
        "version": "v1",
        "license": "MIT",
        "source_url": "example.com",
        "extra": {"note": "x"},
    }


def test_to_dict_extra_defaults_to_empty_dict() -> None:
    meta = DatasetMetadata(name="example", version="v1", license="MIT", source_url="example.com")
    assert meta.to_dict()["extra"] == {}


def test_dataset_metadata_is_immutable() -> None:
    meta = DatasetMetadata(name="example", version="v1", license="MIT", source_url="example.com")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.name = "changed"  # type: ignore[misc]


def test_registry_has_exactly_the_frozen_phase_a_keys() -> None:
    assert set(DATASET_METADATA) == _EXPECTED_PHASE_A_KEYS


def test_registry_keys_match_each_entrys_own_name_field() -> None:
    for key, meta in DATASET_METADATA.items():
        assert meta.name == key


@pytest.mark.parametrize("key", sorted(_EXPECTED_PHASE_A_KEYS))
def test_every_entry_has_non_empty_required_fields(key: str) -> None:
    meta = DATASET_METADATA[key]
    assert isinstance(meta.version, str) and meta.version != ""
    assert isinstance(meta.license, str) and meta.license != ""
    assert isinstance(meta.source_url, str) and meta.source_url != ""


@pytest.mark.parametrize("key", sorted(_EXPECTED_PHASE_A_KEYS))
def test_every_entry_to_dict_is_json_shaped(key: str) -> None:
    rendered = DATASET_METADATA[key].to_dict()
    assert set(rendered) == {"name", "version", "license", "source_url", "extra"}
    assert isinstance(rendered["extra"], dict)


#: Keys whose `license` field is honestly recorded as NOT independently
#: confirmable (MS6.2 Resolution Pass, Task 5) rather than resolved or
#: guessed — see the module docstring's per-entry rationale.
_LICENSE_NOT_CONFIRMABLE_KEYS = frozenset({"acdc", "kvasir_seg"})

_NOT_CONFIRMABLE_MARKER = "NOT independently confirmable"


def test_no_version_field_uses_the_old_blanket_unverified_sentinel() -> None:
    # MS6.2 Resolution Pass (Task 5): every dataset now has a precise
    # provenance identifier (paper/challenge citation) instead of the
    # initial pass's blanket "UNVERIFIED" placeholder -- none of these
    # datasets is formally software-versioned, so the identifier itself
    # documents that absence rather than a version number.
    for meta in DATASET_METADATA.values():
        assert not meta.version.startswith("UNVERIFIED")
        assert len(meta.version) > 20  # a real citation, not a bare placeholder


@pytest.mark.parametrize("key", sorted(_EXPECTED_PHASE_A_KEYS - _LICENSE_NOT_CONFIRMABLE_KEYS))
def test_resolved_license_fields_do_not_carry_the_not_confirmable_marker(key: str) -> None:
    assert _NOT_CONFIRMABLE_MARKER not in DATASET_METADATA[key].license


@pytest.mark.parametrize("key", sorted(_LICENSE_NOT_CONFIRMABLE_KEYS))
def test_unconfirmable_license_fields_are_precisely_flagged_not_guessed(key: str) -> None:
    # acdc and kvasir_seg: genuine attempts against the official upstream
    # source did not yield extractable license text (module docstring);
    # this must remain an honest, precise statement, never a plausible-
    # looking guess.
    assert _NOT_CONFIRMABLE_MARKER in DATASET_METADATA[key].license


def test_cifar10_license_confirms_no_formal_license_exists() -> None:
    # MS6.2 Resolution Pass: independently confirmed via the official page
    # that CIFAR-10 has no formal license at all (citation-only) --
    # correcting the initial pass's imprecise "MIT-style/research" hedge
    # in the opposite direction (not MIT-*like*, simply unlicensed).
    license_text = DATASET_METADATA["cifar10"].license
    assert license_text.startswith("No formal software/data license")


def test_cifar10_1_license_distinguishes_code_license_from_data_terms() -> None:
    # MS6.2 Resolution Pass: the repository's MIT license covers its code,
    # not the actual image/label data (which follows Tiny Images' own
    # terms) -- correcting the initial pass's plain "MIT" label, which
    # conflated the two.
    license_text = DATASET_METADATA["cifar10_1"].license
    assert "MIT" in license_text
    assert "NOT apply to the actual image and label data" in license_text


def test_msd_entries_reconfirm_cc_by_sa_license() -> None:
    for key in ("msd_hippocampus", "msd_pancreas"):
        assert "CC-BY-SA 4.0" in DATASET_METADATA[key].license


def test_cityscapes_license_names_the_specific_agreement() -> None:
    # MS6.2 Resolution Pass: replaced the generic "Research, non-commercial"
    # label with the specific, named Cityscapes License Agreement.
    license_text = DATASET_METADATA["cityscapes"].license
    assert "Cityscapes License Agreement" in license_text
    assert "non-commercial" in license_text.lower()
