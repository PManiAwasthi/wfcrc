"""Dataset Metadata — canonical machine-readable source for the frozen
Phase-A dataset suite's version/license/provenance (MS6.2, §3.6; MS6.2
Resolution Pass Task 5).

Populates :data:`DATASET_METADATA` for exactly the eight Phase-A dataset
artifacts named in the Dataset Selection Audit's Acquisition Plan
(§5) and Experiment Environment Audit §6's ``configs/dataset_*.yaml``
naming convention: ``cityscapes``, ``cityscapes_c``, ``acdc``,
``msd_hippocampus``, ``msd_pancreas``, ``cifar10``, ``cifar10_1``,
``kvasir_seg``. These are exactly the keys :data:`wfcrc.datasets.registry.DATASETS`
will use once MS6.3 registers concrete loaders (MS6 Architecture
Specification §3.1's stated consistency requirement).

**MS6.2 Resolution Pass — verified against authoritative upstream sources.**
The initial MS6.2 pass populated every field by literal transcription of
the Dataset Selection Audit §5 table alone, with ``version`` left
uniformly unresolved and ``cifar10``'s ``license`` flagged (the audit's
own "MIT-style/research" hedge). This pass independently checked each
dataset's *official* source (not just the vault's summary) and records
the outcome precisely:

- **Resolved with a real provenance identifier (no formal software
  version exists for any of these eight datasets — none is
  semantically-versioned — so, per the resolution policy, each
  ``version`` field now holds the precise publication/challenge
  identifier that upstream actually uses):** ``cityscapes``,
  ``msd_hippocampus``, ``msd_pancreas``, ``cifar10``, ``cifar10_1``.
- **License corrected from a vague/hedged label to a precise,
  source-confirmed statement:** ``cityscapes`` (a specific named,
  non-OSI "Cityscapes License Agreement", not generic "Research,
  non-commercial"); ``cifar10`` (confirmed: **no formal license at
  all**, citation-only — the prior "MIT-style" hedge was itself
  imprecise in the opposite direction); ``cifar10_1`` (confirmed: the
  repository's MIT license covers its *code*, explicitly **not** the
  image/label data, which follows the original Tiny Images dataset's
  own terms); ``msd_hippocampus``/``msd_pancreas`` (CC-BY-SA 4.0
  independently reconfirmed via the AWS Open Data registry).
- **Still not independently confirmable, honestly reported rather than
  guessed:** ``acdc``'s and ``kvasir_seg``'s exact license text. Both
  official sites (``acdc.vision.ee.ethz.ch``,
  ``datasets.simula.no/kvasir-seg``) returned either a JavaScript-only
  page with no extractable license text, or a TLS certificate error /
  stub page, across multiple fetch attempts in this session (recorded
  per entry below, including exactly what *was* independently
  confirmed — dataset identity, authors, associated paper, hosting
  institution). Per the resolution policy ("record accurately rather
  than guess"), these two ``license`` fields state precisely what was
  and was not verifiable, not a plausible-looking placeholder.

Every field is either a confirmed fact (with its source cited inline)
or an explicit, precise statement of what remains unconfirmed and why —
never an invented value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DATASET_METADATA", "DatasetMetadata"]

#: acdc's license: what was independently confirmed vs. not (module
#: docstring). No formal license text could be extracted; recorded
#: precisely rather than guessed.
_ACDC_LICENSE = (
    "NOT independently confirmable via automated fetch (MS6.2 Resolution "
    "Pass, multiple attempts): the official site (acdc.vision.ee.ethz.ch) "
    "is a JavaScript-rendered single-page app with no license text "
    "extractable by automated tooling; the associated paper (Sakaridis, "
    "Dai & Van Gool, 'ACDC: The Adverse Conditions Dataset with "
    "Correspondences', ICCV 2021, arXiv:2104.13395) states only that the "
    "dataset is 'publicly available', with no license section. Likely "
    "similar non-commercial/research-only terms to Cityscapes (shared "
    "driving-scene/label format, ETH Zurich-affiliated), but this is an "
    "inference, not a confirmed fact. Manual confirmation via the official "
    "site required before MS6.3 downloads this dataset."
)

#: kvasir_seg's license: what was independently confirmed vs. not (module
#: docstring).
_KVASIR_SEG_LICENSE = (
    "NOT independently confirmable via automated fetch (MS6.2 Resolution "
    "Pass, multiple attempts): datasets.simula.no/kvasir-seg returned a "
    "TLS certificate verification error, and other Simula/mirror pages "
    "returned only stub content with no license text. Independently "
    "confirmed instead: dataset identity, authors (developed by Simula "
    "Research Laboratory and the Cancer Registry of Norway), and the "
    "associated paper (Jha et al., 'Kvasir-SEG: A Segmented Polyp "
    "Dataset', MMM 2020, arXiv:1911.07069). A related-but-distinct Simula "
    "dataset (HyperKvasir) is independently confirmed CC BY 4.0 "
    "elsewhere; this must NOT be assumed to transfer to Kvasir-SEG "
    "without direct confirmation. Manual verification via the official "
    "site required before MS6.3 downloads this dataset."
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Static provenance record for one Phase-A dataset artifact.

    Attributes:
        name: The dataset's registry key (matches a future
            :data:`wfcrc.datasets.registry.DATASETS` entry).
        version: A precise provenance identifier — none of these eight
            datasets is formally software-versioned, so this holds the
            publication/challenge/release identifier upstream actually
            uses (e.g. a paper citation, challenge year, or named
            variant), per the MS6.2 Resolution Pass's explicit policy of
            recording the absence of a formal version accurately rather
            than inventing a semantic-version-looking string.
        license: The dataset's license/terms, independently verified
            against the official upstream source where that source was
            reachable; an explicit "not independently confirmable"
            statement (never a guess) where it was not — see the module
            docstring.
        source_url: The official acquisition source.
        extra: Additional attested, non-required provenance detail
            (storage estimate, preprocessing note, repo cache directory,
            registration requirement), from the Dataset Selection Audit
            §5 table.
    """

    name: str
    version: str
    license: str
    source_url: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this metadata record as a plain, JSON-serializable dict.

        Returns:
            A dict with keys ``name``/``version``/``license``/``source_url``/
            ``extra`` (``extra`` copied to a plain ``dict``).
        """
        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "source_url": self.source_url,
            "extra": dict(self.extra),
        }


#: Registry mapping a dataset name to its :class:`DatasetMetadata`, keyed
#: identically to the future :data:`wfcrc.datasets.registry.DATASETS`.
#: See the module docstring for the MS6.2 Resolution Pass verification
#: performed against each dataset's official upstream source.
DATASET_METADATA: dict[str, DatasetMetadata] = {
    "cityscapes": DatasetMetadata(
        name="cityscapes",
        version=(
            "Original release: Cordts et al., 'The Cityscapes Dataset for "
            "Semantic Urban Scene Understanding', CVPR 2016. No formal "
            "version/release number is published for the leftImg8bit+gtFine "
            "split used here — distinct from the later 'Cityscapes 3D' "
            "dataset/benchmark extension (2020-08-30/2020-10-17), which "
            "Phase-A does not use."
        ),
        license=(
            "Cityscapes License Agreement (a specific, non-OSI, custom "
            "license — not a generic 'research' label): academic "
            "research/teaching/personal use only; no redistribution of the "
            "dataset or derivatives; no commercial use; citation required; "
            "provided AS IS; all rights not expressly granted reserved by "
            "Daimler AG, MPI Informatics, and TU Darmstadt. Confirmed via "
            "https://www.cityscapes-dataset.com/license/ (MS6.2 Resolution "
            "Pass), correcting the initial pass's generic 'Research, "
            "non-commercial' label."
        ),
        source_url="cityscapes-dataset.com",
        extra={
            "storage_estimate": "~11 GB (leftImg8bit+gtFine)",
            "preprocessing": "Standard res/normalize; cache base-model softmax on val",
            "repo_cache_dir": "data/cityscapes/",
            "registration_required": "Yes — account approval (start first; can take a day)",
        },
    ),
    "cityscapes_c": DatasetMetadata(
        name="cityscapes_c",
        version=(
            "Derived, on-the-fly, from the Cityscapes val split (see "
            "'cityscapes' entry for its own provenance) via the frozen Q2 "
            "corruption protocol (wfcrc.datasets.corruptions) — this "
            "artifact has no independent publication or release identifier "
            "of its own."
        ),
        license=(
            "Follows Cityscapes terms (derived dataset; no independent "
            "license) — see the 'cityscapes' entry's now-confirmed "
            "Cityscapes License Agreement, which this inherits unchanged."
        ),
        source_url=(
            "Generated (imagecorruptions) or public benchmark mirror — derived "
            "from the Cityscapes val split; no independent canonical source URL "
            "(Dataset Selection Audit §5)"
        ),
        extra={
            "storage_estimate": "~0 (on-the-fly) to ~40 GB (if cached)",
            "preprocessing": "Apply 15 corruptions x severities 1-5 to Cityscapes val",
            "repo_cache_dir": "data/cityscapes_c/",
            "registration_required": "No (derived) — inherits Cityscapes terms",
        },
    ),
    "acdc": DatasetMetadata(
        name="acdc",
        version=(
            "Original release: Sakaridis, Dai & Van Gool, 'ACDC: The "
            "Adverse Conditions Dataset with Correspondences for Semantic "
            "Driving Scene Understanding', ICCV 2021, arXiv:2104.13395. No "
            "further version/release number found."
        ),
        license=_ACDC_LICENSE,
        source_url="acdc.vision.ee.ethz.ch",
        extra={
            "storage_estimate": "~16 GB",
            "preprocessing": "Cityscapes-compatible 19 classes; reuse Cityscapes model",
            "repo_cache_dir": "data/acdc/",
            "registration_required": "Yes — account (start early)",
            "name_collision_warning": "This is the driving Adverse-Conditions dataset, "
            "NOT the cardiac-MRI 'ACDC' dataset (Dataset Selection Audit §7, risk R2).",
        },
    ),
    "msd_hippocampus": DatasetMetadata(
        name="msd_hippocampus",
        version=(
            "Antonelli et al., arXiv:1902.09063 (2019); later published as "
            "'The Medical Segmentation Decathlon', Nature Communications "
            "13, 4128 (2022). Associated with the Medical Segmentation "
            "Decathlon challenge (MICCAI 2018). Task04_Hippocampus "
            "specifically has no separate version number."
        ),
        license=(
            "CC-BY-SA 4.0 International — independently reconfirmed via the "
            "AWS Open Data registry (registry.opendata.aws/msd/), MS6.2 "
            "Resolution Pass; consistent with the initial pass's record "
            "(medicaldecathlon.com itself returned a TLS certificate "
            "mismatch when fetched directly in this session — see "
            "source_url note)."
        ),
        source_url=(
            "medicaldecathlon.com (GDrive/AWS) — note: direct fetch of "
            "medicaldecathlon.com in this session returned a TLS "
            "certificate hostname mismatch (served cert names only "
            "*.storage.googleapis.com and related GCS hosts); license was "
            "instead confirmed via the AWS Open Data registry mirror. This "
            "may reflect a hosting/CDN configuration issue on the official "
            "site as of this verification date, not a change in the "
            "dataset's terms."
        ),
        extra={
            "storage_estimate": "~30 MB",
            "preprocessing": "Resample/normalize (nnU-Net); cache scores",
            "repo_cache_dir": "data/msd/Task04_Hippocampus/",
            "registration_required": "No (open)",
        },
    ),
    "msd_pancreas": DatasetMetadata(
        name="msd_pancreas",
        version=(
            "Antonelli et al., arXiv:1902.09063 (2019); later published as "
            "'The Medical Segmentation Decathlon', Nature Communications "
            "13, 4128 (2022). Associated with the Medical Segmentation "
            "Decathlon challenge (MICCAI 2018). Task07_Pancreas "
            "specifically has no separate version number."
        ),
        license=(
            "CC-BY-SA 4.0 International — independently reconfirmed via the "
            "AWS Open Data registry (registry.opendata.aws/msd/), MS6.2 "
            "Resolution Pass; see 'msd_hippocampus' entry for the "
            "medicaldecathlon.com TLS note (identical for this task)."
        ),
        source_url="medicaldecathlon.com — see 'msd_hippocampus' entry's TLS note",
        extra={
            "storage_estimate": "~12 GB",
            "preprocessing": "Resample/normalize (nnU-Net); cache scores",
            "repo_cache_dir": "data/msd/Task07_Pancreas/",
            "registration_required": "No (open)",
        },
    ),
    "cifar10": DatasetMetadata(
        name="cifar10",
        version=(
            "Krizhevsky, 'Learning Multiple Layers of Features from Tiny "
            "Images', tech report, 2009. No separate dataset version/"
            "release number; created by Alex Krizhevsky, Vinod Nair, and "
            "Geoffrey Hinton."
        ),
        license=(
            "No formal software/data license. Confirmed via direct fetch "
            "of the official page (MS6.2 Resolution Pass; originally "
            "cs.toronto.edu/~kriz/cifar.html, which now 301-redirects to "
            "cave.cs.toronto.edu/kriz/cifar.html): the page states only a "
            "citation requirement ('If you're going to use this dataset, "
            "please cite the tech report'), no license text of any kind. "
            "This replaces the initial pass's 'MIT-style/research' hedge "
            "with a precise, source-confirmed statement: citation-required, "
            "no named license — per the resolution policy's explicit "
            "instruction not to label citation-only terms as license-like."
        ),
        source_url=(
            "cave.cs.toronto.edu/kriz/cifar.html (the "
            "cs.toronto.edu/~kriz/cifar.html URL the Dataset Selection "
            "Audit recorded now 301-redirects here; confirmed MS6.2 "
            "Resolution Pass)"
        ),
        extra={
            "storage_estimate": "~163 MB",
            "preprocessing": "Standard; cache logits",
            "repo_cache_dir": "data/cifar10/",
            "registration_required": "No",
        },
    ),
    "cifar10_1": DatasetMetadata(
        name="cifar10_1",
        version=(
            "Recht, Roelofs, Schmidt & Shankar, 'Do CIFAR-10 Classifiers "
            "Generalize to CIFAR-10.1?', 2018. Two data variants exist, "
            "confirmed via direct fetch of the repository (MS6.2 "
            "Resolution Pass): v4 (2,021 images, the original tested "
            "version) and v6 (2,000 images, class-balanced, recommended "
            "for future use)."
        ),
        license=(
            "The GitHub repository's code is MIT-licensed, but — per the "
            "repository's own README, confirmed via direct fetch, MS6.2 "
            "Resolution Pass — 'The LICENSE file does NOT apply to the "
            "actual image and label data', which derives from the (later "
            "withdrawn) Tiny Images dataset and follows that dataset's own "
            "terms, not a clean MIT grant on the data itself. This corrects "
            "the initial pass's plain 'MIT' label, which conflated the "
            "repository's code license with the data's actual terms."
        ),
        source_url="github.com/modestyachts/CIFAR-10.1",
        extra={
            "storage_estimate": "~30 MB",
            "preprocessing": "Match CIFAR-10 preprocessing",
            "repo_cache_dir": "data/cifar10_1/",
            "registration_required": "No",
            "recommended_variant": "v6 (class-balanced, 2,000 images)",
        },
    ),
    "kvasir_seg": DatasetMetadata(
        name="kvasir_seg",
        version=(
            "Jha et al., 'Kvasir-SEG: A Segmented Polyp Dataset', MMM 2020, "
            "arXiv:1911.07069 (2019). No further version/release number "
            "found."
        ),
        license=_KVASIR_SEG_LICENSE,
        source_url="datasets.simula.no/kvasir-seg",
        extra={
            "storage_estimate": "~46 MB",
            "preprocessing": "Resize/normalize; cache masks",
            "repo_cache_dir": "data/kvasir_seg/",
            "registration_required": "No (direct download)",
            "developed_by": "Simula Research Laboratory and the Cancer Registry of Norway",
        },
    ),
}
