"""Durable validation of the bundled dataset registry."""

from collections import Counter
from pathlib import Path

import yaml

from src.gasbench.constants import VALID_MEDIA_TYPES
from src.gasbench.dataset.config import load_benchmark_datasets_from_yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/gasbench/dataset/configs"


def test_registry_entries_are_well_formed():
    configs = load_benchmark_datasets_from_yaml()

    assert set(configs) == {"image", "video", "audio"}
    for modality, datasets in configs.items():
        assert datasets, f"no {modality} datasets loaded"
        for dataset in datasets:
            assert dataset.name, f"{modality} dataset missing name"
            assert dataset.path, f"{modality}/{dataset.name} missing path"
            assert dataset.modality == modality, (
                f"{modality}/{dataset.name} declares modality {dataset.modality}"
            )
            assert dataset.media_type in VALID_MEDIA_TYPES[modality], (
                f"{modality}/{dataset.name} has invalid media type {dataset.media_type}"
            )
            if dataset.data_columns is not None:
                assert dataset.data_columns
                assert all(
                    isinstance(column, str) and column
                    for column in dataset.data_columns
                )
                assert len(dataset.data_columns) == len(set(dataset.data_columns))


def test_dataset_names_are_unique_within_each_modality():
    configs = load_benchmark_datasets_from_yaml()

    for modality, datasets in configs.items():
        counts = Counter(dataset.name for dataset in datasets)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        assert not duplicates, f"duplicate {modality} dataset names: {duplicates}"


def test_legacy_datasets_are_not_in_the_active_registry():
    configs = load_benchmark_datasets_from_yaml()
    active = {
        (modality, dataset.name)
        for modality, datasets in configs.items()
        for dataset in datasets
    }

    legacy = set()
    for modality, filename in (
        ("image", "legacy_images.yaml"),
        ("video", "legacy_videos.yaml"),
    ):
        document = yaml.safe_load((CONFIG_DIR / filename).read_text())
        legacy.update((modality, dataset["name"]) for dataset in document["datasets"])

    overlap = sorted(active & legacy)
    assert not overlap, f"legacy datasets still active: {overlap}"


def test_direct_huggingface_sources_are_pinned_and_scoped():
    configs = load_benchmark_datasets_from_yaml()
    datasets = {
        dataset.name: dataset
        for modality in ("image", "video")
        for dataset in configs[modality]
    }
    expected = {
        "synwts": (
            "mlcglab/synwts",
            "3c57d64f8603441b43f69dca9a123999776bc014",
            ["data/videos"],
        ),
        "social-media-deepfakes-real": (
            "acroitoru/social_media_deepfakes",
            "14bf14c255e222ebd2f0adadb2de83784695e72a",
            ["social_media/real"],
        ),
        "social-media-deepfakes-fake": (
            "acroitoru/social_media_deepfakes",
            "14bf14c255e222ebd2f0adadb2de83784695e72a",
            ["social_media/fake"],
        ),
        "vigilvid-research-real": (
            "farouk04/vigilvid-research",
            "f4ec1afcc9599861b41cffb7768ad16340c41612",
            ["videos/test/real"],
        ),
        "vigilvid-research-fake": (
            "farouk04/vigilvid-research",
            "f4ec1afcc9599861b41cffb7768ad16340c41612",
            ["videos/test/fake"],
        ),
    }

    for name, (path, revision, subfolders) in expected.items():
        dataset = datasets[name]
        assert (dataset.path, dataset.hf_revision, dataset.hf_subfolders) == (
            path,
            revision,
            subfolders,
        )
