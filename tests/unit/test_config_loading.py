"""Durable validation of the bundled dataset registry."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from gasbench.constants import VALID_MEDIA_TYPES
from gasbench.dataset.config import load_benchmark_datasets_from_yaml


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


@pytest.mark.parametrize("layout", ["flat", "grouped"])
def test_custom_yaml_preserves_download_constraints(tmp_path, layout):
    from gasbench.dataset.config import load_datasets_from_yaml

    entry = {
        "name": "custom", "path": "example/repo", "modality": "image",
        "media_type": "synthetic", "source_format": ["jpg", "png"],
        "hf_revision": "pinned-revision", "hf_subfolders": ["subset"],
        "include_paths": ["keep"], "exclude_paths": ["drop"],
        "data_columns": ["source", "edited"], "media_per_archive": 7,
    }
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump({"datasets" if layout == "flat" else "image": [entry]}))
    result = load_datasets_from_yaml(str(path))
    (dataset,) = result["image"]
    assert not result["video"] and not result["audio"]
    for field, value in entry.items():
        assert getattr(dataset, field) == value


@pytest.mark.parametrize("entry", [
    {"path": "example/repo", "modality": "image", "media_type": "real"},
    {"name": "bad", "path": "example/repo", "modality": "image", "media_type": "unknown"},
])
def test_custom_yaml_rejects_invalid_entries_instead_of_silently_dropping_them(tmp_path, entry):
    from gasbench.dataset.config import load_datasets_from_yaml

    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump({"datasets": [entry]}))
    with pytest.raises(ValueError, match="Validation errors"):
        load_datasets_from_yaml(str(path))
