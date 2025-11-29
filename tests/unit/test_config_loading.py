"""
Unit tests for dataset configuration loading.

These tests ensure YAML configs are valid and properly structured.
"""

import pytest
from src.gasbench.dataset.config import (
    load_benchmark_datasets_from_yaml,
    BenchmarkDatasetConfig,
)


class TestConfigLoading:
    """Test dataset configuration loading."""

    def test_all_configs_load_successfully(self):
        """Ensure all YAML configs parse without errors."""
        configs = load_benchmark_datasets_from_yaml()
        
        assert "image" in configs
        assert "video" in configs
        assert "audio" in configs
        
        assert len(configs["image"]) > 0, "No image datasets loaded"
        assert len(configs["video"]) > 0, "No video datasets loaded"
        assert len(configs["audio"]) > 0, "No audio datasets loaded"

    def test_expected_dataset_counts(self):
        """Verify expected number of datasets per modality."""
        configs = load_benchmark_datasets_from_yaml()
        
        # Update these counts when adding new datasets
        assert len(configs["image"]) == 42, f"Expected 42 image datasets, got {len(configs['image'])}"
        assert len(configs["video"]) == 32, f"Expected 32 video datasets, got {len(configs['video'])}"
        assert len(configs["audio"]) == 14, f"Expected 14 audio datasets, got {len(configs['audio'])}"

    def test_all_datasets_have_required_fields(self):
        """Ensure all datasets have required fields."""
        configs = load_benchmark_datasets_from_yaml()
        
        for modality, datasets in configs.items():
            for dataset in datasets:
                assert dataset.name, f"{modality} dataset missing name"
                assert dataset.path, f"{modality}/{dataset.name} missing path"
                assert dataset.modality, f"{modality}/{dataset.name} missing modality"
                assert dataset.media_type in ["real", "synthetic", "semisynthetic"], \
                    f"{modality}/{dataset.name} has invalid media_type: {dataset.media_type}"

    def test_no_duplicate_dataset_names(self):
        """Ensure no duplicate dataset names within each modality."""
        configs = load_benchmark_datasets_from_yaml()
        
        for modality, datasets in configs.items():
            names = [d.name for d in datasets]
            duplicates = [name for name in names if names.count(name) > 1]
            assert not duplicates, f"{modality} has duplicate dataset names: {set(duplicates)}"

    def test_dataset_paths_are_valid_format(self):
        """Ensure dataset paths follow expected format."""
        configs = load_benchmark_datasets_from_yaml()
        
        for modality, datasets in configs.items():
            for dataset in datasets:
                # Paths should be either "username/repo" or "gasstation/..."
                assert "/" in dataset.path, \
                    f"{modality}/{dataset.name} path missing '/': {dataset.path}"


class TestImageDatasets:
    """Test image-specific dataset configurations."""

    def test_pica_100k_exists(self):
        """Verify PICA-100K dataset is configured."""
        configs = load_benchmark_datasets_from_yaml()
        pica = next((d for d in configs["image"] if d.name == "pica-100k"), None)
        
        assert pica is not None, "PICA-100K dataset not found"

    def test_pica_100k_has_dual_columns(self):
        """Verify PICA-100K has image_columns configured for dual-column support."""
        configs = load_benchmark_datasets_from_yaml()
        pica = next((d for d in configs["image"] if d.name == "pica-100k"), None)
        
        assert pica.image_columns is not None, "PICA-100K missing image_columns"
        assert pica.image_columns == ["src_img", "tgt_img"], \
            f"PICA-100K image_columns incorrect: {pica.image_columns}"

    def test_pica_100k_has_correct_format(self):
        """Verify PICA-100K configuration is correct."""
        configs = load_benchmark_datasets_from_yaml()
        pica = next((d for d in configs["image"] if d.name == "pica-100k"), None)
        
        assert pica.path == "Andrew613/PICA-100K"
        assert pica.modality == "image"
        assert pica.media_type == "synthetic"
        assert pica.source_format == "parquet"

    def test_new_image_datasets_exist(self):
        """Verify all newly added image datasets exist."""
        configs = load_benchmark_datasets_from_yaml()
        image_names = [d.name for d in configs["image"]]
        
        new_datasets = [
            "pica-100k",
            "text-to-image-2m",
            "nano-banana-150k",
            "artifact",
            "cosyn-400k",
        ]
        
        for name in new_datasets:
            assert name in image_names, f"New image dataset '{name}' not found"

    def test_only_pica_has_image_columns(self):
        """Verify only PICA-100K uses image_columns (for now)."""
        configs = load_benchmark_datasets_from_yaml()
        
        datasets_with_image_cols = [
            d.name for d in configs["image"] 
            if getattr(d, "image_columns", None) is not None
        ]
        
        # Currently only PICA-100K should have this
        assert datasets_with_image_cols == ["pica-100k"], \
            f"Unexpected datasets with image_columns: {datasets_with_image_cols}"


class TestVideoDatasets:
    """Test video-specific dataset configurations."""

    def test_new_video_datasets_exist(self):
        """Verify all newly added video datasets exist."""
        configs = load_benchmark_datasets_from_yaml()
        video_names = [d.name for d in configs["video"]]
        
        new_datasets = [
            "vidprom",
            "moments-in-time",
            "ucf101-fullvideo",
            "vap-data",
        ]
        
        for name in new_datasets:
            assert name in video_names, f"New video dataset '{name}' not found"

    def test_video_source_formats_are_valid(self):
        """Verify video datasets have valid source formats."""
        configs = load_benchmark_datasets_from_yaml()
        valid_formats = ["mp4", "avi", "tar", "zip", "tar.gz", "parquet", ""]
        
        for dataset in configs["video"]:
            assert dataset.source_format in valid_formats, \
                f"Video dataset {dataset.name} has invalid format: {dataset.source_format}"


class TestAudioDatasets:
    """Test audio-specific dataset configurations."""

    def test_audio_datasets_exist(self):
        """Verify audio datasets are configured."""
        configs = load_benchmark_datasets_from_yaml()
        
        assert len(configs["audio"]) >= 14, "Expected at least 14 audio datasets"

    def test_audio_source_formats_are_valid(self):
        """Verify audio datasets have valid source formats."""
        configs = load_benchmark_datasets_from_yaml()
        valid_formats = ["wav", "mp3", "tar", "tar.gz", "zip", "parquet", ""]
        
        for dataset in configs["audio"]:
            assert dataset.source_format in valid_formats, \
                f"Audio dataset {dataset.name} has invalid format: {dataset.source_format}"


class TestMediaTypeDistribution:
    """Test distribution of real vs synthetic datasets."""

    def test_image_has_balanced_types(self):
        """Verify image datasets have both real and synthetic."""
        configs = load_benchmark_datasets_from_yaml()
        
        real_count = sum(1 for d in configs["image"] if d.media_type == "real")
        synthetic_count = sum(1 for d in configs["image"] if d.media_type == "synthetic")
        
        assert real_count > 0, "No real image datasets"
        assert synthetic_count > 0, "No synthetic image datasets"
        assert real_count >= 10, f"Expected at least 10 real image datasets, got {real_count}"
        assert synthetic_count >= 10, f"Expected at least 10 synthetic image datasets, got {synthetic_count}"

    def test_video_has_balanced_types(self):
        """Verify video datasets have both real and synthetic."""
        configs = load_benchmark_datasets_from_yaml()
        
        real_count = sum(1 for d in configs["video"] if d.media_type == "real")
        synthetic_count = sum(1 for d in configs["video"] if d.media_type == "synthetic")
        
        assert real_count > 0, "No real video datasets"
        assert synthetic_count > 0, "No synthetic video datasets"

    def test_audio_has_balanced_types(self):
        """Verify audio datasets have both real and synthetic."""
        configs = load_benchmark_datasets_from_yaml()
        
        real_count = sum(1 for d in configs["audio"] if d.media_type == "real")
        synthetic_count = sum(1 for d in configs["audio"] if d.media_type == "synthetic")
        
        assert real_count > 0, "No real audio datasets"
        assert synthetic_count > 0, "No synthetic audio datasets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

