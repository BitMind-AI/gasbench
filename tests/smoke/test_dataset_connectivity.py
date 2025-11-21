"""
Smoke tests for dataset connectivity.

These tests verify that datasets exist and are accessible on HuggingFace.
No actual downloads - just HEAD requests and repo metadata checks.

Mark as slow: pytest -m "not slow" to skip these tests.
"""

import pytest
from huggingface_hub import HfApi, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from src.gasbench.dataset.config import load_benchmark_datasets_from_yaml


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


class TestNewImageDatasetConnectivity:
    """Test that newly added image datasets are accessible."""

    @pytest.mark.parametrize("dataset_name", [
        "pica-100k",
        "text-to-image-2m",
        "nano-banana-150k",
        "artifact",
        "cosyn-400k",
    ])
    def test_new_image_dataset_exists(self, dataset_name):
        """Verify new image dataset exists and is accessible."""
        configs = load_benchmark_datasets_from_yaml()
        dataset = next((d for d in configs["image"] if d.name == dataset_name), None)
        
        assert dataset is not None, f"Dataset {dataset_name} not found in config"
        
        api = HfApi()
        
        try:
            repo_info = api.repo_info(dataset.path, repo_type="dataset")
            assert repo_info is not None, f"Could not get repo info for {dataset.path}"
            
        except GatedRepoError:
            pytest.skip(f"Dataset {dataset.path} is gated - requires authentication")
            
        except RepositoryNotFoundError:
            pytest.fail(f"Dataset {dataset.path} not found on HuggingFace")

    @pytest.mark.parametrize("dataset_name", [
        "pica-100k",
        "text-to-image-2m",
        "nano-banana-150k",
        "artifact",
        "cosyn-400k",
    ])
    def test_new_image_dataset_has_files(self, dataset_name):
        """Verify new image dataset has expected files."""
        configs = load_benchmark_datasets_from_yaml()
        dataset = next((d for d in configs["image"] if d.name == dataset_name), None)
        
        try:
            files = list(list_repo_files(dataset.path, repo_type="dataset"))
            assert len(files) > 0, f"Dataset {dataset.path} has no files"
            
            # Check for expected format files
            if dataset.source_format:
                matching_files = [f for f in files if dataset.source_format in f.lower()]
                assert len(matching_files) > 0, \
                    f"No files with format {dataset.source_format} found in {dataset.path}"
            
        except GatedRepoError:
            pytest.skip(f"Dataset {dataset.path} is gated")
        except RepositoryNotFoundError:
            pytest.fail(f"Dataset {dataset.path} not found")


class TestNewVideoDatasetConnectivity:
    """Test that newly added video datasets are accessible."""

    @pytest.mark.parametrize("dataset_name", [
        "vidprom",
        "moments-in-time",
        "ucf101-fullvideo",
        "vap-data",
    ])
    def test_new_video_dataset_exists(self, dataset_name):
        """Verify new video dataset exists and is accessible."""
        configs = load_benchmark_datasets_from_yaml()
        dataset = next((d for d in configs["video"] if d.name == dataset_name), None)
        
        assert dataset is not None, f"Dataset {dataset_name} not found in config"
        
        api = HfApi()
        
        try:
            repo_info = api.repo_info(dataset.path, repo_type="dataset")
            assert repo_info is not None, f"Could not get repo info for {dataset.path}"
            
        except GatedRepoError:
            pytest.skip(f"Dataset {dataset.path} is gated")
        except RepositoryNotFoundError:
            pytest.fail(f"Dataset {dataset.path} not found")

    @pytest.mark.parametrize("dataset_name", [
        "vidprom",
        "moments-in-time",
        "ucf101-fullvideo",
        "vap-data",
    ])
    def test_new_video_dataset_has_files(self, dataset_name):
        """Verify new video dataset has expected files."""
        configs = load_benchmark_datasets_from_yaml()
        dataset = next((d for d in configs["video"] if d.name == dataset_name), None)
        
        try:
            files = list(list_repo_files(dataset.path, repo_type="dataset"))
            assert len(files) > 0, f"Dataset {dataset.path} has no files"
            
        except GatedRepoError:
            pytest.skip(f"Dataset {dataset.path} is gated")
        except RepositoryNotFoundError:
            pytest.fail(f"Dataset {dataset.path} not found")


class TestCriticalDatasetConnectivity:
    """Test connectivity for critical/commonly used datasets."""

    @pytest.mark.parametrize("dataset_name,modality", [
        ("pica-100k", "image"),  # Multi-column dataset
        ("fashionpedia", "image"),  # Standard image dataset
        ("vap-data", "video"),  # MP4 video dataset
        ("deepfake-audio-dataset", "audio"),  # Audio dataset
    ])
    def test_critical_dataset_accessible(self, dataset_name, modality):
        """Verify critical datasets are accessible."""
        configs = load_benchmark_datasets_from_yaml()
        dataset = next((d for d in configs[modality] if d.name == dataset_name), None)
        
        assert dataset is not None, f"Critical dataset {dataset_name} not found"
        
        api = HfApi()
        
        try:
            repo_info = api.repo_info(dataset.path, repo_type="dataset")
            assert repo_info is not None
            
            # Verify has files
            files = list(list_repo_files(dataset.path, repo_type="dataset"))
            assert len(files) > 0
            
        except GatedRepoError:
            pytest.skip(f"Dataset {dataset.path} is gated")
        except RepositoryNotFoundError:
            pytest.fail(f"Critical dataset {dataset.path} not found!")


class TestGasstationDatasets:
    """Test gasstation datasets (always available)."""

    @pytest.mark.parametrize("modality", ["image", "video"])
    def test_gasstation_dataset_exists(self, modality):
        """Verify gasstation datasets are configured."""
        configs = load_benchmark_datasets_from_yaml()
        
        gasstation_datasets = [
            d for d in configs[modality]
            if "gasstation" in d.name.lower()
        ]
        
        assert len(gasstation_datasets) > 0, \
            f"No gasstation datasets found for {modality}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])

