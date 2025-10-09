"""Dataset handling utilities for SN34Bench.

This package contains modules for:
- Dataset configuration and discovery
- Dataset iteration and sampling  
- Dataset caching and persistence
- Dataset downloading and extraction
"""

from .config import (
    BenchmarkDatasetConfig,
    IMAGE_BENCHMARK_SIZE,
    VIDEO_BENCHMARK_SIZE,
    discover_benchmark_image_datasets,
    discover_benchmark_video_datasets,
)
from .iterator import DatasetIterator
from .cache import check_dataset_cache, save_sample_to_cache
from .download import download_and_extract

__all__ = [
    "BenchmarkDatasetConfig",
    "IMAGE_BENCHMARK_SIZE", 
    "VIDEO_BENCHMARK_SIZE",
    "discover_benchmark_image_datasets",
    "discover_benchmark_video_datasets",
    "DatasetIterator",
    "check_dataset_cache",
    "save_sample_to_cache", 
    "download_and_extract",
]
