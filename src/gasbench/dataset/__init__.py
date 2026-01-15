"""Dataset handling utilities for SN34Bench.

This package contains modules for:
- Dataset configuration and discovery
- Dataset iteration and sampling  
- Dataset caching and persistence
- Dataset downloading and extraction
"""

from .config import (
    BenchmarkDatasetConfig,
    BENCHMARK_TOTAL_OVERRIDES,
    DOWNLOAD_SIZE_OVERRIDES,
    REGULAR_DATASET_MIN_SAMPLES,
    REGULAR_DATASET_MAX_SAMPLES,
    GASSTATION_DATASET_MIN_SAMPLES,
    GASSTATION_DATASET_MAX_SAMPLES,
    GASSTATION_WEIGHT_MULTIPLIER,
    get_benchmark_size,
    discover_benchmark_datasets,
)
from .iterator import DatasetIterator, CACHE_MAX_SAMPLES, GASSTATION_CACHE_MAX_SAMPLES
from .cache import check_dataset_cache, save_sample_to_cache
from .download import download_and_extract, DatasetAccessError

__all__ = [
    "BenchmarkDatasetConfig",
    "BENCHMARK_TOTAL_OVERRIDES",
    "DOWNLOAD_SIZE_OVERRIDES",
    "REGULAR_DATASET_MIN_SAMPLES",
    "REGULAR_DATASET_MAX_SAMPLES",
    "GASSTATION_DATASET_MIN_SAMPLES",
    "GASSTATION_DATASET_MAX_SAMPLES",
    "GASSTATION_WEIGHT_MULTIPLIER",
    "get_benchmark_size",
    "discover_benchmark_datasets",
    "DatasetIterator",
    "CACHE_MAX_SAMPLES",
    "GASSTATION_CACHE_MAX_SAMPLES",
    "check_dataset_cache",
    "save_sample_to_cache", 
    "download_and_extract",
    "DatasetAccessError",
]
