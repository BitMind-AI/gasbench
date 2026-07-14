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

# The config exports above stay eager: they need nothing beyond pyyaml and
# are what dataset-registry consumers (base install) use. The
# iterator/cache/download exports below resolve lazily (PEP 562) because
# their modules import torch / huggingface_hub / datasets, which only exist
# in the `gpu` extra: pip install gasbench[gpu]
_LAZY_EXPORTS = {
    "DatasetIterator": ("gasbench.dataset.iterator", "DatasetIterator"),
    "CACHE_MAX_SAMPLES": ("gasbench.dataset.iterator", "CACHE_MAX_SAMPLES"),
    "GASSTATION_CACHE_MAX_SAMPLES": ("gasbench.dataset.iterator", "GASSTATION_CACHE_MAX_SAMPLES"),
    "check_dataset_cache": ("gasbench.dataset.cache", "check_dataset_cache"),
    "save_sample_to_cache": ("gasbench.dataset.cache", "save_sample_to_cache"),
    "download_and_extract": ("gasbench.dataset.download", "download_and_extract"),
    "DatasetAccessError": ("gasbench.dataset.download", "DatasetAccessError"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"gasbench.dataset.{name} requires the benchmark dependencies, "
                f"which are not part of the base install. Install them with: "
                f"pip install 'gasbench[gpu]' (original error: {e})"
            ) from e
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
