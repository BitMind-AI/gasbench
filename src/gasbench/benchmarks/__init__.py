"""Lazy exports, matching gasbench: recorder recovery needs no inference backend."""

from importlib import import_module

_LAZY_EXPORTS = {
    "run_image_benchmark": (".image_bench", "run_image_benchmark"),
    "run_video_benchmark": (".video_bench", "run_video_benchmark"),
    "Metrics": (".utils.metrics", "Metrics"),
    "update_generator_stats": (".utils.metrics", "update_generator_stats"),
    "calculate_per_source_accuracy": (
        ".utils.metrics",
        "calculate_per_source_accuracy",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr)
