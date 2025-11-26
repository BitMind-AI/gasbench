"""Benchmark execution and metrics."""

from .image_bench import run_image_benchmark
from .video_bench import run_video_benchmark
from .utils import (
    Metrics,
    update_generator_stats,
    calculate_per_source_accuracy,
)

__all__ = [
    "run_image_benchmark",
    "run_video_benchmark",
    "Metrics",
    "update_generator_stats",
    "calculate_per_source_accuracy",
]

