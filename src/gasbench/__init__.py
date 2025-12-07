"""GASBench - AIGC Detection

Main components:
- benchmark_execution: Core benchmark execution logic
- benchmark: High-level driver for running benchmarks
- utils: Utilities for data processing, configuration, and logging

Example usage:
    from gasbench import run_benchmark
    
    results = await run_benchmark(
        model_path="path/to/model.onnx",
        modality="image",
        mode="debug"
    )
"""

from .benchmarks.image_bench import run_image_benchmark
from .benchmarks.video_bench import run_video_benchmark
from .benchmark import run_benchmark, print_benchmark_summary, save_results_to_json
from .benchmarks.utils.inference import create_inference_session

__version__ = "0.4.0"
__author__ = "bmbm"

__all__ = [
    "run_image_benchmark",
    "run_video_benchmark", 
    "run_benchmark",
    "create_inference_session",
    "print_benchmark_summary",
    "save_results_to_json",
]
