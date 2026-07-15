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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gasbench")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+unknown"
__author__ = "bmbm"

__all__ = [
    "run_image_benchmark",
    "run_video_benchmark",
    "run_benchmark",
    "create_inference_session",
    "print_benchmark_summary",
    "format_benchmark_summary",
    "save_results_to_json",
]

# Benchmark exports resolve lazily (PEP 562) so that the base install —
# which carries only the dataset registry and pyyaml — can import
# gasbench.dataset.config without pulling in torch/onnxruntime. The full
# benchmark stack requires the `gpu` extra: pip install gasbench[gpu]
_LAZY_EXPORTS = {
    "run_image_benchmark": ("gasbench.benchmarks.image_bench", "run_image_benchmark"),
    "run_video_benchmark": ("gasbench.benchmarks.video_bench", "run_video_benchmark"),
    "run_benchmark": ("gasbench.benchmark", "run_benchmark"),
    "print_benchmark_summary": ("gasbench.benchmark", "print_benchmark_summary"),
    "format_benchmark_summary": ("gasbench.benchmark", "format_benchmark_summary"),
    "save_results_to_json": ("gasbench.benchmark", "save_results_to_json"),
    "create_inference_session": ("gasbench.benchmarks.utils.inference", "create_inference_session"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"gasbench.{name} requires the benchmark dependencies, which are "
                f"not part of the base install. Install them with: "
                f"pip install 'gasbench[gpu]' (original error: {e})"
            ) from e
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
