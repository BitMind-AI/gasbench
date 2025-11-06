"""Automatic resource optimization for preprocessing and batching."""

import os
import logging
import numpy as np
import psutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_optimal_preprocessing_workers() -> int:
    """
    Calculate optimal number of preprocessing workers based on available CPU cores.

    Returns:
        int: Optimal number of worker processes
    """
    try:
        # Get physical CPU cores (not logical/hyperthreaded)
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores is None:
            physical_cores = os.cpu_count() or 4

        # Leave 1-2 cores for main process and system
        # Use 75% of available cores for preprocessing
        optimal_workers = max(1, int(physical_cores * 0.75))

        logger.info(
            f"Detected {physical_cores} physical CPU cores, using {optimal_workers} preprocessing workers"
        )
        return optimal_workers

    except Exception as e:
        logger.warning(f"Failed to detect CPU cores, defaulting to 4 workers: {e}")
        return 4


def calculate_optimal_batch_size(
    session,
    input_specs: dict,
    modality: str,
    target_size: Tuple[int, int],
    min_batch_size: int = 1,
    max_batch_size: int = 512,
    vram_safety_margin: float = 0.85,
) -> int:
    """
    Calculate optimal batch size based on GPU memory profiling.

    Args:
        session: ONNX Runtime session
        input_specs: Model input specifications
        modality: 'image' or 'video'
        target_size: (H, W) tuple for input size
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        vram_safety_margin: Use only this fraction of available VRAM (default 85%)

    Returns:
        int: Optimal batch size
    """
    try:
        import pynvml

        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Get total and available GPU memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_mb = mem_info.total / (1024**2)
        free_vram_mb = mem_info.free / (1024**2)
        used_vram_mb = mem_info.used / (1024**2)

        logger.info(
            f"GPU Memory: {used_vram_mb:.0f}MB used, {free_vram_mb:.0f}MB free, {total_vram_mb:.0f}MB total"
        )

        # Apply safety margin
        available_vram_mb = free_vram_mb * vram_safety_margin

        # Profile with increasing batch sizes
        input_name = list(input_specs.keys())[0]
        input_shape = input_specs[input_name]["shape"]

        # Determine frames for video models
        if modality == "video":
            # Assume 16 frames as default for profiling
            num_frames = 16
            test_shape = [1, num_frames, 3, target_size[0], target_size[1]]
        else:
            test_shape = [1, 3, target_size[0], target_size[1]]

        # Test with batch_size=1 first
        logger.info("Profiling GPU memory with batch_size=1...")
        test_input = np.random.randint(0, 255, test_shape, dtype=np.uint8).astype(
            np.float32
        )

        # Warmup
        _ = session.run(None, {input_name: test_input})

        # Measure baseline memory
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)

        # Run inference
        _ = session.run(None, {input_name: test_input})

        # Measure after inference
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        per_sample_mb = mem_after - mem_before

        if per_sample_mb <= 0:
            # Fallback if we can't measure accurately
            logger.warning("Could not measure per-sample GPU memory, using heuristic")
            per_sample_mb = 10  # Conservative estimate

        # Calculate optimal batch size
        optimal_batch = int(available_vram_mb / per_sample_mb)
        optimal_batch = max(min_batch_size, min(optimal_batch, max_batch_size))

        logger.info(
            f"Calculated optimal batch size: {optimal_batch} "
            f"(~{per_sample_mb:.1f}MB per sample, {available_vram_mb:.0f}MB available)"
        )

        # Cleanup
        pynvml.nvmlShutdown()

        return optimal_batch

    except ImportError:
        logger.warning("pynvml not available, using default batch sizes")
        return 32 if modality == "image" else 4

    except Exception as e:
        logger.warning(f"Failed to calculate optimal batch size: {e}, using defaults")
        return 32 if modality == "image" else 4
