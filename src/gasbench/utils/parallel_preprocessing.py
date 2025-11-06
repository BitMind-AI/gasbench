import logging
import numpy as np
from typing import Tuple, Optional, Any
from multiprocessing import Pool
from functools import partial

from ..processing.transforms import (
    apply_random_augmentations,
    compress_image_jpeg_pil,
)

logger = logging.getLogger(__name__)


def preprocess_image_sample(
    sample_data: Tuple[np.ndarray, int, int, Any],
    target_size: Tuple[int, int],
    apply_jpeg: bool = True,
) -> Optional[Tuple[np.ndarray, int, int, Any]]:
    """
    Preprocess a single image sample (for use in multiprocessing pool).

    Args:
        sample_data: Tuple of (image_array, true_label_multiclass, sample_index, sample_metadata)
        target_size: Target (H, W) for resize
        apply_jpeg: Whether to apply JPEG compression

    Returns:
        Tuple of (aug_chw, true_label_multiclass, sample_index, sample_metadata) or None if failed
    """
    image_array, true_label_multiclass, sample_index, sample_metadata = sample_data

    try:
        chw = image_array[0]
        hwc = np.transpose(chw, (1, 2, 0))

        # Use sample_index as seed for reproducibility
        aug_hwc, _, _, _ = apply_random_augmentations(
            hwc, target_size, seed=sample_index
        )

        # Apply JPEG compression if requested
        if apply_jpeg:
            aug_hwc = compress_image_jpeg_pil(aug_hwc, quality=75)

        aug_chw = np.transpose(aug_hwc, (2, 0, 1))

        return (aug_chw, true_label_multiclass, sample_index, sample_metadata)

    except Exception as e:
        logger.debug(f"Failed to preprocess image sample {sample_index}: {e}")
        return None


def preprocess_video_sample(
    sample_data: Tuple[np.ndarray, int, int, Any], target_size: Tuple[int, int]
) -> Optional[Tuple[np.ndarray, int, int, Any]]:
    """
    Preprocess a single video sample (for use in multiprocessing pool).

    Args:
        sample_data: Tuple of (video_array, true_label_multiclass, sample_index, sample_metadata)
        target_size: Target (H, W) for resize

    Returns:
        Tuple of (aug_tchw, true_label_multiclass, sample_index, sample_metadata) or None if failed
    """
    video_array, true_label_multiclass, sample_index, sample_metadata = sample_data

    try:
        tchw = video_array[0]
        thwc = np.transpose(tchw, (0, 2, 3, 1))

        # Use sample_index as seed for reproducibility
        aug_thwc, _, _, _ = apply_random_augmentations(
            thwc, target_size, seed=sample_index
        )

        aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))

        return (aug_tchw, true_label_multiclass, sample_index, sample_metadata)

    except Exception as e:
        logger.debug(f"Failed to preprocess video sample {sample_index}: {e}")
        return None


class ParallelPreprocessor:

    def __init__(
        self,
        num_workers: int,
        modality: str,
        target_size: Tuple[int, int],
        apply_jpeg: bool = True,
    ):
        """
        Args:
            num_workers: Number of worker processes
            modality: 'image' or 'video'
            target_size: Target (H, W) for resize
            apply_jpeg: Whether to apply JPEG compression (image only)
        """
        self.num_workers = num_workers
        self.modality = modality
        self.target_size = target_size
        self.apply_jpeg = apply_jpeg
        self.pool = None

    def __enter__(self):
        self.pool = Pool(processes=self.num_workers)
        logger.info(
            f"Started preprocessing pool with {self.num_workers} workers for {self.modality}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def process_batch(self, samples: list) -> list:
        """
        Process a batch of samples in parallel.

        Args:
            samples: List of (array, label, index, metadata) tuples

        Returns:
            List of processed (aug_array, label, index, metadata) tuples (None filtered out)
        """
        if not self.pool:
            raise RuntimeError(
                "Pool not initialized. Use 'with ParallelPreprocessor(...) as preprocessor:'"
            )

        if self.modality == "image":
            preprocess_fn = partial(
                preprocess_image_sample,
                target_size=self.target_size,
                apply_jpeg=self.apply_jpeg,
            )
        else:
            preprocess_fn = partial(
                preprocess_video_sample, target_size=self.target_size
            )

        results = self.pool.map(preprocess_fn, samples)
        return [r for r in results if r is not None]
