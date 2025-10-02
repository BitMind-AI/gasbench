"""Media processing utilities for SN34Bench.

This package contains modules for:
- Media sample processing (images and videos)
- Data transformations and augmentations
- Archive management and caching
- HuggingFace integration
"""

from .media import (
    configure_huggingface_cache,
    process_image_sample,
    process_video_bytes_sample,
)
from .transforms import (
    apply_random_augmentations,
    compress_image_jpeg_pil,
    compress_video_frames_jpeg_torchvision,
)
from .archive import video_archive_manager

__all__ = [
    "configure_huggingface_cache",
    "process_image_sample", 
    "process_video_bytes_sample",
    "apply_random_augmentations",
    "compress_image_jpeg_pil",
    "compress_video_frames_jpeg_torchvision",
    "video_archive_manager",
]
