"""Global configuration constants for GASBench."""

# Default target size for models with dynamic axes
DEFAULT_TARGET_SIZE = (224, 224)

# Default batch sizes for inference
DEFAULT_IMAGE_BATCH_SIZE = 32
DEFAULT_VIDEO_BATCH_SIZE = 4  # Smaller because videos have multiple frames per sample

