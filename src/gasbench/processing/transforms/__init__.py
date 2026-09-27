"""Public transform API, organized by shared spatial and modality-specific operations."""

from .audio import (
    apply_audio_robustness_augmentations as apply_audio_robustness_augmentations,
)
from .distortions import (
    ApplyDeeperForensicsDistortion as ApplyDeeperForensicsDistortion,
)
from .distortions import block_wise as block_wise
from .distortions import color_contrast as color_contrast
from .distortions import color_saturation as color_saturation
from .distortions import gaussian_blur as gaussian_blur
from .distortions import gaussian_noise_color as gaussian_noise_color
from .distortions import get_distortion_function as get_distortion_function
from .distortions import get_distortion_parameter as get_distortion_parameter
from .distortions import jpeg_compression as jpeg_compression
from .distortions import rgb2ycbcr as rgb2ycbcr
from .distortions import ycbcr2rgb as ycbcr2rgb
from .image import apply_robustness_augmentations as apply_robustness_augmentations
from .image import compress_image_jpeg_pil as compress_image_jpeg_pil
from .image import compress_image_webp_pil as compress_image_webp_pil
from .spatial import ComposeWithParams as ComposeWithParams
from .spatial import RandomCropWithParams as RandomCropWithParams
from .spatial import RandomHorizontalFlipWithParams as RandomHorizontalFlipWithParams
from .spatial import RandomVerticalFlipWithParams as RandomVerticalFlipWithParams
from .spatial import ResizeShortestEdge as ResizeShortestEdge
from .spatial import apply_random_augmentations as apply_random_augmentations
from .spatial import ensure_mask_3d as ensure_mask_3d
from .spatial import (
    extract_num_frames_from_input_specs as extract_num_frames_from_input_specs,
)
from .spatial import (
    extract_target_size_from_input_specs as extract_target_size_from_input_specs,
)
from .spatial import get_base_transforms as get_base_transforms
from .spatial import get_random_augmentations as get_random_augmentations
from .spatial import get_random_augmentations_hard as get_random_augmentations_hard
from .spatial import get_random_augmentations_medium as get_random_augmentations_medium
from .video import (
    apply_video_robustness_augmentations as apply_video_robustness_augmentations,
)
from .video import (
    compress_video_frames_jpeg_torchvision as compress_video_frames_jpeg_torchvision,
)
