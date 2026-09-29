"""Distortions transforms for benchmark inputs."""

import math
import random

import cv2
import numpy as np


class ApplyDeeperForensicsDistortion:
    """Wrapper for applying DeeperForensics distortions."""

    def __init__(self, distortion_type, level_min=0, level_max=3, rng=None, pyrng=None):
        """
        Initialize distortion transform.

        Args:
            distortion_type: str, type of distortion to apply
            level_min: int, minimum distortion level
            level_max: int, maximum distortion level
            rng: np.random.RandomState (or None for the global module).
            pyrng: random.Random (or None for the global module).
        """
        self.__name__ = distortion_type
        self.distortion_type = distortion_type
        self.level = None
        self.level_min = level_min
        self.level_max = level_max
        self.rng = rng
        self.pyrng = pyrng
        self.params = {}  # level
        self.distortion_params = {}  # distortion_type specific

    def __call__(self, img, level=None, **kwargs):
        """
        Apply distortion transform.

        Args:
            img: np.ndarray, input image to distort
            level: int or None, optional distortion level
            **kwargs: additional keyword arguments

        Returns:
            np.ndarray: Distorted image
        """
        if level is None and self.level is None:
            pyrng = self.pyrng if self.pyrng is not None else random
            self.level = pyrng.randint(self.level_min, self.level_max)
            self.params = {"level": self.level}
        elif self.level is None:
            self.level = level
            self.params = {"level": self.level}

        if self.level > 0:
            self.distortion_func = get_distortion_function(self.distortion_type)
            if len(self.distortion_params) == 0:
                self.distortion_param = get_distortion_parameter(
                    self.distortion_type, self.level
                )
                self.distortion_params = {"param": self.distortion_param}
        else:
            return img

        # Only the noise distortions consume RNG; pass the local generator to them.
        if self.distortion_type in ("GNC", "BW"):
            output = self.distortion_func(img, rng=self.rng, **self.distortion_params)
        else:
            output = self.distortion_func(img, **self.distortion_params)
        if isinstance(output, tuple):
            self.distortion_params.update(output[1])
            return output[0]
        else:
            return output


def get_distortion_parameter(distortion_type, level):
    """Get distortion parameter based on type and level.

    Args:
        distortion_type: str, type of distortion
        level: int, distortion level (1-5)

    Returns:
        float or int: Parameter value for the specified distortion

    Parameters are arranged from least severe (level 1) to most severe (level 5).
    Each distortion type has different parameter behavior:

    CS (Color Saturation):
        - Range: [0.4 -> 0.0]
        - Lower values = worse distortion
        - 0.4 = slight desaturation
        - 0.0 = complete desaturation (grayscale)

    CC (Color Contrast):
        - Range: [0.85 -> 0.35]
        - Lower values = worse distortion
        - 0.85 = slight contrast reduction
        - 0.35 = severe contrast reduction

    BW (Block Wise):
        - Range: [16 -> 80]
        - Higher values = worse distortion
        - Controls number of random blocks added
        - 16 = few blocks
        - 80 = many blocks

    GNC (Gaussian Noise Color):
        - Range: [0.001 -> 0.05]
        - Higher values = worse distortion
        - Controls noise variance
        - 0.001 = subtle noise
        - 0.05 = very noisy

    GB (Gaussian Blur):
        - Range: [7 -> 21]
        - Higher values = worse distortion
        - Controls blur kernel size
        - 7 = slight blur
        - 21 = heavy blur

    JPEG (JPEG Compression):
        - Range: [2 -> 6]
        - Higher values = worse distortion
        - Controls downsampling factor
        - 2 = mild compression
        - 6 = severe compression
    """
    param_dict = {
        "CS": [0.4, 0.3, 0.2, 0.1, 0.0],
        "CC": [0.85, 0.725, 0.6, 0.475, 0.35],
        "BW": [16, 32, 48, 64, 80],
        "GNC": [0.001, 0.002, 0.005, 0.01, 0.05],
        "GB": [7, 9, 13, 17, 21],
        "JPEG": [2, 3, 4, 5, 6],
    }
    return param_dict[distortion_type][level - 1]


def get_distortion_function(distortion_type):
    """
    Args:
        distortion_type: str, type of distortion

    Returns:
        callable: Function that implements the specified distortion
    """
    func_dict = {
        "CS": color_saturation,
        "CC": color_contrast,
        "BW": block_wise,
        "GNC": gaussian_noise_color,
        "GB": gaussian_blur,
        "JPEG": jpeg_compression,
    }
    return func_dict[distortion_type]


def rgb2ycbcr(img_rgb):
    """Convert RGB image to YCbCr color space.

    Args:
        img_rgb (np.ndarray): RGB image array of shape (H, W, 3)

    Returns:
        np.ndarray: YCbCr image array of shape (H, W, 3) with values normalized to [0,1]
    """
    img_rgb = img_rgb.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    """Convert YCbCr image to RGB color space.

    Args:
        img_ycbcr (np.ndarray): YCbCr image array of shape (H, W, 3)

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with values in [0,255]
    """
    img_ycbcr = img_ycbcr.astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return img_rgb


def color_saturation(img, param):
    """Apply color saturation distortion.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Saturation multiplier parameter

    Returns:
        np.ndarray: Distorted RGB image array with modified saturation
    """
    ycbcr = rgb2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2rgb(ycbcr).astype(np.uint8)
    return img


def color_contrast(img, param):
    """Apply color contrast distortion.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Contrast multiplier parameter

    Returns:
        np.ndarray: Distorted RGB image array with modified contrast
    """
    img = img.astype(np.float32) * param
    return img.astype(np.uint8)


def block_wise(img, param, rng=None):
    """Apply block-wise distortion by adding random gray blocks.

    NOTE: CURRENTLY NOT USED

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Number of blocks to add, scaled by image dimensions
        rng: np.random.RandomState (or None for the global module).

    Returns:
        np.ndarray: Distorted RGB image array with added gray blocks
    """
    draw = rng if rng is not None else np.random
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for _ in range(param):
        r_w = draw.randint(0, img.shape[1] - width)
        r_h = draw.randint(0, img.shape[0] - width)
        img[r_h : r_h + width, r_w : r_w + width, :] = block
    return img


def gaussian_noise_color(img, param, rng=None):
    """Apply colored Gaussian noise in YCbCr color space.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Variance of the Gaussian noise
        rng: np.random.RandomState (or None for the global module).

    Returns:
        tuple: (distorted_image, params)
            distorted_image: np.ndarray, image with added color noise
    """
    draw = rng if rng is not None else np.random
    ycbcr = rgb2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) * draw.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2rgb(b)
    return np.clip(b, 0, 255).astype(np.uint8)


def gaussian_blur(img, param):
    """Apply Gaussian blur with specified kernel size.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Gaussian kernel size (must be odd)

    Returns:
        np.ndarray: Blurred RGB image array
    """
    return cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)


def jpeg_compression(img, param):
    """Apply JPEG compression-like distortion through downsampling.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Downsampling factor

    Returns:
        np.ndarray: Distorted RGB image array with compression artifacts
    """
    h, w, _ = img.shape
    s_h = max(1, h // param)
    s_w = max(1, w // param)
    img = cv2.resize(img, (s_w, s_h))
    return cv2.resize(img, (w, h))
