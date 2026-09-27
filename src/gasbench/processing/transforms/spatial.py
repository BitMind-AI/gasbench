"""Spatial transforms for benchmark inputs."""

import math
import random

import cv2
import numpy as np

from .distortions import ApplyDeeperForensicsDistortion


def extract_target_size_from_input_specs(input_specs):
    """
    Extract target (H, W) dimensions from model input specs.
    Expected shapes:
        - Image: [batch, channels, height, width] or [batch, height, width, channels]
        - Video: [batch, time, channels, height, width] or [batch, time, height, width, channels]
    Returns:
        tuple: (H, W) target dimensions, or None if cannot be determined (e.g., dynamic axes)

    """
    if not input_specs or len(input_specs) == 0:
        return None

    shape = input_specs[0].shape
    if not shape or len(shape) < 3:
        return None

    shape = [dim if isinstance(dim, int) else None for dim in shape]

    if len(shape) == 5:
        # Assume [B, T, C, H, W] format for video models
        h, w = shape[3], shape[4]
        if h is not None and w is not None:
            return (h, w)

    elif len(shape) == 4:
        # Assume [B, C, H, W] format for image models
        h, w = shape[2], shape[3]
        if h is not None and w is not None:
            return (h, w)

    return None


def extract_num_frames_from_input_specs(input_specs):
    """
    Extract the number of frames (T) from video model input specs.
    Expected shape: [batch, time, channels, height, width].

    Returns:
        int: number of frames, or None if shape is dynamic or not a video model
    """
    if not input_specs or len(input_specs) == 0:
        return None

    shape = input_specs[0].shape
    if not shape or len(shape) != 5:
        return None

    t = shape[1]
    return t if isinstance(t, int) else None


def ensure_mask_3d(mask: np.ndarray) -> np.ndarray:
    """
    Ensure the mask is 3D (H, W, 1) if it's 2D (H, W).
    """
    if mask.ndim == 2:
        return mask[:, :, None]
    return mask


def apply_random_augmentations(
    inputs,
    target_size,
    mask=None,
    level_probs=None,
    level=None,
    crop_prob=0.5,
    seed=None,
):
    """
    Apply image transformations based on randomly selected difficulty level.

    Args:
        inputs: np.ndarray or tuple of np.ndarray. Image(s) to transform
        target_size: tuple (H, W). Target size for resize (from model specs or default 224x224)
        mask: np.ndarray, optional. Binary mask to ensure crop contains mask foreground.
            If provided, the returned aug_mask may be 3D (H, W, 1); squeeze to 2D (H, W) for storage or training as needed.
        level_probs: dict with augmentation levels and their probabilities.
            Default probabilities:
            - Level 0 (25%): No augmentations (base transforms)
            - Level 1 (25%): Basic augmentations
            - Level 2 (25%): Medium distortions
            - Level 3 (25%): Hard distortions
        level: set to override level_probs
        crop_prob: probability of cropping randomly
        seed: int, optional. Random seed for reproducible augmentations

    Returns:
        tuple: (aug_image, aug_mask, level, transform_params)
            aug_image: Augmented image(s) as np.ndarray
            aug_mask: Augmented mask (if provided, else None)
                (Note: aug_mask may be 3D (H, W, 1); squeeze to 2D for storage/training.)
            level: int, chosen augmentation level
            transform_params: dict, parameters used for the transforms

    Raises:
        ValueError: If probabilities don't sum to 1.0 (within floating point precision)
    """
    # Use local generators seeded per-call rather than seeding the global RNG.
    # The pipeline runs these in parallel worker threads; seeding the process-wide
    # np.random/random state would let concurrent threads clobber each other's
    # seed, breaking the determinism the seed is supposed to guarantee.
    # RandomState/random.Random reproduce the legacy global-seed algorithm exactly,
    # so seeded outputs (and the aug cache) are unchanged.
    if seed is not None:
        rng = np.random.RandomState(seed)
        pyrng = random.Random(seed)
    else:
        rng = np.random
        pyrng = random
    if level is None:
        if level_probs is None:
            level_probs = {
                0: 0.25,  # No augmentations (base transforms)
                1: 0.25,  # Basic augmentations
                2: 0.25,  # Medium distortions
                3: 0.25,  # Hard distortions
            }

        if not math.isclose(sum(level_probs.values()), 1.0, rel_tol=1e-9):
            raise ValueError("Probabilities of levels must sum to 1.0")

        # get cumulative probs and select augmentation level
        cumulative_probs = {}
        cumsum = 0
        for level, prob in sorted(level_probs.items()):
            cumsum += prob
            cumulative_probs[level] = cumsum

        rand_val = rng.random_sample()
        for curr_level, cum_prob in cumulative_probs.items():
            if rand_val <= cum_prob:
                level = curr_level
                break

    # determine crop scale for h and w
    crop_scale = (1.0, 1.0)
    if rng.rand() < crop_prob:
        crop_scale = (rng.uniform(0.35, 0.99), rng.uniform(0.35, 0.99))
        min_scale_h = max(0.35, 224 / target_size[0])
        min_scale_w = max(0.35, 224 / target_size[1])
        crop_scale = (max(crop_scale[0], min_scale_h), max(crop_scale[1], min_scale_w))

    if level == 0:
        tforms = get_base_transforms(target_size, crop_scale, rng, pyrng)
    elif level == 1:
        tforms = get_random_augmentations(target_size, crop_scale, rng, pyrng)
    elif level == 2:
        tforms = get_random_augmentations_medium(target_size, crop_scale, rng, pyrng)
    else:  # level == 3
        tforms = get_random_augmentations_hard(target_size, crop_scale, rng, pyrng)

    if isinstance(inputs, tuple):
        transformed_A, _ = tforms(inputs[0], reuse_params=False)
        transformed_B, _ = tforms(inputs[1], reuse_params=True)
        transformed = np.concatenate([transformed_A, transformed_B], axis=0)

        return transformed, None, level, tforms.params
    else:
        transformed_inputs, transformed_masks = tforms(inputs, mask, reuse_params=False)
        return transformed_inputs, transformed_masks, level, tforms.params


def get_base_transforms(target_res, crop_scale, rng=None, pyrng=None):
    """
    Get basic transforms (optional crop and resize).

    Args:
        target_res: int or tuple. Output size for resize.
        crop_scale: tuple. Crop scale (ignored if (1.0, 1.0)).
        rng: np.random.RandomState (or None for the global module).
        pyrng: random.Random (or None for the global module).

    Returns:
        ComposeWithParams: Composed transform pipeline
    """
    transforms_list = []

    if crop_scale != (1.0, 1.0):
        transforms_list.append(RandomCropWithParams(crop_scale, rng=rng))

    transforms_list.append(ResizeShortestEdge(target_res))
    return ComposeWithParams(transforms_list)


def get_random_augmentations(target_res, crop_scale, rng=None, pyrng=None):
    """
    Get basic augmentations with geometric transforms.

    Args:
        target_res: int or tuple. Output size for resize.
        crop_scale: tuple. Crop scale.
        rng: np.random.RandomState (or None for the global module).
        pyrng: random.Random (or None for the global module).

    Returns:
        ComposeWithParams: Composed transform pipeline with basic augmentations
    """
    base = get_base_transforms(target_res, crop_scale, rng, pyrng)
    transforms_list = base.transforms + [
        RandomHorizontalFlipWithParams(rng=rng),
        RandomVerticalFlipWithParams(rng=rng),
    ]
    return ComposeWithParams(transforms_list)


def get_random_augmentations_medium(target_res, crop_scale, rng=None, pyrng=None):
    """
    Get medium difficulty transforms with mild distortions.

    Args:
        target_res: int or tuple. Output size for resize.
        crop_scale: tuple. Crop scale.
        rng: np.random.RandomState (or None for the global module).
        pyrng: random.Random (or None for the global module).

    Returns:
        ComposeWithParams: Composed transform pipeline with medium distortions
    """
    base = get_base_transforms(target_res, crop_scale, rng, pyrng)
    transforms_list = base.transforms + [
        RandomHorizontalFlipWithParams(rng=rng),
        RandomVerticalFlipWithParams(rng=rng),
        ApplyDeeperForensicsDistortion(
            "CS", level_min=0, level_max=1, rng=rng, pyrng=pyrng
        ),
        ApplyDeeperForensicsDistortion(
            "CC", level_min=0, level_max=1, rng=rng, pyrng=pyrng
        ),
    ]
    return ComposeWithParams(transforms_list)


def get_random_augmentations_hard(target_res, crop_scale, rng=None, pyrng=None):
    """
    Get hard difficulty transforms with more severe distortions.

    Args:
        target_res: int or tuple. Output size for resize.
        crop_scale: tuple. Crop scale.
        rng: np.random.RandomState (or None for the global module).
        pyrng: random.Random (or None for the global module).

    Returns:
        ComposeWithParams: Composed transform pipeline with severe distortions
    """
    base = get_base_transforms(target_res, crop_scale, rng, pyrng)
    transforms_list = base.transforms + [
        RandomHorizontalFlipWithParams(rng=rng),
        RandomVerticalFlipWithParams(rng=rng),
        ApplyDeeperForensicsDistortion(
            "CS", level_min=0, level_max=2, rng=rng, pyrng=pyrng
        ),
        ApplyDeeperForensicsDistortion(
            "CC", level_min=0, level_max=2, rng=rng, pyrng=pyrng
        ),
        ApplyDeeperForensicsDistortion(
            "GNC", level_min=0, level_max=2, rng=rng, pyrng=pyrng
        ),
        ApplyDeeperForensicsDistortion(
            "GB", level_min=0, level_max=2, rng=rng, pyrng=pyrng
        ),
    ]
    return ComposeWithParams(transforms_list)


class ComposeWithParams:
    def __init__(self, transforms):
        """
        Compose multiple transforms while tracking their randomly selected parameters.
        Useful for logging or situations where transforms need to be reapplied with
        the same parameters.

        Args:
            transforms: list of transform objects to compose
        """
        self.transforms = transforms
        self.params = {}

    def __call__(self, frames, masks=None, reuse_params=False):
        """
        Apply composed transforms to frames and optional masks.

        Args:
            frames: np.ndarray, the image(s) to transform
            masks: np.ndarray, optional mask(s) to transform
            reuse_params: bool, if True, reuse previous params (for paired images/mask)

        Returns:
            tuple: (output_frames, output_masks)
                output_frames: np.ndarray, transformed frames
                output_masks: np.ndarray or None, transformed masks if provided
        """
        if not reuse_params:
            self.params = {}

        is_single_image = frames.ndim == 3  # (H, W, C)
        if is_single_image:
            # Add fake temporal dim → (1, H, W, C)
            frames = frames[None, ...]
            masks = masks[None, ...] if masks is not None else None

        output_frames = []
        output_masks = []

        for i in range(frames.shape[0]):
            frame = frames[i]
            mask = ensure_mask_3d(masks[i]) if masks is not None else None

            for transform in self.transforms:
                frame, mask_ = self.apply_transform(
                    image=frame, mask=mask, transform=transform
                )
                mask = mask if mask_ is None else mask_

            output_frames.append(frame)
            if mask is not None:
                if mask.ndim == 3:
                    mask = (mask.sum(axis=2) > 0).astype(np.uint8)
                output_masks.append(mask)

        if is_single_image:
            output_frames = output_frames[0]
            if len(output_masks):
                output_masks = output_masks[0]
        else:
            output_frames = np.array(output_frames)
            if len(output_masks):
                output_masks = np.array(output_masks)
                output_masks = (output_masks > 0).astype(np.uint8)

        return output_frames, output_masks

    def apply_transform(self, image, mask, transform):
        """
        Apply a single transform while tracking its parameters.

        Args:
            image: np.ndarray, image to transform
            mask: np.ndarray or None, optional mask to transform
            transform: transform object to apply

        Returns:
            tuple: (transformed_image, transformed_mask)
                transformed_image: np.ndarray, transformed image
                transformed_mask: np.ndarray or None, transformed mask if provided
        """
        transform_name = getattr(transform, "__name__", transform.__class__.__name__)
        if transform_name in self.params and self.params[transform_name]:
            output = transform(image, mask=mask, **self.params[transform_name])
        else:
            output = transform(image, mask=mask)
            if hasattr(transform, "params"):
                self.params[transform_name] = transform.params

        if isinstance(output, tuple):
            tform_image = output[0]
            tform_mask = output[1]
        else:
            tform_image = output
            tform_mask = None

        return tform_image, tform_mask


class RandomCropWithParams:
    """Randomly crop an image with optional mask-aware cropping."""

    def __init__(self, crop_scale, rng=None):
        """
        Initialize random crop transform.

        Args:
            crop_scale (tuple): (height_scale, width_scale) for crop size as fraction of original
            rng: np.random.RandomState (or None for the global module).
        """
        self.params = None
        self.crop_scale = crop_scale
        self.rng = rng

    def __call__(self, img, mask=None, crop_params=None):
        """
        Apply random crop transform.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            crop_params (tuple, optional): Pre-computed crop parameters (i, j, h, w)

        Returns:
            np.ndarray or tuple: Cropped image, or tuple of (image, mask) if mask provided
        """
        if crop_params is None:
            draw = self.rng if self.rng is not None else np.random
            height, width = img.shape[:2]
            h = max(1, int(height * self.crop_scale[0]))
            w = max(1, int(width * self.crop_scale[1]))

            h = min(h, height)
            w = min(w, width)

            if mask is not None:
                coords = np.where(mask > 0)
                ys, xs = coords[0], coords[1]

                if len(xs) == 0 or len(ys) == 0:
                    # No foreground, fall back to random crop
                    i = draw.randint(0, height - h + 1) if height > h else 0
                    j = draw.randint(0, width - w + 1) if width > w else 0
                else:
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()
                    min_i = max(0, y1 - h + 1)
                    max_i = min(y0, height - h)
                    min_j = max(0, x1 - w + 1)
                    max_j = min(x0, width - w)
                    if min_i > max_i or min_j > max_j:
                        i = max(0, y0 - (h // 2))
                        j = max(0, x0 - (w // 2))
                    else:
                        i = draw.randint(min_i, max_i + 1) if max_i >= min_i else 0
                        j = draw.randint(min_j, max_j + 1) if max_j >= min_j else 0
            else:
                i = draw.randint(0, height - h + 1) if height > h else 0
                j = draw.randint(0, width - w + 1) if width > w else 0
        else:
            i, j, h, w = crop_params

        self.params = {"crop_params": (i, j, h, w)}

        # Crop image
        if img.ndim == 2:
            cropped_img = img[i : i + h, j : j + w]
        else:
            cropped_img = img[i : i + h, j : j + w, :]

        # Crop mask if provided
        if mask is not None:
            if mask.ndim == 2:
                cropped_mask = mask[i : i + h, j : j + w]
            else:
                cropped_mask = mask[i : i + h, j : j + w, :]
            return cropped_img, cropped_mask

        return cropped_img


class ResizeShortestEdge:
    """Center crop to target aspect ratio, then resize to exact target size.

    Cropping before resizing ensures every image is downsampled by the same
    factor regardless of original aspect ratio, avoiding inconsistent
    interpolation artifacts across differently-shaped inputs.
    """

    def __init__(self, target_size):
        """
        Args:
            target_size (int or tuple): Target size (H, W)
        """
        if isinstance(target_size, int):
            self.target_h = self.target_w = target_size
        else:
            self.target_h, self.target_w = target_size

    def __call__(self, img, mask=None):
        """
        Apply center-crop-then-resize transform.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array

        Returns:
            np.ndarray or tuple: Transformed image, or tuple of (image, mask) if mask provided
        """
        h, w = img.shape[:2]

        # Largest crop that fits in the image and matches the target aspect ratio
        scale = min(h / self.target_h, w / self.target_w)
        crop_h = min(int(round(scale * self.target_h)), h)
        crop_w = min(int(round(scale * self.target_w)), w)

        # Center crop
        i = (h - crop_h) // 2
        j = (w - crop_w) // 2

        if img.ndim == 2:
            img = img[i : i + crop_h, j : j + crop_w]
        else:
            img = img[i : i + crop_h, j : j + crop_w, :]

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[i : i + crop_h, j : j + crop_w]
            else:
                mask = mask[i : i + crop_h, j : j + crop_w, :]

        # Resize to exact target
        img = cv2.resize(
            img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
        )
        if mask is not None:
            mask = cv2.resize(
                mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST
            )

        if mask is not None:
            return img, mask
        return img


class RandomHorizontalFlipWithParams:
    def __init__(self, p=0.5, rng=None):
        """
        Args:
            p (float): Probability of flipping the image
            rng: np.random.RandomState (or None for the global module).
        """
        self.p = p
        self.rng = rng
        self.params = {}

    def __call__(self, img, mask=None, flip=None):
        """
        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            flip (bool, optional): Pre-computed flip decision

        Returns:
            np.ndarray or tuple: Flipped image, or tuple of (image, mask) if mask provided
        """
        if flip is not None:
            self.params = {"flip": flip}
        elif not hasattr(self, "params") or len(self.params) == 0:
            draw = self.rng if self.rng is not None else np.random
            flip = draw.random_sample() < self.p
            self.params = {"flip": flip}

        if self.params.get("flip", False):
            img = np.fliplr(img)
            mask = None if mask is None else np.fliplr(mask)

        if mask is not None:
            return img, mask
        return img


class RandomVerticalFlipWithParams:
    def __init__(self, p=0.5, rng=None):
        """
        Args:
            p (float): Probability of flipping the image
            rng: np.random.RandomState (or None for the global module).
        """
        self.p = p
        self.rng = rng
        self.params = {}

    def __call__(self, img, mask=None, flip=None):
        """
        Apply vertical flip transform.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            flip (bool, optional): Pre-computed flip decision

        Returns:
            np.ndarray or tuple: Flipped image, or tuple of (image, mask) if mask provided
        """
        if flip is not None:
            self.params = {"flip": flip}
        elif not hasattr(self, "params") or len(self.params) == 0:
            draw = self.rng if self.rng is not None else np.random
            flip = draw.random_sample() < self.p
            self.params = {"flip": flip}

        if self.params.get("flip", False):
            img = np.flipud(img)
            mask = None if mask is None else np.flipud(mask)

        if mask is not None:
            return img, mask
        return img
