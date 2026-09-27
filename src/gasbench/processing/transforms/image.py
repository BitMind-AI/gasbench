"""Image transforms for benchmark inputs."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from .spatial import get_base_transforms


def apply_robustness_augmentations(
    image_array,
    target_size,
    seed=None,
    jpeg_quality=55,
    scale_factor=0.5,
    webp_quality=75,
):
    """Fixed augmentation suite for image augmentation robustness evaluation.

    Simulates the dominant real-world internet distribution pipeline:
      1. Downscale + upscale — thumbnail/CDN resize chain
      2. JPEG roundtrip at jpeg_quality — first platform upload (e.g. WhatsApp ~55)
      3. WebP roundtrip at webp_quality — CDN/platform re-host (Facebook, Google)
      4. Second JPEG roundtrip at 80 — re-share / re-host recompression

    Step 3 exercises the cross-codec re-hosting case: many platforms serve
    WebP, whose VP8 intra coding leaves a different artifact family than JPEG
    DCT, so a detector that survives repeated JPEG can still collapse on it.
    Pass webp_quality=None to skip it and recover the JPEG-only chain.

    Returns the same 4-tuple as apply_random_augmentations for drop-in use
    in PrefetchPipeline when robustness_pass=True.  Deterministic given seed
    so paired base/augmented samples share sample_id for degradation join.
    """
    img = image_array.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]

    # Downscale then upscale — simulates share/thumbnail pipeline artifacts.
    # Floor at 256px: platforms don't downscale content that's already small,
    # and AI-generated content being distributed is at minimum 256px (DeeperForensics
    # evaluation floor; below this the step is unrealistic, not representative).
    _MIN_DOWNSCALE_PX = 256
    if scale_factor < 1.0:
        small_h = max(_MIN_DOWNSCALE_PX, int(round(h * scale_factor)))
        small_w = max(_MIN_DOWNSCALE_PX, int(round(w * scale_factor)))
        if small_h < h and small_w < w:
            img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # First JPEG pass — heavy platform compression (WhatsApp/Telegram ~q55)
    img = compress_image_jpeg_pil(img, quality=jpeg_quality)

    # Cross-codec re-host — CDN/platform WebP transcode (Facebook, Google)
    if webp_quality is not None:
        img = compress_image_webp_pil(img, quality=webp_quality)

    # Second JPEG pass — lighter re-share recompression (Twitter/Instagram ~q80)
    img = compress_image_jpeg_pil(img, quality=80)

    # Resize to model input size (same crop+resize as base pipeline)
    tforms = get_base_transforms(target_size, (1.0, 1.0))
    aug_hwc, _ = tforms(img, None, reuse_params=False)

    params = {
        "jpeg_quality": jpeg_quality,
        "scale_factor": scale_factor,
        "webp_quality": webp_quality,
        "jpeg_quality_2": 80,
    }
    return aug_hwc, None, "robustness", params


def compress_image_jpeg_pil(image_hwc: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Compress a single image using PIL JPEG round-trip at fixed quality.

    Args:
        image_hwc: numpy array (H, W, C), dtype uint8, RGB
        quality: JPEG quality (default 75)

    Returns:
        numpy array (H, W, C), dtype uint8, RGB
    """
    if image_hwc is None:
        return image_hwc
    if image_hwc.dtype != np.uint8:
        image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        return image_hwc

    pil_img = Image.fromarray(image_hwc, mode="RGB")
    buffer = BytesIO()
    # subsampling=2 forces 4:2:0 chroma subsampling regardless of quality/Pillow
    # version. This is the operation the social-media compression literature
    # identifies as destroying high-frequency DCT fingerprints, so we pin it
    # rather than letting Pillow pick subsampling per quality level.
    pil_img.save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    decoded_pil = Image.open(buffer).convert("RGB")
    return np.array(decoded_pil)


def compress_image_webp_pil(image_hwc: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Compress a single image using a PIL WebP (lossy) round-trip at fixed quality.

    Facebook, Google, and many CDNs re-host uploads as WebP, whose VP8 intra
    coding leaves a different artifact family than JPEG's DCT blocks. Including
    a WebP pass alongside the JPEG passes exercises detectors against the
    cross-codec re-hosting that real distribution chains produce.

    Args:
        image_hwc: numpy array (H, W, C), dtype uint8, RGB
        quality: WebP quality (default 75)

    Returns:
        numpy array (H, W, C), dtype uint8, RGB
    """
    if image_hwc is None:
        return image_hwc
    if image_hwc.dtype != np.uint8:
        image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        return image_hwc

    try:
        pil_img = Image.fromarray(image_hwc, mode="RGB")
        buffer = BytesIO()
        pil_img.save(buffer, format="WEBP", quality=int(quality), method=4)
        buffer.seek(0)
        decoded_pil = Image.open(buffer).convert("RGB")
        return np.array(decoded_pil)
    except Exception:
        # WebP support is missing in some Pillow builds; fall back to original.
        return image_hwc
