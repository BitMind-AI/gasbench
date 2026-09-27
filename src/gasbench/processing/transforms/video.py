"""Video transforms for benchmark inputs."""

import os

import cv2
import numpy as np
import torch
from torchvision.io import decode_jpeg, encode_jpeg

from .spatial import apply_random_augmentations


def _decode_video_rgb(tmp_path, num_frames):
    """Decode up to num_frames RGB frames from a video file, padding the last
    frame if the decoder returns fewer. Returns a (T, H, W, 3) uint8 array or
    None on failure."""
    cap = cv2.VideoCapture(tmp_path)
    decoded = []
    while len(decoded) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not decoded:
        return None
    while len(decoded) < num_frames:
        decoded.append(decoded[-1])
    return np.stack(decoded[:num_frames], axis=0)


def _h264_roundtrip_ffmpeg(video_array, crf, fps):
    """Faithful H.264 roundtrip via the ffmpeg CLI using a real ``-crf`` value.

    This is the only path that reproduces the FaceForensics++ CRF protocol
    exactly; cv2's VideoWriter quality knob does not map to CRF and is ignored
    on many OpenCV builds. Returns the decoded (T, H, W, 3) uint8 array, or
    None if ffmpeg is unavailable or the roundtrip fails (caller falls back).
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        return None

    T, H, W, C = video_array.shape
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            tmp_path,
        ]
        proc = subprocess.run(
            cmd,
            input=np.ascontiguousarray(video_array).tobytes(),
            capture_output=True,
        )
        if (
            proc.returncode != 0
            or not os.path.exists(tmp_path)
            or os.path.getsize(tmp_path) == 0
        ):
            return None
        return _decode_video_rgb(tmp_path, T)
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _h264_roundtrip_cv2(video_array, crf, fps):
    """Best-effort H.264 roundtrip via cv2's avc1 writer. CRF cannot be set
    directly, so it is approximated through VIDEOWRITER_PROP_QUALITY (a perceptual
    0-100 knob that some builds ignore). Returns (T, H, W, 3) uint8 or None."""
    import tempfile

    T, H, W, C = video_array.shape
    # Map CRF [18, 51] → cv2 quality [100, 0] linearly (approximate only)
    cv2_quality = max(0, min(100, round((51 - crf) / 33.0 * 100)))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (W, H))
        if not writer.isOpened():
            writer.release()
            return None

        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, cv2_quality)
        for t in range(T):
            writer.write(cv2.cvtColor(video_array[t], cv2.COLOR_RGB2BGR))
        writer.release()
        return _decode_video_rgb(tmp_path, T)
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def apply_video_robustness_augmentations(
    video_array,
    target_size,
    seed=None,
    crf=23,
    fps=25,
    scale_factor=0.5,
):
    """H.264 compression roundtrip for video augmentation robustness evaluation.

    Mirrors the FaceForensics++ c23/c40 evaluation protocol — encode to H.264
    at a given CRF then decode back, simulating platform re-encoding pipelines.
    CRF 23 = light (FF++ c23, YouTube-tier), CRF 40 = heavy (FF++ c40,
    WhatsApp/Messenger-tier).

    Encoding tries, in order:
      1. ffmpeg CLI with a real ``-crf`` value — faithful FF++ reproduction.
      2. cv2 avc1 writer with an approximate quality mapping — used only if
         ffmpeg is not on PATH.
      3. per-frame JPEG at an equivalent severity — last-resort fallback that
         still preserves chroma-subsampling artifacts.

    scale_factor < 1.0 first downscales every frame (resolution ladder) before
    encoding, mirroring platform transcodes that drop 1080p → 720p → 480p.
    Defaults to 0.5 to match the image robustness pipeline and reflect real
    platform behaviour (WhatsApp, Instagram, Twitter all downscale on ingest).

    Returns the same 4-tuple as apply_random_augmentations for drop-in use in
    VideoPrefetchPipeline when robustness_pass=True.
    """
    if video_array.dtype != np.uint8:
        video_array = np.clip(video_array, 0, 255).astype(np.uint8)

    # Resolution ladder — downscale frames before encoding (platform transcode).
    # Floor at 256px: platforms don't downscale content that's already small,
    # and AI-generated content being distributed is at minimum 256px.
    _MIN_DOWNSCALE_PX = 256
    if scale_factor < 1.0:
        T, H, W, C = video_array.shape
        sh = max(_MIN_DOWNSCALE_PX, int(round(H * scale_factor)))
        sw = max(_MIN_DOWNSCALE_PX, int(round(W * scale_factor)))
        if sh < H and sw < W:
            video_array = np.stack(
                [
                    cv2.resize(video_array[t], (sw, sh), interpolation=cv2.INTER_AREA)
                    for t in range(T)
                ],
                axis=0,
            )

    T, H, W, C = video_array.shape

    # H.264 with yuv420p (both the ffmpeg -crf and cv2 avc1 encoders) requires
    # even width and height; libx264 rejects odd dimensions outright. Trim a
    # trailing row/column when needed so encoding succeeds instead of silently
    # erroring into the fallback path (which would also mis-set method). At most
    # one pixel per axis is dropped, and the frame is resized to target anyway.
    even_h, even_w = H - (H % 2), W - (W % 2)
    if (even_h, even_w) != (H, W) and even_h >= 2 and even_w >= 2:
        video_array = video_array[:, :even_h, :even_w, :]
        T, H, W, C = video_array.shape

    method = "ffmpeg_crf"
    compressed = _h264_roundtrip_ffmpeg(video_array, crf, fps)
    if compressed is None:
        method = "cv2_avc1"
        compressed = _h264_roundtrip_cv2(video_array, crf, fps)
    if compressed is None:
        # Fallback: per-frame JPEG at quality approximating the requested CRF severity
        method = "jpeg_fallback"
        fallback_q = max(20, min(95, round(100 - (crf - 18) * 2.3)))
        compressed = compress_video_frames_jpeg_torchvision(
            video_array, quality=fallback_q
        )

    # Resize to target via base transforms (no random crop/flip)
    aug_thwc, _, _, _ = apply_random_augmentations(
        compressed, target_size, seed=seed, level=0, crop_prob=0.0
    )

    params = {"crf": crf, "fps": fps, "scale_factor": scale_factor, "method": method}
    return aug_thwc, None, "robustness_video", params


def compress_video_frames_jpeg_torchvision(
    video_thwc: np.ndarray, quality: int = 75
) -> np.ndarray:
    """
    Compress each frame of a video using torchvision's encode_jpeg/decode_jpeg at fixed quality.

    Args:
        video_thwc: numpy array (T, H, W, C), dtype uint8, RGB
        quality: JPEG quality (default 75)

    Returns:
        numpy array (T, H, W, C), dtype uint8, RGB
    """
    if video_thwc is None:
        return video_thwc
    if video_thwc.dtype != np.uint8:
        video_thwc = np.clip(video_thwc, 0, 255).astype(np.uint8)
    if video_thwc.ndim != 4 or video_thwc.shape[-1] != 3:
        return video_thwc

    T, H, W, C = video_thwc.shape
    out = np.empty_like(video_thwc)

    for t in range(T):
        frame_chw = (
            torch.from_numpy(video_thwc[t]).permute(2, 0, 1).contiguous()
        )  # CHW uint8
        encoded_bytes = encode_jpeg(frame_chw, quality=int(quality))
        decoded_chw = decode_jpeg(encoded_bytes)
        out[t] = decoded_chw.permute(1, 2, 0).contiguous().numpy()

    return out
