import os
import subprocess
import tempfile
from .audio_decode import AUDIO_DECODE_TIMEOUT, decode_audio as _decode_audio_with_timeout
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch

from ..logger import get_logger
from ..constants import AUDIO_DURATION_SECONDS, AUDIO_SAMPLE_RATE, media_type_to_label

from .video_decode import decode_video

logger = get_logger(__name__)


def configure_huggingface_cache(volume_dir: str = "/benchmark_data"):
    """Configure HuggingFace to use consolidated temp directory for all downloads."""
    global _hf_cache_configured

    temp_dir = os.path.join(volume_dir, "temp_downloads")
    hf_cache_dir = os.path.join(temp_dir, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")

    if not _hf_cache_configured:
        logger.info(f"Configured HuggingFace cache: {hf_cache_dir}")
        _hf_cache_configured = True

    return hf_cache_dir


def process_video_bytes_sample(
    sample: Dict,
    num_frames: int = 16,
    frame_rate: Optional[float] = None,
) -> Tuple[any, int]:
    """Decode a cached video path or raw bytes into RGB uint8 THWC frames.

    Paths are opened directly; byte-backed samples stay in memory. Both use
    sequential prefix decoding, optional frame-index strides and last-frame
    padding. Unreadable prefixes return ``(None, None)``; damage beyond the
    requested prefix is not scanned.
    """
    try:
        source = sample.get("video_path") or sample.get("video_bytes")
        if not source:
            return None, None
        label = media_type_to_label(sample.get("media_type", "synthetic"), "video")
        return decode_video(source, num_frames, frame_rate), label
    except Exception as e:
        logger.warning(f"Failed to process video sample: {e}")
        return None, None


def process_video_frames_sample(
    sample: Dict,
    num_frames: int = 16,
) -> Tuple[any, int]:
    """Process a video sample that contains pre-extracted frames (list of frame paths or bytes).

    This is used for datasets where frames are already extracted (e.g., PNG files in directories).
    Returns the same format as process_video_bytes_sample: (T, H, W, C) uint8 numpy array.

    Uses cv2 for fast image loading, converts BGR to RGB.

    Args:
        sample: Dict containing either:
            - 'video_frames': List of frame file paths or frame bytes
            - 'media_type': 'real', 'synthetic', or 'semisynthetic'
        num_frames: Number of frames to use (default 16).

    Returns:
        Tuple of (video_array, label) where video_array is (num_frames, H, W, 3) uint8 numpy array in RGB
    """
    try:
        frames_data = sample.get("video_frames")
        if not frames_data:
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = media_type_to_label(media_type, "video")

        frames = []

        for frame_data in frames_data[:num_frames]:
            try:
                if isinstance(frame_data, (str, Path)):
                    frame_array = cv2.imread(str(frame_data))
                    if frame_array is None:
                        logger.warning(f"Failed to load frame: {frame_data}")
                        continue
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
                elif isinstance(frame_data, bytes):
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_array is None:
                        logger.warning("Failed to decode frame bytes")
                        continue
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
                else:
                    logger.warning(f"Unsupported frame data type: {type(frame_data)}")
                    continue

                frames.append(frame_array)

            except Exception as e:
                logger.warning(f"Failed to load frame: {e}")
                continue

        if len(frames) == 0:
            logger.warning("No frames could be loaded")
            return None, None

        if len(frames) < num_frames:
            last_frame = frames[-1]
            for _ in range(len(frames), num_frames):
                frames.append(last_frame)

        video_array = np.array(frames, dtype=np.uint8)

        return video_array, label

    except Exception as e:
        logger.warning(f"Failed to process video frames sample: {e}")
        return None, None


def process_image_sample(sample: Dict) -> Tuple[any, int]:
    """Process an image sample (bytes) for classification evaluation."""
    try:
        image_bytes = sample.get("image") or sample.get("image_bytes")
        if image_bytes is None:
            logger.warning("No image bytes in sample")
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = media_type_to_label(media_type, "image")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        image.close()

        return image_array, label

    except Exception as e:
        logger.warning(f"Failed to process image sample: {e}")
        return None, None


def _decode_audio_waveform_ffmpeg_cli(audio_bytes: bytes, target_sr: int) -> torch.Tensor:
    """Decode arbitrary audio bytes to mono float32 PCM using the ffmpeg CLI.

    Used when TorchCodec cannot load (e.g. PyTorch wheel vs system FFmpeg ABI mismatch on
    Ubuntu 22.04 / Modal images). Requires ``ffmpeg`` on PATH.
    """
    if not audio_bytes:
        raise ValueError("empty audio bytes")

    tmp = tempfile.NamedTemporaryFile(prefix="gasbench_audio_", delete=False)
    tmp_path = tmp.name
    try:
        tmp.write(audio_bytes)
        tmp.close()
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
            tmp_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(target_sr)),
            "-f",
            "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=AUDIO_DECODE_TIMEOUT,
            check=False,
        )
        if proc.returncode != 0:
            err = (
                proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
            )
            raise RuntimeError(
                f"ffmpeg decode failed (exit {proc.returncode}): {err.strip()}"
            )
        raw = proc.stdout
        if not raw or len(raw) % 4 != 0:
            raise RuntimeError("invalid f32le output from ffmpeg")
        pcm = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.from_numpy(pcm)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def process_audio_sample(
    sample: Dict,
    target_sr: int = AUDIO_SAMPLE_RATE,
    target_duration_seconds: float = AUDIO_DURATION_SECONDS,
    use_random_crop: bool = False,
    seed: Optional[int] = 42,
    device: Optional[str] = None,
) -> Tuple[any, int]:
    """
    Process an audio sample for deepfake detection benchmark.

    Decodes with TorchCodec when available; otherwise uses the ``ffmpeg`` CLI (same
    sample rate and mono), which matches typical Modal/Ubuntu images that ship ffmpeg
    but lack the libavutil versions bundled TorchCodec expects.

    Preprocessing Pipeline:
    1. Decode audio bytes (TorchCodec or ffmpeg fallback)
    2. Crop/pad to exactly 6 seconds (96,000 samples at 16kHz)
       - If longer: random crop (with seed) or center crop
       - If shorter: zero-pad on the right

    A 30-second per-sample timeout prevents ffmpeg deadlocks on malformed files.
    No normalization is applied -- decoded float32 PCM is in roughly [-1, 1].

    Args:
        sample: Dictionary containing 'audio_bytes' and metadata
        target_sr: Target sample rate (default 16000 Hz)
        target_duration_seconds: Target duration in seconds (default 6.0)
        use_random_crop: If True, randomly crop; if False, center crop
        seed: Random seed for deterministic cropping
        device: Unused (kept for API compatibility)

    Returns:
        Tuple of (waveform, label)
        - waveform: torch.Tensor of shape (96000,) as float32 on CPU
        - label: int (0 for real, 1 for synthetic)
    """
    try:
        audio_bytes = sample.get("audio_bytes")
        if audio_bytes is None:
            logger.warning("No audio bytes in sample")
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = media_type_to_label(media_type, "audio")

        # Decode with timeout (TorchCodec or ffmpeg CLI)
        waveform = _decode_audio_with_timeout(audio_bytes, target_sr)

        # Crop/pad to target length (e.g. 96,000 samples = 6s at 16kHz)
        target_length = int(target_sr * target_duration_seconds)

        if waveform.shape[0] > target_length:
            if use_random_crop:
                generator = torch.Generator().manual_seed(seed) if seed is not None else None
                max_start = waveform.shape[0] - target_length
                start_idx = torch.randint(0, max_start + 1, (1,), generator=generator).item()
            else:
                start_idx = (waveform.shape[0] - target_length) // 2
            waveform = waveform[start_idx:start_idx + target_length]
        elif waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform.float(), label

    except TimeoutError:
        logger.warning(
            f"Audio decode timed out after {AUDIO_DECODE_TIMEOUT}s, skipping sample"
        )
        return None, None
    except Exception as e:
        logger.warning(f"Failed to process audio sample: {e}")
        return None, None


_hf_cache_configured = False
