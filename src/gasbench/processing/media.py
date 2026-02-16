import os
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from decord import VideoReader, cpu
from io import BytesIO
from PIL import Image
from torchcodec.decoders import AudioDecoder
import torch

from ..logger import get_logger
from ..constants import MEDIA_TYPE_TO_LABEL

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


def process_video_bytes_sample(sample: Dict) -> Tuple[any, int]:
    """Process a video sample that contains raw video bytes using decord.
    
    Returns uint8 data for ONNX models in RGB format.
    Decord outputs RGB directly (no BGR->RGB conversion needed).
    """
    try:
        video_bytes = sample.get("video_bytes")
        if not video_bytes:
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = MEDIA_TYPE_TO_LABEL[media_type]

        src_name = str(sample.get("source_file", ""))
        ext = Path(src_name).suffix.lower() if src_name else ".mp4"
        if ext not in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"):
            ext = ".mp4"

        temp_video = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_video_path = temp_video.name

        try:
            temp_video.write(video_bytes)
            temp_video.flush()
            temp_video.close()

            vr = VideoReader(temp_video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            
            if total_frames == 0:
                logger.warning(f"No frames in video")
                return None, None

            max_frames = 16
            frames = []
            
            for i in range(min(max_frames, total_frames)):
                frame = vr[i].asnumpy()  # Already RGB, shape (H, W, C)
                if frame is None or frame.size == 0:
                    logger.warning(f"Skipping invalid frame at index {i}")
                    continue
                frames.append(frame)

            if len(frames) == 0:
                logger.warning(f"No frames extracted from video")
                return None, None

            if len(frames) < max_frames:
                last_frame = frames[-1]
                for i in range(len(frames), max_frames):
                    frames.append(last_frame)

            video_array = np.array(frames, dtype=np.uint8)  # THWC uint8 RGB

            return video_array, label

        finally:
            try:
                os.unlink(temp_video_path)
            except:
                pass

    except Exception as e:
        logger.warning(f"Failed to process video bytes sample: {e}")
        return None, None


def process_video_frames_sample(sample: Dict) -> Tuple[any, int]:
    """Process a video sample that contains pre-extracted frames (list of frame paths or bytes).

    This is used for datasets where frames are already extracted (e.g., PNG files in directories).
    Returns the same format as process_video_bytes_sample: (T, H, W, C) uint8 numpy array with T=16 frames.
    
    Uses cv2 for fast image loading, converts BGR to RGB.

    Args:
        sample: Dict containing either:
            - 'video_frames': List of frame file paths or frame bytes
            - 'media_type': 'real', 'synthetic', or 'semisynthetic'

    Returns:
        Tuple of (video_array, label) where video_array is (16, H, W, 3) uint8 numpy array in RGB
    """
    try:
        frames_data = sample.get("video_frames")
        if not frames_data:
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = MEDIA_TYPE_TO_LABEL[media_type]

        max_frames = 16
        frames = []

        for frame_data in frames_data[:max_frames]:
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
                        logger.warning(f"Failed to decode frame bytes")
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

        if len(frames) < max_frames:
            last_frame = frames[-1]
            for i in range(len(frames), max_frames):
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
        label = MEDIA_TYPE_TO_LABEL[media_type]

        # Decode bytes -> PIL -> numpy (HWC uint8)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)

        return image_array, label

    except Exception as e:
        logger.warning(f"Failed to process image sample: {e}")
        return None, None


AUDIO_DECODE_TIMEOUT = 30  # seconds -- prevents ffmpeg deadlocks on malformed files


def _decode_audio_with_timeout(audio_bytes: bytes, target_sr: int, timeout: int = AUDIO_DECODE_TIMEOUT):
    """Decode audio bytes using TorchCodec with a timeout.

    Uses a thread pool to enforce a timeout on the underlying ffmpeg decode,
    which can hang indefinitely on certain malformed audio files.

    Args:
        audio_bytes: Raw audio file bytes
        target_sr: Target sample rate (resampling handled by AudioDecoder)
        timeout: Maximum seconds to wait for decode

    Returns:
        AudioSamples from TorchCodec (data shape: (num_channels, num_samples), float32 in [-1, 1])

    Raises:
        concurrent.futures.TimeoutError: If decode hangs beyond timeout
    """
    def _decode():
        decoder = AudioDecoder(audio_bytes, sample_rate=target_sr, num_channels=1)
        return decoder.get_all_samples()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_decode)
        return future.result(timeout=timeout)


def process_audio_sample(
    sample: Dict,
    target_sr: int = 16000,
    target_duration_seconds: float = 6.0,
    use_random_crop: bool = False,
    seed: Optional[int] = 42,
    device: Optional[str] = None,
) -> Tuple[any, int]:
    """
    Process an audio sample for deepfake detection benchmark.

    Uses TorchCodec AudioDecoder which accepts bytes directly (no temp files),
    handles resampling to target_sr and mono conversion internally.

    Preprocessing Pipeline:
    1. Decode audio bytes with TorchCodec (resamples to 16kHz, converts to mono)
    2. Crop/pad to exactly 6 seconds (96,000 samples at 16kHz)
       - If longer: random crop (with seed) or center crop
       - If shorter: zero-pad on the right

    A 30-second per-sample timeout prevents ffmpeg deadlocks on malformed files.
    No normalization is applied -- TorchCodec returns float32 in [-1, 1].

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
        label = MEDIA_TYPE_TO_LABEL.get(media_type, 1)

        # Decode with timeout -- AudioDecoder handles resampling + mono conversion
        samples = _decode_audio_with_timeout(audio_bytes, target_sr)
        waveform = samples.data.squeeze(0)  # (1, num_samples) -> (num_samples,)

        # Crop/pad to target length (e.g. 96,000 samples = 6s at 16kHz)
        target_length = int(target_sr * target_duration_seconds)

        if waveform.shape[0] > target_length:
            if use_random_crop:
                if seed is not None:
                    torch.manual_seed(seed)
                max_start = waveform.shape[0] - target_length
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start_idx = (waveform.shape[0] - target_length) // 2
            waveform = waveform[start_idx:start_idx + target_length]
        elif waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform.float(), label

    except concurrent.futures.TimeoutError:
        logger.warning(
            f"Audio decode timed out after {AUDIO_DECODE_TIMEOUT}s, skipping sample"
        )
        return None, None
    except Exception as e:
        logger.warning(f"Failed to process audio sample: {e}")
        return None, None


_hf_cache_configured = False
