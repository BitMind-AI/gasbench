import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from decord import VideoReader, cpu
from io import BytesIO
from PIL import Image
import torchaudio
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
    Decord outputs RGB directly (no BGR→RGB conversion needed).
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


def process_audio_sample(
    sample: Dict, 
    target_sr: int = 16000, 
    target_duration_seconds: float = 6.0,
    use_random_crop: bool = False,
    seed: Optional[int] = 42,
    device: Optional[str] = None
) -> Tuple[any, int]:
    """
    Process an audio sample for deepfake detection benchmark.
    
    Preprocessing Pipeline (in exact order):
    1. Load audio with torchaudio
    2. Convert to mono (average all channels)
    3. Resample to 16kHz if needed (GPU-accelerated if available)
    4. Crop/pad to exactly 6 seconds (96,000 samples at 16kHz)
       - If longer: random crop (with seed) or center crop
       - If shorter: zero-pad on the right
    
    No normalization is applied. torchaudio loads PCM as float32 in [-1, 1].
    This matches image/video preprocessing where raw uint8 values are passed.
    
    Args:
        sample: Dictionary containing 'audio_bytes' and metadata
        target_sr: Target sample rate (hardcoded to 16000 Hz)
        target_duration_seconds: Target duration in seconds (hardcoded to 6.0)
        use_random_crop: If True, randomly crop; if False, center crop (default: False for reproducibility)
        seed: Random seed for deterministic cropping during validation (default: 42)
        device: Device for processing ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        Tuple of (waveform, label)
        - waveform: torch.Tensor of shape (1, 96000) as float32 on CPU
        - label: int (0 for real, 1 for synthetic)
    """
    try:
        audio_bytes = sample.get("audio_bytes")
        if audio_bytes is None:
            logger.warning("No audio bytes in sample")
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = MEDIA_TYPE_TO_LABEL.get(media_type, 1)  # Default to synthetic/1 if unknown

        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Determine format from source_file or cached_filename if available
        source_file = sample.get("cached_filename") or sample.get("source_file", "")
        audio_format = ".wav"  # default
        if source_file:
            # Extract extension from filename
            ext = Path(source_file).suffix.lower()
            if ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]:
                audio_format = ext

        # torchaudio requires a file path, so write to a temporary file
        temp_audio = tempfile.NamedTemporaryFile(suffix=audio_format, delete=False)
        temp_audio_path = temp_audio.name
        
        try:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            temp_audio.close()

            # Step 1: Load audio from temporary file
            waveform, sample_rate = torchaudio.load(temp_audio_path)
            
            # Move to GPU if available for faster processing
            if device == 'cuda' and torch.cuda.is_available():
                waveform = waveform.cuda()

            # Step 2: Convert to mono (average all channels)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Step 3: Resample to 16kHz if necessary (GPU-accelerated)
            if sample_rate != target_sr:
                if device == 'cuda' and torch.cuda.is_available():
                    # GPU-accelerated resampling
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=target_sr
                    ).to(device)
                else:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=target_sr
                    )
                waveform = resampler(waveform)
            
            # Step 4: Length handling - exactly 6 seconds (96,000 samples at 16kHz)
            target_length = int(target_sr * target_duration_seconds)  # 96,000 samples
            current_length = waveform.shape[1]
            
            if current_length > target_length:
                # Audio is longer than target - crop it
                if use_random_crop:
                    # Random crop with seed for reproducibility
                    if seed is not None:
                        torch.manual_seed(seed)
                    max_start = current_length - target_length
                    start_idx = torch.randint(0, max_start + 1, (1,)).item()
                    waveform = waveform[:, start_idx:start_idx + target_length]
                else:
                    # Center crop (deterministic, default for benchmarking)
                    start_idx = (current_length - target_length) // 2
                    waveform = waveform[:, start_idx:start_idx + target_length]
            elif current_length < target_length:
                # Audio is shorter than target - zero-pad on the right
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Ensure output is float32 and shape (1, 96000)
            # No normalization - torchaudio already loads as float32 in [-1, 1]
            # This matches image/video approach where we pass raw values (uint8)
            waveform = waveform.float()
            
            # Move back to CPU for consistency with existing code
            if device == 'cuda' and waveform.is_cuda:
                waveform = waveform.cpu()
            
            assert waveform.shape == (1, target_length), f"Expected shape (1, {target_length}), got {waveform.shape}"
            
            # Squeeze channel dimension for compatibility with most audio models
            # Output: (96000,) instead of (1, 96000)
            waveform = waveform.squeeze(0)
            
            return waveform, label
            
        finally:
            try:
                os.unlink(temp_audio_path)
            except:
                pass

    except Exception as e:
        logger.warning(f"Failed to process audio sample: {e}")
        return None, None


_hf_cache_configured = False
