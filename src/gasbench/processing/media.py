import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
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
    """Process a video sample that contains raw video bytes. Returns uint8 data for ONNX models."""
    try:
        video_bytes = sample.get("video_bytes")
        if not video_bytes:
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = MEDIA_TYPE_TO_LABEL[media_type]

        src_name = str(sample.get("source_file", ""))
        ext = Path(src_name).suffix.lower() if src_name else ".mp4"
        if ext not in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"):
            ext = ".mp4"
        
        temp_video = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_video_path = temp_video.name
        
        try:
            temp_video.write(video_bytes)
            temp_video.flush()
            temp_video.close()

            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                logger.warning(f"Failed to open temporary video: {temp_video_path}")
                return None, None
                
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                max_frames = 16
                frames = []
                frames_read = 0
                
                while frames_read < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    frames_read += 1
                
                if frames_read == 0:
                    logger.warning(f"No frames extracted from video")
                    return None, None
                
                if frames_read < max_frames:
                    last_frame = frames[-1]
                    for i in range(frames_read, max_frames):
                        frames.append(last_frame)
                
                video_array = np.array(frames, dtype=np.uint8)  # THWC uint8
                
                return video_array, label
                
            finally:
                cap.release()
                
        finally:
            try:
                os.unlink(temp_video_path)
            except:
                pass

    except Exception as e:
        logger.warning(f"Failed to process video bytes sample: {e}")
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


def process_audio_sample(sample: Dict, target_sr: int = 16000) -> Tuple[any, int]:
    """Process an audio sample (bytes) for classification evaluation.
    
    Args:
        sample: Dictionary containing 'audio_bytes' and metadata
        target_sr: Target sample rate to resample to (default 16000 Hz)
        
    Returns:
        Tuple of (audio_tensor, label)
        - audio_tensor: torch.Tensor of shape (channels, time)
        - label: int (0 for real, 1 for synthetic)
    """
    try:
        audio_bytes = sample.get("audio_bytes")
        if audio_bytes is None:
            logger.warning("No audio bytes in sample")
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = MEDIA_TYPE_TO_LABEL.get(media_type, 1)  # Default to synthetic/1 if unknown

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

            # Load audio from temporary file
            waveform, sample_rate = torchaudio.load(temp_audio_path)

            # Resample if necessary
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                waveform = resampler(waveform)
            
            # Trim or pad to fixed length (3 seconds at target_sr)
            target_length = target_sr * 3  # 3 seconds
            current_length = waveform.shape[1]
            
            if current_length > target_length:
                # Trim to first 3 seconds
                waveform = waveform[:, :target_length]
            elif current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))

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
