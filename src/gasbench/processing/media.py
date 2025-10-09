import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2

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
                
                video_array = np.array(frames, dtype=np.uint8)
                video_array = np.transpose(video_array, (0, 3, 1, 2))
                video_array = np.expand_dims(video_array, axis=0)
                
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
    """Process an image sample for classification evaluation."""
    try:
        image_data = sample.get("image")
        if image_data is None:
            logger.warning("No image data in sample")
            return None, None

        media_type = sample.get("media_type", "synthetic")
        label = MEDIA_TYPE_TO_LABEL[media_type]

        if hasattr(image_data, "convert"):
            image = image_data
            logger.debug(f"PIL Image received: {type(image)} size={getattr(image, 'size', 'unknown')}")
        else:
            logger.error(f"Image data is not a PIL Image: {type(image_data)}")
            return None, None

        if image is None:
            return None, None

        image = image.convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, 0)

        return image_array, label

    except Exception as e:
        logger.warning(f"Failed to process image sample: {e}")
        return None, None


_hf_cache_configured = False
