import os
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch

from ..logger import get_logger
from ..constants import MEDIA_TYPE_TO_LABEL

# decord is the primary video decoder (fast, frame-accurate random access).
# It has no macOS ARM wheel, so on Darwin we fall back to OpenCV —
# suitable for local development / smoke-testing but NOT for benchmark
# runs that need to be comparable to Linux results.
try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False
    logger = None  # placeholder, will be reassigned below

try:
    from torchcodec.decoders import AudioDecoder
except Exception:
    AudioDecoder = None

logger = get_logger(__name__)

if not _HAS_DECORD:
    logger.info(
        "decord not available — using OpenCV fallback "
        "(fine for dev/testing, not recommended for benchmark runs)"
    )


class _VideoReader:
    """Video frame reader — decord (production) or cv2 (macOS dev fallback).

    decord is the preferred backend: fast random access and frame-accurate
    seeking.  cv2 is only used when decord is unavailable (macOS) and
    falls back to sequential decode to guarantee identical frame selection.

    Both backends return RGB uint8 numpy arrays of shape (H, W, C).
    """

    def __init__(self, path: str):
        if _HAS_DECORD:
            self._vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            self._cap = None
        else:
            self._vr = None
            self._cap = cv2.VideoCapture(path)

    @property
    def total_frames(self) -> int:
        if self._vr is not None:
            return len(self._vr)
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        """Average FPS, or None if unavailable."""
        if self._vr is not None:
            return self._vr.get_avg_fps()
        f = self._cap.get(cv2.CAP_PROP_FPS)
        return f if f > 0 else None

    def read_frames(self, indices):
        """Read frames at *indices* (sorted list of ints).

        Returns list of RGB uint8 (H, W, C) arrays.  Frames that fail
        to decode are silently skipped.
        """
        if self._vr is not None:
            # decord: true random access
            frames = []
            for i in indices:
                frame = self._vr[i].asnumpy()  # Already RGB
                if frame is None or frame.size == 0:
                    logger.warning(f"Skipping invalid frame at index {i}")
                    continue
                frames.append(frame)
            return frames
        else:
            # cv2: sequential scan — cv2 seeking is imprecise, so we
            # decode every frame and keep only the ones we want.  This
            # matches decord's frame selection exactly but is slower,
            # hence only used as a macOS dev fallback.
            frames = []
            idx_set = set(indices)
            max_idx = max(indices) if indices else -1
            for i in range(max_idx + 1):
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    break
                if i in idx_set:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame is None or frame.size == 0:
                        logger.warning(f"Skipping invalid frame at index {i}")
                        continue
                    frames.append(frame)
            return frames

    def close(self):
        if self._cap is not None:
            self._cap.release()
        # decord VideoReader needs no explicit cleanup


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
    """Process a video sample that contains raw video bytes.

    Uses decord (the preferred, production decoder) when available.
    On macOS where decord has no wheel, falls back to OpenCV — suitable
    for local development but benchmark comparisons should be run on
    Linux with decord.

    Both backends return uint8 RGB frames of shape (H, W, C).

    Args:
        sample: Dict containing 'video_bytes' and metadata.
        num_frames: Number of frames to extract (default 16).
        frame_rate: If set, sample frames at this fps from the video; otherwise
            take the first ``num_frames`` frames sequentially.
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

            vr = _VideoReader(temp_video_path)
            total_frames = vr.total_frames

            if total_frames == 0:
                logger.warning("No frames in video")
                return None, None

            if frame_rate is not None:
                video_fps = vr.fps
                if not video_fps or video_fps <= 0:
                    logger.warning("Video has no fps metadata, assuming 30fps")
                    video_fps = 30.0
                frame_step = max(1, round(video_fps / frame_rate))
                frame_indices = list(range(0, total_frames, frame_step))[:num_frames]
            else:
                frame_indices = list(range(min(num_frames, total_frames)))

            frames = vr.read_frames(frame_indices)

            if len(frames) == 0:
                logger.warning("No frames extracted from video")
                return None, None

            if len(frames) < num_frames:
                last_frame = frames[-1]
                for _ in range(len(frames), num_frames):
                    frames.append(last_frame)

            video_array = np.array(frames, dtype=np.uint8)  # THWC uint8 RGB

            return video_array, label

        finally:
            try:
                os.unlink(temp_video_path)
            except:
                pass
            try:
                vr.close()
            except:
                pass

    except Exception as e:
        logger.warning(f"Failed to process video bytes sample: {e}")
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
        label = MEDIA_TYPE_TO_LABEL[media_type]

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
        label = MEDIA_TYPE_TO_LABEL[media_type]

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        image.close()

        return image_array, label

    except Exception as e:
        logger.warning(f"Failed to process image sample: {e}")
        return None, None


AUDIO_DECODE_TIMEOUT = 30  # seconds -- prevents ffmpeg deadlocks on malformed files


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


def _decode_audio_with_timeout(audio_bytes: bytes, target_sr: int, timeout: int = AUDIO_DECODE_TIMEOUT):
    """Decode audio bytes with a timeout.

    Tries TorchCodec first (in-process, resamples to ``target_sr``, mono). If TorchCodec
    cannot load or fails, falls back to the ``ffmpeg`` CLI (same sample rate / channel layout).

    A thread pool enforces ``timeout`` so a stuck decoder cannot block indefinitely.

    Args:
        audio_bytes: Raw audio file bytes
        target_sr: Target sample rate
        timeout: Maximum seconds to wait for decode

    Returns:
        ``torch.Tensor`` of shape ``(num_samples,)``, float32 in roughly [-1, 1]

    Raises:
        concurrent.futures.TimeoutError: If decode hangs beyond timeout
    """
    def _decode() -> torch.Tensor:
        try:
            if AudioDecoder is None:
                raise RuntimeError("torchcodec AudioDecoder not importable")
            decoder = AudioDecoder(audio_bytes, sample_rate=target_sr, num_channels=1)
            samples = decoder.get_all_samples()
            return samples.data.squeeze(0)
        except Exception as e:
            logger.debug(
                "TorchCodec audio decode unavailable (%s); using ffmpeg CLI fallback",
                e,
            )
            return _decode_audio_waveform_ffmpeg_cli(audio_bytes, target_sr)

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
        label = MEDIA_TYPE_TO_LABEL.get(media_type, 1)

        # Decode with timeout (TorchCodec or ffmpeg CLI)
        waveform = _decode_audio_with_timeout(audio_bytes, target_sr)

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
