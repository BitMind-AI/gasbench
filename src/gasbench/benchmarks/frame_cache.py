"""Shared decoded-video frames.

The cache stores the first real frames of a clip, in decode order, plus the
source frame count and fps. Each model still chooses its own frame count,
frame rate, and resize. A cached tensor is only reused when every requested
index is present. Resized and augmented inputs are never stored here.
"""

import os
from typing import Optional, Tuple

import numpy as np

from ..constants import MAX_VIDEO_NUM_FRAMES

_VERSION = "frames_v1"


def frame_cache_path(cache_dir: str, sample_id: str) -> str:
    return os.path.join(cache_dir, sample_id[:2], f"{sample_id}_{_VERSION}.npz")


def frame_indices(total_frames: int, fps: Optional[float], num_frames: int, frame_rate: Optional[float]) -> list:
    """Match process_video_bytes_sample's index selection, including the 30fps fallback."""
    if total_frames <= 0 or num_frames <= 0:
        return []
    if frame_rate is not None:
        video_fps = fps if fps and fps > 0 else 30.0
        frame_step = max(1, round(video_fps / frame_rate))
        return list(range(0, total_frames, frame_step))[:num_frames]
    return list(range(min(num_frames, total_frames)))


def select_cached_frames(
    stored: np.ndarray,
    total_frames: int,
    fps: Optional[float],
    num_frames: int,
    frame_rate: Optional[float],
) -> Optional[np.ndarray]:
    """Return a (num_frames, H, W, 3) array, or None when the cache cannot serve this model."""
    if stored.ndim != 4 or stored.shape[0] == 0:
        return None
    indices = frame_indices(total_frames, fps, num_frames, frame_rate)
    if not indices or max(indices) >= stored.shape[0]:
        return None
    frames = [stored[i] for i in indices]
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return np.stack(frames, axis=0)


def load_frame_cache(cache_dir: str, sample_id: str) -> Optional[Tuple[np.ndarray, int, Optional[float]]]:
    path = frame_cache_path(cache_dir, sample_id)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
            total_frames = int(data["total_frames"])
            fps_value = float(data["fps"])
        if frames.dtype != np.uint8 or frames.ndim != 4:
            return None
        fps = None if np.isnan(fps_value) else fps_value
        return frames, total_frames, fps
    except Exception:
        return None


def cache_covers_prefix(cache_dir: str, sample_id: str, max_frames: int = MAX_VIDEO_NUM_FRAMES) -> bool:
    loaded = load_frame_cache(cache_dir, sample_id)
    if loaded is None:
        return False
    frames, total_frames, _fps = loaded
    return frames.shape[0] >= min(max_frames, total_frames)


def write_frame_cache(
    cache_dir: str,
    sample_id: str,
    frames: np.ndarray,
    total_frames: int,
    fps: Optional[float],
) -> None:
    """Atomically store a sequential prefix. frames[i] is source frame i."""
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[0] == 0:
        raise ValueError("frame cache expects a non-empty uint8 (N, H, W, 3) array")
    path = frame_cache_path(cache_dir, sample_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    fps_value = np.float64(np.nan if fps is None or fps <= 0 else fps)
    try:
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                frames=np.ascontiguousarray(frames),
                total_frames=np.int64(total_frames),
                fps=fps_value,
            )
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def predecode_video_frame_cache(
    *,
    cache_dir: str,
    frame_cache_dir: str,
    holdout_config: Optional[str] = None,
    dataset_config: Optional[str] = None,
    seed: int,
    mode: str = "full",
    commit=None,
    commit_every: int = 25,
) -> dict:
    """Decode each planned video sample once and store its leading frames.

    Samples already covered up to MAX_VIDEO_NUM_FRAMES are skipped, so the job
    can be run again after a preemption. commit() is invoked periodically so a
    volume mount keeps the frames that finished.
    """
    from ..dataset.iterator import DatasetIterator
    from ..logger import get_logger
    from .common import BenchmarkRunConfig, build_plan
    from .recording import build_sample_id
    from ..processing.media import decode_video_prefix

    logger = get_logger(__name__)
    config = BenchmarkRunConfig(
        modality="video",
        mode=mode,
        gasstation_only=False,
        dataset_config_path=dataset_config,
        holdout_config_path=holdout_config,
        cache_dir=cache_dir,
        hf_token=None,
        batch_size=1,
        augment_level=0,
        crop_prob=0.0,
        records_parquet_path=None,
    )
    plan = build_plan(logger, config, input_specs=None)
    if not plan:
        return {"written": 0, "skipped": 0, "failed": 0, "datasets": 0}

    written = skipped = failed = 0
    since_commit = 0
    for dataset in plan.available_datasets:
        cap = plan.sampling_plan.get(dataset.name, 0)
        if cap <= 0:
            continue
        iterator = DatasetIterator(
            dataset,
            max_samples=cap,
            cache_dir=cache_dir,
            download=False,
            seed=seed,
            lazy_read=True,
        )
        for sample in iterator:
            sample_id = build_sample_id(sample)
            if cache_covers_prefix(frame_cache_dir, sample_id):
                skipped += 1
                continue
            video_path = sample.get("video_path")
            try:
                if video_path:
                    with open(video_path, "rb") as handle:
                        video_bytes = handle.read()
                else:
                    video_bytes = sample.get("video_bytes")
                decoded = decode_video_prefix(
                    video_bytes, sample.get("source_file") or "", MAX_VIDEO_NUM_FRAMES
                )
            except Exception as exc:
                logger.warning("Frame cache decode failed for %s: %s", dataset.name, exc)
                decoded = None
            if decoded is None:
                failed += 1
                continue
            frames, total_frames, fps = decoded
            try:
                write_frame_cache(frame_cache_dir, sample_id, frames, total_frames, fps)
            except Exception as exc:
                logger.warning("Frame cache write failed for %s: %s", sample_id, exc)
                failed += 1
                continue
            written += 1
            since_commit += 1
            if commit is not None and since_commit >= commit_every:
                commit()
                since_commit = 0
    if commit is not None and since_commit:
        commit()
    logger.info(
        "Video frame cache: wrote %s, skipped %s, failed %s", written, skipped, failed
    )
    return {
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "datasets": len(plan.available_datasets),
    }
