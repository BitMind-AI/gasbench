"""Dataset caching utilities for benchmark datasets."""

import os
import json
import shutil
from collections import defaultdict
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import torch

from ..logger import get_logger
from .config import BenchmarkDatasetConfig

logger = get_logger(__name__)


def check_dataset_cache(
    dataset_config: BenchmarkDatasetConfig, base_dir: str = "/.cache/gasbench"
) -> Dict:
    """Check if a dataset is already cached locally."""
    try:
        dataset_dir = os.path.join(base_dir, "datasets", dataset_config.name)
        dataset_info_file = os.path.join(dataset_dir, "dataset_info.json")
        samples_dir = os.path.join(dataset_dir, "samples")
        metadata_file = os.path.join(dataset_dir, "sample_metadata.json")

        if (
            os.path.exists(dataset_info_file)
            and os.path.exists(samples_dir)
            and os.path.exists(metadata_file)
        ):
            try:
                with open(dataset_info_file, "r") as f:
                    dataset_info = json.load(f)

                sample_count = len(
                    [
                        f
                        for f in os.listdir(samples_dir)
                        if os.path.isfile(os.path.join(samples_dir, f))
                    ]
                )

                if sample_count > 0:
                    return {
                        "cached": True,
                        "sample_count": sample_count,
                    }
            except Exception:
                pass

        return {
            "cached": False,
            "sample_count": 0,
        }
    except Exception as e:
        logger.warning(f"Failed to check cache for {dataset_config.name}: {e}")
        return {"cached": False, "sample_count": 0}


def save_sample_to_cache(
    sample: dict,
    dataset_config: BenchmarkDatasetConfig,
    samples_dir: str,
    sample_count: int,
) -> Optional[str]:
    """Save individual sample to local cache in original format."""
    try:
        if dataset_config.modality == "image":
            # Save image in original format
            image = sample.get("image")
            if image is None:
                return None

            img_format = (
                image.format if hasattr(image, "format") and image.format else "JPEG"
            )
            ext = ".jpg" if img_format.upper() in ["JPEG"] else f".{img_format.lower()}"
            filename = f"img_{sample_count:06d}{ext}"
            file_path = os.path.join(samples_dir, filename)

            image.save(file_path)
            return filename

        elif dataset_config.modality == "video":
            # Check if sample has video_bytes (regular video) or video_frames (pre-extracted frames)
            video_bytes = sample.get("video_bytes")
            video_frames = sample.get("video_frames")

            if video_bytes:
                # Regular video file
                source_name = str(sample.get("source_file", ""))
                ext = Path(source_name).suffix.lower() if source_name else ".mp4"
                if not ext or ext not in {
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".wmv",
                    ".webm",
                    ".m4v",
                }:
                    ext = ".mp4"
                filename = f"vid_{sample_count:06d}{ext}"
                file_path = os.path.join(samples_dir, filename)

                with open(file_path, "wb") as f:
                    f.write(video_bytes)
                return filename

            elif video_frames:
                # Frame directory - save as a directory
                source_name = str(sample.get("source_file", "frame_dir"))
                dirname = f"vid_{sample_count:06d}_frames"
                dir_path = os.path.join(samples_dir, dirname)
                os.makedirs(dir_path, exist_ok=True)

                # Copy/link frames to cache directory
                import shutil

                for i, frame_path in enumerate(video_frames):
                    ext = Path(frame_path).suffix
                    dest_path = os.path.join(dir_path, f"{i:06d}{ext}")
                    shutil.copy2(frame_path, dest_path)

                return dirname

            else:
                # No video data found
                return None

        elif dataset_config.modality == "audio":
            audio_bytes = sample.get("audio_bytes")
            if audio_bytes is None:
                return None

            source_name = str(sample.get("source_file", ""))
            ext = Path(source_name).suffix.lower() if source_name else ".wav"
            if not ext or ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
                ext = ".wav"
            filename = f"aud_{sample_count:06d}{ext}"
            file_path = os.path.join(samples_dir, filename)

            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            return filename

    except Exception as e:
        logger.warning(
            f"Failed to save sample {sample_count} for {dataset_config.name}: {e}"
        )
        return None


def save_dataset_cache_files(
    dataset_config: BenchmarkDatasetConfig,
    dataset_cache_dir: str,
    dataset_samples: dict,
    sample_count: int,
    dataset_info_extras: Optional[Dict] = None,
):
    """Save dataset info and metadata files to local cache."""
    try:
        dataset_info = {
            "name": dataset_config.name,
            "path": dataset_config.path,
            "modality": dataset_config.modality,
            "media_type": dataset_config.media_type,
            "source_format": getattr(dataset_config, "source_format", ""),
            "sample_count": sample_count,
            "cached_at": datetime.now().isoformat(),
            "config": {
                "media_per_archive": dataset_config.media_per_archive,
                "archives_per_dataset": dataset_config.archives_per_dataset,
            },
        }
        if dataset_info_extras:
            try:
                for k, v in dataset_info_extras.items():
                    dataset_info[k] = v
            except Exception:
                pass

        dataset_info_file = os.path.join(dataset_cache_dir, "dataset_info.json")
        with open(dataset_info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        metadata_file = os.path.join(dataset_cache_dir, "sample_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(dataset_samples, f, indent=2)

        logger.debug(f"💾 Saved dataset cache files for {dataset_config.name}")

    except Exception as e:
        logger.error(
            f"Failed to save dataset cache files for {dataset_config.name}: {e}"
        )
        raise


def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory after dataset processing."""
    if os.path.exists(temp_dir):
        try:
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    logger.debug(f"🧹 Cleaned up temp file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logger.debug(f"🧹 Cleaned up temp directory: {item}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean temp files: {cleanup_error}")


def cleanup_temp_directory_full(temp_dir: str):
    """Completely remove temporary directory."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info("🧹 Cleaned up consolidated temporary directory")
        except Exception as e:
            logger.warning(f"Failed to clean temp directory: {e}")


def save_preprocessed_audio_tensor(
    waveform: torch.Tensor,
    label: int,
    metadata: dict,
    samples_dir: str,
    sample_count: int
) -> Optional[str]:
    """
    Save preprocessed audio tensor to cache as .pt file.
    
    This is much faster than saving raw audio bytes and re-processing every time.
    
    Args:
        waveform: Preprocessed audio tensor (1, 96000)
        label: Audio label (0=real, 1=synthetic)
        metadata: Additional metadata to save
        samples_dir: Directory to save the tensor
        sample_count: Sample index for naming
        
    Returns:
        Filename if successful, None otherwise
    """
    try:
        filename = f"aud_{sample_count:06d}.pt"
        file_path = os.path.join(samples_dir, filename)
        
        # Save tensor with label and metadata
        torch.save({
            'waveform': waveform.cpu(),  # Ensure it's on CPU for storage
            'label': label,
            'metadata': metadata,
            'shape': waveform.shape,
            'dtype': str(waveform.dtype)
        }, file_path)
        
        return filename
    except Exception as e:
        logger.warning(f"Failed to save preprocessed audio tensor {sample_count}: {e}")
        return None


def load_preprocessed_audio_tensor(file_path: str) -> Optional[Dict]:
    """
    Load preprocessed audio tensor from cache.
    
    Args:
        file_path: Path to the .pt file
        
    Returns:
        Dictionary with 'waveform', 'label', and 'metadata' if successful, None otherwise
    """
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        logger.warning(f"Failed to load preprocessed audio tensor from {file_path}: {e}")
        return None


def check_preprocessed_cache_exists(dataset_config: BenchmarkDatasetConfig, base_dir: str = "/.cache/gasbench") -> bool:
    """
    Check if preprocessed audio tensors exist in cache.
    
    Args:
        dataset_config: Dataset configuration
        base_dir: Base cache directory
        
    Returns:
        True if preprocessed cache exists and has .pt files, False otherwise
    """
    try:
        if dataset_config.modality != "audio":
            return False
            
        dataset_dir = os.path.join(base_dir, "datasets", dataset_config.name)
        samples_dir = os.path.join(dataset_dir, "samples")
        
        if not os.path.exists(samples_dir):
            return False
        
        # Check if there are any .pt files
        pt_files = [f for f in os.listdir(samples_dir) if f.endswith('.pt')]
        return len(pt_files) > 0
        
    except Exception as e:
        logger.debug(f"Failed to check preprocessed cache for {dataset_config.name}: {e}")
        return False


def format_size_bytes(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def scan_cache_directory(cache_dir: str = "/.cache/gasbench") -> List[Dict]:
    """Scan cache directory and return list of dataset information.

    Returns:
        List of dicts containing dataset metadata including name, modality,
        media_type, sample_count, size_bytes, etc.
    """
    datasets_dir = Path(cache_dir) / "datasets"

    if not datasets_dir.exists():
        return []

    datasets = []

    for dataset_dir in sorted(datasets_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        info_file = dataset_dir / "dataset_info.json"
        samples_dir = dataset_dir / "samples"

        if not info_file.exists():
            continue

        try:
            with open(info_file) as f:
                info = json.load(f)

            sample_count = info.get("sample_count", 0)

            size_bytes = 0
            if samples_dir.exists():
                for item in samples_dir.rglob("*"):
                    if item.is_file():
                        size_bytes += item.stat().st_size

            datasets.append(
                {
                    "name": info.get("name", dataset_dir.name),
                    "modality": info.get("modality", "unknown"),
                    "media_type": info.get("media_type", "unknown"),
                    "source_format": info.get("source_format", "unknown"),
                    "sample_count": sample_count,
                    "size_bytes": size_bytes,
                    "cached_at": info.get("cached_at", "unknown"),
                }
            )

        except Exception as e:
            logger.warning(f"Error reading {dataset_dir.name}: {e}")
            continue

    return datasets


def compute_cache_statistics(datasets: List[Dict]) -> Tuple[Dict, Dict]:
    """Compute statistics from cached datasets.

    Returns:
        Tuple of (by_modality, by_media_type) dictionaries with counts, samples, and sizes
    """
    by_modality = defaultdict(lambda: {"count": 0, "samples": 0, "size": 0})
    by_media_type = defaultdict(lambda: {"count": 0, "samples": 0, "size": 0})

    for ds in datasets:
        mod = ds["modality"]
        mtype = ds["media_type"]

        by_modality[mod]["count"] += 1
        by_modality[mod]["samples"] += ds["sample_count"]
        by_modality[mod]["size"] += ds["size_bytes"]

        by_media_type[mtype]["count"] += 1
        by_media_type[mtype]["samples"] += ds["sample_count"]
        by_media_type[mtype]["size"] += ds["size_bytes"]

    return by_modality, by_media_type


def verify_cache_against_configs(
    cached_names: set,
    cached_datasets: List[Dict],
    dataset_config: Optional[str] = None,
    holdout_config: Optional[str] = None,
    cache_dir: str = "/.cache/gasbench",
) -> Tuple[set, set, set, List, set]:
    """Verify cache completeness against config files.

    Args:
        cached_names: Set of dataset names present in cache
        cached_datasets: List of dataset dicts from scan_cache_directory
        dataset_config: Path to dataset config YAML
        holdout_config: Path to holdout config YAML
        cache_dir: Base cache directory

    Returns:
        Tuple of (present_names, missing_names, extra_names, expected_datasets, config_modalities)
    """
    from .config import load_datasets_from_yaml, load_holdout_datasets_from_yaml

    expected_datasets = []
    config_modalities = set()

    if dataset_config:
        datasets_dict = load_datasets_from_yaml(yaml_path=dataset_config)
        for modality in ["image", "video", "audio"]:
            if modality in datasets_dict and datasets_dict[modality]:
                config_modalities.add(modality)
                expected_datasets.extend(datasets_dict[modality])

    if holdout_config:
        datasets_dict = load_holdout_datasets_from_yaml(
            yaml_path=holdout_config, cache_dir=cache_dir
        )
        for modality in ["image", "video", "audio"]:
            if modality in datasets_dict and datasets_dict[modality]:
                config_modalities.add(modality)
                expected_datasets.extend(datasets_dict.get(modality, []))

    expected_names = {ds.name for ds in expected_datasets}
    present = expected_names & cached_names
    missing = expected_names - cached_names

    cached_in_modalities = {
        ds["name"] for ds in cached_datasets
        if ds.get("modality", "").lower() in config_modalities
    }
    extra = cached_in_modalities - expected_names

    return present, missing, extra, expected_datasets, config_modalities
