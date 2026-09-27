"""Dataset caching utilities for benchmark datasets."""

import os
import json
import re
from collections import defaultdict
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from ..logger import get_logger
from .config import BenchmarkDatasetConfig

logger = get_logger(__name__)


CACHE_MAX_SAMPLES = 500


def load_partial_cache(directory):
    """Read intact indexed samples and reserve all previously used file indices."""
    directory = Path(directory)
    samples = directory / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    index = directory / "sample_metadata.json"
    metadata = json.loads(index.read_text()) if index.exists() else {}
    metadata = {
        name: value for name, value in metadata.items() if (samples / name).exists()
    }
    indices = [
        int(match.group(1))
        for path in samples.iterdir()
        if (match := re.search(r"_(\d+)", path.name))
    ]
    return metadata, max(indices, default=-1) + 1


def cache_state(directory):
    """One cache completeness policy for discovery, download and iteration.

    Metadata is the committed index. Unindexed files can be remnants of an
    interrupted write and must not become additional benchmark samples.
    """
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "sample_metadata.json").read_text())
        json.loads((directory / "dataset_info.json").read_text())
        if not isinstance(metadata, dict):
            raise ValueError("Invalid sample metadata")
        available = {
            p.name
            for p in (directory / "samples").iterdir()
            if not p.name.startswith(".")
        }
        count = len(available.intersection(metadata))
        intact = count == len(metadata)
        complete = bool(
            count
            and intact
            and (
                (directory / ".download_complete").is_file()
                or count >= CACHE_MAX_SAMPLES
            )
        )
        return {"cached": count > 0, "sample_count": count, "complete": complete}
    except (OSError, ValueError, TypeError):
        return {"cached": False, "sample_count": 0, "complete": False}


def check_dataset_cache(dataset_config, base_dir="/.cache/gasbench"):
    state = cache_state(Path(base_dir) / "datasets" / dataset_config.name)
    return {key: state[key] for key in ("cached", "sample_count")}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    """Publish a complete JSON index and sync its containing directory."""
    import tempfile

    path = Path(path)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".pending-", delete=False
        ) as stream:
            pending = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        fsync_directory(path.parent)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def load_audio_sample(sample):
    """Read raw or preprocessed audio, retaining the selected source identity."""
    from ..constants import media_type_to_label

    path = Path(sample["audio_path"])
    if path.suffix == ".pt":
        import torch

        data = torch.load(path, map_location="cpu", weights_only=True)
        return {
            **data.get("metadata", {}),
            **sample,
            "preprocessed_waveform": data["waveform"],
            "label": media_type_to_label(sample["media_type"], "audio"),
            "cached_filename": path.name,
            "is_preprocessed": True,
        }
    return {
        **sample,
        "audio_bytes": path.read_bytes(),
        "cached_filename": path.name,
        "is_preprocessed": False,
    }


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
                    ".mpeg",
                    ".mpg",
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

                for i, frame in enumerate(video_frames):
                    if isinstance(frame, (bytes, bytearray)):
                        # Frames-parquet samples carry raw bytes, not paths
                        data = bytes(frame)
                        ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                        with open(os.path.join(dir_path, f"{i:06d}{ext}"), "wb") as fh:
                            fh.write(data)
                    else:
                        ext = Path(frame).suffix
                        dest_path = os.path.join(dir_path, f"{i:06d}{ext}")
                        shutil.copy2(frame, dest_path)

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
            dataset_info.update(dataset_info_extras)

        dataset_info_file = os.path.join(dataset_cache_dir, "dataset_info.json")
        atomic_json(dataset_info_file, dataset_info)

        metadata_file = os.path.join(dataset_cache_dir, "sample_metadata.json")
        atomic_json(metadata_file, dataset_samples)

        logger.debug(f"💾 Saved dataset cache files for {dataset_config.name}")

    except Exception as e:
        logger.error(
            f"Failed to save dataset cache files for {dataset_config.name}: {e}"
        )
        raise


def format_size_bytes(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def scan_cache_directory(cache_dir: str = "/.cache/gasbench") -> List[Dict]:
    """Scan cache directory and return list of dataset information.

    Accepts either:
    - A base cache root (e.g. /workspace/modal-image-datasets): looks in
      cache_dir/datasets/ for dataset subdirs.
    - The datasets folder itself (e.g. /workspace/modal-image-datasets/datasets):
      looks directly in cache_dir for dataset subdirs.

    Returns:
        List of dicts containing dataset metadata including name, modality,
        media_type, sample_count, size_bytes, etc.
    """
    root = Path(cache_dir)
    candidates = [root / "datasets", root]

    datasets_dir = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if this looks like the datasets container (has subdirs with dataset_info.json)
            for sub in candidate.iterdir():
                if sub.is_dir() and (sub / "dataset_info.json").exists():
                    datasets_dir = candidate
                    break
            if datasets_dir is not None:
                break

    if datasets_dir is None:
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
        ds["name"]
        for ds in cached_datasets
        if ds.get("modality", "").lower() in config_modalities
    }
    extra = cached_in_modalities - expected_names

    return present, missing, extra, expected_datasets, config_modalities
