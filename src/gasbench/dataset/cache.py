"""Dataset caching utilities for benchmark datasets."""

import os
import json
import shutil
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from ..logger import get_logger
from .config import BenchmarkDatasetConfig

logger = get_logger(__name__)


def check_dataset_cache(dataset_config: BenchmarkDatasetConfig, base_dir: str = "/.cache/gasbench") -> Dict:
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
    sample_count: int
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
            video_bytes = sample.get("video_bytes")
            if video_bytes is None:
                return None

            source_name = str(sample.get("source_file", ""))
            ext = Path(source_name).suffix.lower() if source_name else ".mp4"
            if not ext or ext not in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}:
                ext = ".mp4"
            filename = f"vid_{sample_count:06d}{ext}"
            file_path = os.path.join(samples_dir, filename)

            with open(file_path, "wb") as f:
                f.write(video_bytes)
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
    sample_count: int
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
                "images_per_parquet": dataset_config.images_per_parquet,
                "videos_per_zip": dataset_config.videos_per_zip,
                "parquet_per_dataset": dataset_config.parquet_per_dataset,
                "zips_per_dataset": dataset_config.zips_per_dataset,
            },
        }

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
