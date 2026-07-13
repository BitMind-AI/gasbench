"""Load previously-cached dataset samples from the persistent cache volume."""

import json
import os
from pathlib import Path

from PIL import Image

from ...logger import get_logger

from .constants import IMAGE_FILE_EXTENSIONS

logger = get_logger(__name__)


def _is_dataset_cached(dataset, cache_dir: str = "/.cache/gasbench") -> bool:
    """
    Check if a specific dataset is already cached locally.

    Returns True if the dataset appears to be cached and has samples available.
    """
    try:
        dataset_dir = f"{cache_dir}/datasets/{dataset.name}"

        dataset_info_file = os.path.join(dataset_dir, "dataset_info.json")
        samples_dir = os.path.join(dataset_dir, "samples")
        metadata_file = os.path.join(dataset_dir, "sample_metadata.json")

        # Check if all required files exist and samples directory has content
        return (
            os.path.exists(dataset_info_file)
            and os.path.exists(samples_dir)
            and os.path.exists(metadata_file)
            and len(os.listdir(samples_dir)) > 0
        )
    except (OSError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Error checking cache for dataset {dataset.name}: {e}")
        return False



def _load_dataset_from_cache(dataset, cache_dir: str = "/.cache/gasbench"):
    """
    Load samples from a cached dataset locally.

    Yields samples in the same format as download_and_extract.
    """

    dataset_dir = f"{cache_dir}/datasets/{dataset.name}"
    samples_dir = os.path.join(dataset_dir, "samples")
    metadata_file = os.path.join(dataset_dir, "sample_metadata.json")

    # Load sample metadata
    with open(metadata_file, "r") as f:
        sample_metadata = json.load(f)

    sample_files = [f for f in os.listdir(samples_dir) if not f.startswith(".")]

    logger.info(f"Loading {len(sample_files)} cached samples for {dataset.name}")

    for filename in sample_files:
        try:
            metadata = sample_metadata.get(filename, {})
            filepath = os.path.join(samples_dir, filename)

            if dataset.modality == "image":
                # Load image file
                img = Image.open(filepath)
                sample = {
                    "image": img,
                    "media_type": dataset.media_type,
                    "dataset_name": dataset.name,
                    "dataset_path": dataset.path,
                    "source_file": f"cached_{filename}",
                    **metadata,
                }
                yield sample

            elif dataset.modality == "audio":
                # Load audio file as bytes
                with open(filepath, "rb") as f:
                    audio_bytes = f.read()
                sample = {
                    "audio_bytes": audio_bytes,
                    "media_type": dataset.media_type,
                    "dataset_name": dataset.name,
                    "dataset_path": dataset.path,
                    "source_file": f"cached_{filename}",
                    **metadata,  # Include all original metadata
                }
                yield sample

            elif dataset.modality == "video":
                # Check if it's a frame directory or a video file
                if os.path.isdir(filepath):
                    # Frame directory - load all frames
                    frame_files = []
                    for ext in IMAGE_FILE_EXTENSIONS:
                        frame_files.extend(sorted(Path(filepath).glob(f"*{ext}")))
                        frame_files.extend(
                            sorted(Path(filepath).glob(f"*{ext.upper()}"))
                        )

                    frame_files = sorted(set(frame_files), key=lambda p: p.name)

                    sample = {
                        "video_frames": frame_files,
                        "media_type": dataset.media_type,
                        "dataset_name": dataset.name,
                        "dataset_path": dataset.path,
                        "source_file": f"cached_{filename}",
                        **metadata,
                    }
                    yield sample
                else:
                    # Regular video file
                    with open(filepath, "rb") as f:
                        video_bytes = f.read()

                    sample = {
                        "video_bytes": video_bytes,
                        "media_type": dataset.media_type,
                        "dataset_name": dataset.name,
                        "dataset_path": dataset.path,
                        "source_file": f"cached_{filename}",
                        **metadata,
                    }
                    yield sample

        except (OSError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(
                f"Failed to load cached sample {filename} from {dataset.name}: {e}"
            )
            continue

