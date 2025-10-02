"""Benchmark dataset configuration and constants."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkDatasetConfig:
    """Configuration for a benchmark dataset using unified download approach."""

    name: str
    path: str
    modality: str    # "image" or "video"
    media_type: str  # "real", "synthetic", or "semisynthetic"

    # Download parameters
    images_per_parquet: int = 100
    videos_per_zip: int = 50
    parquet_per_dataset: int = 5
    zips_per_dataset: int = 2
    source_format: str = ""  # Auto-detected if empty


# Benchmark sample sizes - total samples across all datasets
IMAGE_BENCHMARK_SIZE = 10000   # Total image samples across all image datasets  
VIDEO_BENCHMARK_SIZE = 5000   # Total video samples across all video datasets



BENCHMARK_DATASETS = {
    "image": [
        # Gasstation dataset
        BenchmarkDatasetConfig(
            name="gasstation-generated-images",
            path="gasstation/generated-images",
            modality="image",
            media_type="synthetic",
            images_per_parquet=-1,
            parquet_per_dataset=-1,
        ),
        # Real image datasets
        BenchmarkDatasetConfig(
            name="megalith-10m",
            path="drawthingsai/megalith-10m",
            modality="image",
            media_type="real",
            source_format="tar",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="eidon-image",
            path="bitmind/bm-eidon-image",
            modality="image",
            media_type="real",
            images_per_parquet=-1,
            parquet_per_dataset=-1,
        ),
        BenchmarkDatasetConfig(
            name="celeb-a-hq",
            path="bitmind/celeb-a-hq",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="ffhq-256",
            path="bitmind/ffhq-256",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="ms-coco-unique",
            path="bitmind/MS-COCO-unique-256",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        # Synthetic image datasets for comprehensive AI detection
        BenchmarkDatasetConfig(
            name="journeydb",
            path="bitmind/JourneyDB",
            modality="image",
            media_type="synthetic",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="genimage-midjourney",
            path="bitmind/GenImage_MidJourney",
            modality="image",
            media_type="synthetic",
            images_per_parquet=100,
            parquet_per_dataset=10,
        ),
        BenchmarkDatasetConfig(
            name="aura-imagegen",
            path="bitmind/bm-aura-imagegen",
            modality="image",
            media_type="synthetic",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="imagine-grok",
            path="bitmind/bm-imagine",
            modality="image",
            media_type="synthetic",
            source_format="jpg",
            images_per_parquet=200,  # Direct image files
            parquet_per_dataset=5,
        ),
        # Semisynthetic datasets
        BenchmarkDatasetConfig(
            name="face-swap",
            path="bitmind/face-swap",
            modality="image",
            media_type="semisynthetic",
            images_per_parquet=333,
            parquet_per_dataset=3,
        ),
        # Additional diverse real datasets
        BenchmarkDatasetConfig(
            name="open-image-v7",
            path="bitmind/open-image-v7-256",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="afhq",
            path="bitmind/AFHQ",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="lfw",
            path="bitmind/lfw",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
        BenchmarkDatasetConfig(
            name="caltech-256",
            path="bitmind/caltech-256",
            modality="image",
            media_type="real",
            images_per_parquet=250,
            parquet_per_dataset=4,
        ),
        BenchmarkDatasetConfig(
            name="caltech-101",
            path="bitmind/caltech-101",
            modality="image",
            media_type="real",
            images_per_parquet=1000,
            parquet_per_dataset=1,
        ),
        BenchmarkDatasetConfig(
            name="dtd-textures",
            path="bitmind/dtd",
            modality="image",
            media_type="real",
            images_per_parquet=500,
            parquet_per_dataset=2,
        ),
        BenchmarkDatasetConfig(
            name="idoc-mugshots",
            path="bitmind/idoc-mugshots-images",
            modality="image",
            media_type="real",
            images_per_parquet=200,
            parquet_per_dataset=5,
        ),
    ],
    "video": [
        # Gasstation synthetic datasets (primary benchmark targets)
        BenchmarkDatasetConfig(
            name="gasstation-generated-videos",
            path="gasstation/generated-videos",
            modality="video",
            media_type="synthetic",
            source_format="zip",
            videos_per_zip=-1,
            zips_per_dataset=-1,
        ),
        # Real video datasets
        BenchmarkDatasetConfig(
            name="eidon-video",
            path="bitmind/bm-eidon-video",
            modality="video",
            media_type="real",
            source_format="zip",
            videos_per_zip=-1,
            zips_per_dataset=-1,
        ),
        BenchmarkDatasetConfig(
            name="imagenet-vidvrd",
            path="shangxd/imagenet-vidvrd",
            modality="video",
            media_type="real",
            source_format="zip",
            videos_per_zip=500,
            zips_per_dataset=2,
        ),
        BenchmarkDatasetConfig(
            name="pe-video",
            path="facebook/PE-Video",
            modality="video",
            media_type="real",
            source_format="tar",
            videos_per_zip=500,
            zips_per_dataset=2,
        ),
        # Synthetic video datasets
        BenchmarkDatasetConfig(
            name="veo3-preferences",
            path="Rapidata/text-2-video-human-preferences-veo3",
            modality="video",
            media_type="synthetic",
            source_format="mp4",
            videos_per_zip=-1,
            zips_per_dataset=-1,
        ),
        BenchmarkDatasetConfig(
            name="veo2-preferences",
            path="Rapidata/text-2-video-human-preferences-veo2",
            modality="video",
            media_type="synthetic",
            source_format="mp4",
            videos_per_zip=-1,  # Direct MP4 files
            zips_per_dataset=-1,
        ),
        BenchmarkDatasetConfig(
            name="aura-video",
            path="bitmind/aura-video",
            modality="video",
            media_type="synthetic",
            source_format="parquet",
            videos_per_zip=-1,
            zips_per_dataset=-1,
        ),
        BenchmarkDatasetConfig(
            name="aislop-videos",
            path="bitmind/aislop-videos",
            modality="video",
            media_type="synthetic",
            source_format="tar",
            videos_per_zip=200,
            zips_per_dataset=5,
        ),
        # Semisynthetic datasets
        BenchmarkDatasetConfig(
            name="semisynthetic-video",
            path="bitmind/semisynthetic-video",
            modality="video",
            media_type="semisynthetic",
            source_format="zip",
            videos_per_zip=100,
            zips_per_dataset=10,
        ),
    ]
}

DEBUG_BENCHMARK_DATASETS = {
    "image": [
        BenchmarkDatasetConfig(
            name="gasstation-generated-images",
            path="gasstation/generated-images",
            modality="image",
            media_type="synthetic",
            source_format="parquet",
            images_per_parquet=50,
            parquet_per_dataset=1,
        ),
        BenchmarkDatasetConfig(
            name="celeb-a-hq",
            path="bitmind/celeb-a-hq",
            modality="image",
            media_type="real",
            source_format="parquet",
            images_per_parquet=50,
            parquet_per_dataset=1,
        ),
        BenchmarkDatasetConfig(
            name="face-swap",
            path="bitmind/face-swap",
            modality="image",
            media_type="semisynthetic",
            source_format="parquet",
            images_per_parquet=50,
            parquet_per_dataset=1,
        ),
    ],
    "video": [
        BenchmarkDatasetConfig(
            name="gasstation-generated-videos",
            path="gasstation/generated-videos",
            modality="video",
            media_type="synthetic",
            source_format="tar",
            videos_per_zip=20,
            zips_per_dataset=1,
        ),
        BenchmarkDatasetConfig(
            name="pe-video",
            path="facebook/PE-Video",
            modality="video",
            media_type="real",
            source_format="tar",
            videos_per_zip=15,
            zips_per_dataset=1,
        ),
        BenchmarkDatasetConfig(
            name="semisynthetic-video",
            path="bitmind/semisynthetic-video",
            modality="video",
            media_type="semisynthetic",
            source_format="zip",
            videos_per_zip=15,
            zips_per_dataset=1,
        ),
    ],
}

def get_benchmark_dataset_summary() -> Dict:
    """Get a summary of available benchmark datasets by category."""
    image_datasets = BENCHMARK_DATASETS["image"]
    video_datasets = BENCHMARK_DATASETS["video"]

    summary = {
        "image": {
            "total": len(image_datasets),
            "active": len(
                [d for d in image_datasets if d.media_type in ["real", "synthetic"]]
            ),
            "synthetic": len([d for d in image_datasets if d.media_type == "synthetic"]),
            "real": len([d for d in image_datasets if d.media_type == "real"]),
            "datasets": [
                {"name": d.name, "path": d.path, "media_type": d.media_type}
                for d in image_datasets
            ],
        },
        "video": {
            "total": len(video_datasets),
            "active": len(
                [d for d in video_datasets if d.media_type in ["real", "synthetic"]]
            ),
            "synthetic": len([d for d in video_datasets if d.media_type == "synthetic"]),
            "real": len([d for d in video_datasets if d.media_type == "real"]),
            "datasets": [
                {"name": d.name, "path": d.path, "media_type": d.media_type}
                for d in video_datasets
            ],
        },
    }

    return summary


def discover_benchmark_image_datasets(debug_mode: bool = False, gasstation_only: bool = False) -> List[BenchmarkDatasetConfig]:
    """Return list of available benchmark image datasets."""
    dataset_source = DEBUG_BENCHMARK_DATASETS if debug_mode else BENCHMARK_DATASETS
    datasets = dataset_source["image"]
    
    if gasstation_only:
        # Filter to only gasstation datasets
        gasstation_datasets = [d for d in datasets if "gasstation" in d.name.lower()]
        return gasstation_datasets
    
    return datasets


def discover_benchmark_video_datasets(debug_mode: bool = False, gasstation_only: bool = False) -> List[BenchmarkDatasetConfig]:
    """Return list of available benchmark video datasets."""
    dataset_source = DEBUG_BENCHMARK_DATASETS if debug_mode else BENCHMARK_DATASETS
    datasets = dataset_source["video"]
    
    if gasstation_only:
        # Filter to only gasstation datasets
        gasstation_datasets = [d for d in datasets if "gasstation" in d.name.lower()]
        return gasstation_datasets
    
    return datasets


def check_dataset_availability(dataset_config: BenchmarkDatasetConfig) -> bool:
    """Check if a dataset is available."""
    # For now, assume all configured datasets are available
    return True


def calculate_dataset_sampling(
    num_datasets: int,
    target_total_samples: int
) -> tuple[int, int]:
    """
    Calculate per-dataset sampling parameters.
    
    Returns:
        Tuple of (per_dataset_cap, min_samples_per_dataset)
    """
    min_samples_per_dataset = target_total_samples // num_datasets
    per_dataset_cap = min(2000, min_samples_per_dataset * 3)
    return per_dataset_cap, min_samples_per_dataset


def filter_available_datasets(available_datasets: List, check_availability_fn) -> List:
    """Filter datasets based on availability check."""
    from ..logger import get_logger
    logger = get_logger(__name__)
    
    valid_datasets = []
    for dataset_config in available_datasets:
        if check_availability_fn(dataset_config):
            valid_datasets.append(dataset_config)
        else:
            logger.warning(f"Dataset {dataset_config.name} not available, skipping")
    return valid_datasets


def build_dataset_info(valid_datasets: List, samples_per_dataset: int) -> Dict:
    """Build dataset info dictionary for results."""
    return {
        "datasets_used": [d.name for d in valid_datasets],
        "evaluation_type": "ai_generated_detection",
        "samples_per_dataset": samples_per_dataset,
        "dataset_media_types": {d.name: d.media_type for d in valid_datasets},
    }
