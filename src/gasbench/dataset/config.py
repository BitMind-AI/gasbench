from dataclasses import dataclass, replace
from typing import Dict, List, Optional
from pathlib import Path
import os
import yaml
from importlib.resources import files

from ..logger import get_logger

logger = get_logger(__name__)


# Total sample targets for each benchmark mode (used for runtime evaluation sampling)
BENCHMARK_TOTAL_SAMPLES = {
    "debug": {"image": 100, "video": 50},
    "small": {"image": 1800, "video": 600},
    "full": {"image": 10000, "video": 5000},
}

# Per-dataset download limits (only applied in debug/small modes for faster testing)
# In "full" mode, YAML configs are respected
DOWNLOAD_SIZE_OVERRIDES = {
    "debug": {
        "image": {
            "images_per_parquet": 100,
            "parquet_per_dataset": 1,
        },
        "video": {
            "videos_per_zip": 50,
            "zips_per_dataset": 1,
        },
    },
    "small": {
        "image": {
            "images_per_parquet": 100,
            "parquet_per_dataset": 1,
        },
        "video": {
            "videos_per_zip": 100,
            "zips_per_dataset": 1,
        },
    },
}

@dataclass
class BenchmarkDatasetConfig:
    name: str
    path: str
    modality: str  # "image" or "video"
    media_type: str  # "real", "synthetic", or "semisynthetic"

    # Download parameters
    images_per_parquet: int = 100
    videos_per_zip: int = 50
    parquet_per_dataset: int = 5
    zips_per_dataset: int = 2
    source_format: str = ""  # Auto-detected if empty
    source: str = "huggingface"  # "huggingface" or "modelscope"

    include_paths: Optional[List[str]] = None 
    exclude_paths: Optional[List[str]] = None


def get_benchmark_size(modality: str, mode: str = "full") -> int:
    """Get the target benchmark size for a given modality and mode.
    
    Args:
        modality: "image" or "video"
        mode: "debug", "small", or "full"
        
    Returns:
        Target number of samples for the benchmark
    """
    return BENCHMARK_TOTAL_SAMPLES.get(mode, BENCHMARK_TOTAL_SAMPLES["full"])[modality]


def apply_mode_to_datasets(
    datasets: List[BenchmarkDatasetConfig], mode: str
) -> List[BenchmarkDatasetConfig]:
    """Apply mode transformations to dataset configurations.
    
    Args:
        datasets: List of dataset configurations
        mode: One of "debug", "small", or "full"
        
    Returns:
        Transformed list of datasets based on mode
    """
    if mode == "debug":
        if not datasets:
            return []
        dataset = datasets[0]
        mode_config = DOWNLOAD_SIZE_OVERRIDES["debug"][dataset.modality]
        modified = replace(
            dataset,
            images_per_parquet=mode_config.get("images_per_parquet", dataset.images_per_parquet),
            videos_per_zip=mode_config.get("videos_per_zip", dataset.videos_per_zip),
            parquet_per_dataset=mode_config.get("parquet_per_dataset", dataset.parquet_per_dataset),
            zips_per_dataset=mode_config.get("zips_per_dataset", dataset.zips_per_dataset),
        )
        return [modified]
    
    elif mode == "small":
        modified_datasets = []
        for dataset in datasets:
            mode_config = DOWNLOAD_SIZE_OVERRIDES["small"][dataset.modality]
            modified = replace(
                dataset,
                images_per_parquet=mode_config.get("images_per_parquet", dataset.images_per_parquet),
                videos_per_zip=mode_config.get("videos_per_zip", dataset.videos_per_zip),
                parquet_per_dataset=mode_config.get("parquet_per_dataset", dataset.parquet_per_dataset),
                zips_per_dataset=mode_config.get("zips_per_dataset", dataset.zips_per_dataset),
            )
            modified_datasets.append(modified)
        return modified_datasets
    
    else:
        return datasets  # Full mode (or unknown): Use YAML configs as-is


def discover_benchmark_image_datasets(
    mode: str = "full",
    gasstation_only: bool = False,
    yaml_path: Optional[str] = None,
) -> List[BenchmarkDatasetConfig]:
    """Return list of available benchmark image datasets.
    
    Args:
        mode: Benchmark mode - "debug", "small", or "full"
        gasstation_only: If True, only return gasstation datasets
        yaml_path: Optional path to custom yaml config
    """
    dataset_source = load_benchmark_datasets_from_yaml(yaml_path)
    datasets = dataset_source["image"]

    datasets = apply_mode_to_datasets(datasets, mode)

    if gasstation_only:
        gasstation_datasets = [d for d in datasets if "gasstation" in d.name.lower()]
        return gasstation_datasets

    return datasets


def discover_benchmark_video_datasets(
    mode: str = "full",
    gasstation_only: bool = False,
    yaml_path: Optional[str] = None,
) -> List[BenchmarkDatasetConfig]:
    """Return list of available benchmark video datasets.
    
    Args:
        mode: Benchmark mode - "debug", "small", or "full"
        gasstation_only: If True, only return gasstation datasets
        yaml_path: Optional path to custom yaml config
    """
    dataset_source = load_benchmark_datasets_from_yaml(yaml_path)
    datasets = dataset_source["video"]

    datasets = apply_mode_to_datasets(datasets, mode)

    if gasstation_only:
        gasstation_datasets = [d for d in datasets if "gasstation" in d.name.lower()]
        return gasstation_datasets

    return datasets


def calculate_dataset_sampling(
    num_datasets: int, target_total_samples: int
) -> tuple[int, int]:
    """
    Calculate per-dataset sampling parameters.

    Returns:
        Tuple of (per_dataset_cap, min_samples_per_dataset)
    """
    min_samples_per_dataset = target_total_samples // num_datasets
    per_dataset_cap = min(2000, min_samples_per_dataset * 3)
    return per_dataset_cap, min_samples_per_dataset


def build_dataset_info(valid_datasets: List, samples_per_dataset: int) -> Dict:
    """Build dataset info dictionary for results."""
    return {
        "datasets_used": [d.name for d in valid_datasets],
        "evaluation_type": "ai_generated_detection",
        "samples_per_dataset": samples_per_dataset,
        "dataset_media_types": {d.name: d.media_type for d in valid_datasets},
    }


def validate_dataset_config(
    config_dict: Dict, dataset_name: str = "unknown"
) -> List[str]:
    """Validate a dataset configuration dictionary."""
    errors = []

    required_fields = ["name", "path", "modality", "media_type"]
    for field in required_fields:
        if field not in config_dict:
            errors.append(f"Dataset '{dataset_name}': Missing required field '{field}'")

    if "modality" in config_dict:
        valid_modalities = ["image", "video"]
        if config_dict["modality"] not in valid_modalities:
            errors.append(
                f"Dataset '{dataset_name}': Invalid modality '{config_dict['modality']}'. "
                f"Must be one of {valid_modalities}"
            )

    if "media_type" in config_dict:
        valid_media_types = ["real", "synthetic", "semisynthetic"]
        if config_dict["media_type"] not in valid_media_types:
            errors.append(
                f"Dataset '{dataset_name}': Invalid media_type '{config_dict['media_type']}'. "
                f"Must be one of {valid_media_types}"
            )

    numeric_fields = [
        "images_per_parquet",
        "videos_per_zip",
        "parquet_per_dataset",
        "zips_per_dataset",
    ]
    for field in numeric_fields:
        if field in config_dict:
            value = config_dict[field]
            if not isinstance(value, int):
                errors.append(
                    f"Dataset '{dataset_name}': Field '{field}' must be an integer, got {type(value).__name__}"
                )

    return errors


def load_datasets_from_yaml(yaml_path: str) -> Dict[str, List[BenchmarkDatasetConfig]]:
    """Load benchmark dataset configurations from a YAML file."""
    yaml_file = Path(yaml_path)

    if not yaml_file.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {yaml_path}")

    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary at root level")

    # Validate
    all_errors = []
    result = {"image": [], "video": []}
    for modality in ["image", "video"]:
        if modality not in data:
            continue

        if not isinstance(data[modality], list):
            raise ValueError(f"'{modality}' must be a list of dataset configurations")

        for idx, dataset_dict in enumerate(data[modality]):
            if not isinstance(dataset_dict, dict):
                all_errors.append(f"{modality}[{idx}]: Must be a dictionary")
                continue

            dataset_name = dataset_dict.get("name", f"{modality}_dataset_{idx}")

            validation_errors = validate_dataset_config(dataset_dict, dataset_name)
            all_errors.extend(validation_errors)

            if not validation_errors:
                config = BenchmarkDatasetConfig(
                    name=dataset_dict["name"],
                    path=dataset_dict["path"],
                    modality=dataset_dict["modality"],
                    media_type=dataset_dict["media_type"],
                    images_per_parquet=dataset_dict.get("images_per_parquet", 100),
                    videos_per_zip=dataset_dict.get("videos_per_zip", 50),
                    parquet_per_dataset=dataset_dict.get("parquet_per_dataset", 5),
                    zips_per_dataset=dataset_dict.get("zips_per_dataset", 2),
                    source_format=dataset_dict.get("source_format", ""),
                    source=dataset_dict.get("source", "huggingface"),
                    include_paths=dataset_dict.get("include_paths"),
                    exclude_paths=dataset_dict.get("exclude_paths"),
                )
                result[modality].append(config)

    if all_errors:
        error_msg = "Validation errors in YAML file:\n" + "\n".join(
            f"  - {e}" for e in all_errors
        )
        raise ValueError(error_msg)

    return result


def _load_bundled_config(filename: str) -> Dict:
    """Load a config file from package data."""
    try:
        config_resource = files("gasbench.dataset") / "configs" / filename
        yaml_text = config_resource.read_text()
        return yaml.safe_load(yaml_text)
    except Exception as e:
        raise FileNotFoundError(f"Could not load bundled config {filename}: {e}")


def load_benchmark_datasets_from_yaml(
    yaml_path: Optional[str] = None
) -> Dict[str, List[BenchmarkDatasetConfig]]:
    """Load benchmark datasets from YAML file.
    
    Args:
        yaml_path: Optional path to custom yaml config. If None, loads bundled config.
    """
    filename = "benchmark_datasets.yaml"

    try:
        if yaml_path is not None:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = _load_bundled_config(filename)
    except Exception as e:
        logger.error(f"Failed to load YAML config: {e}. Cannot load datasets.")
        return {
            "image": [],
            "video": [],
        }

    try:
        if not isinstance(data, dict):
            raise ValueError("YAML must contain a dictionary at root")

        all_errors = []
        result = {
            "image": [],
            "video": [],
        }

        for modality in ["image", "video"]:
            if modality not in data:
                continue

            if not isinstance(data[modality], list):
                raise ValueError(f"'{modality}' must be a list")

            for idx, dataset_dict in enumerate(data[modality]):
                if not isinstance(dataset_dict, dict):
                    all_errors.append(f"{modality}[{idx}]: Must be a dictionary")
                    continue

                dataset_name = dataset_dict.get("name", f"{modality}_dataset_{idx}")
                validation_errors = validate_dataset_config(dataset_dict, dataset_name)
                all_errors.extend(validation_errors)

                if not validation_errors:
                    config = BenchmarkDatasetConfig(
                        name=dataset_dict["name"],
                        path=dataset_dict["path"],
                        modality=dataset_dict["modality"],
                        media_type=dataset_dict["media_type"],
                        images_per_parquet=dataset_dict.get("images_per_parquet", 100),
                        videos_per_zip=dataset_dict.get("videos_per_zip", 50),
                        parquet_per_dataset=dataset_dict.get("parquet_per_dataset", 5),
                        zips_per_dataset=dataset_dict.get("zips_per_dataset", 2),
                        source_format=dataset_dict.get("source_format", ""),
                        source=dataset_dict.get("source", "huggingface"),
                    )
                    result[modality].append(config)

        if all_errors:
            raise ValueError(
                "Validation errors:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )

        return result

    except Exception as e:
        logger.error(f"Failed to parse YAML: {e}. Cannot load datasets.")
        return {
            "image": [],
            "video": [],
        }


def get_benchmark_dataset_summary() -> Dict:
    """Get a summary of available benchmark datasets by category."""
    datasets = load_benchmark_datasets_from_yaml()
    image_datasets = datasets["image"]
    video_datasets = datasets["video"]

    summary = {
        "image": {
            "total": len(image_datasets),
            "active": len(
                [d for d in image_datasets if d.media_type in ["real", "synthetic"]]
            ),
            "synthetic": len(
                [d for d in image_datasets if d.media_type == "synthetic"]
            ),
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
            "synthetic": len(
                [d for d in video_datasets if d.media_type == "synthetic"]
            ),
            "real": len([d for d in video_datasets if d.media_type == "real"]),
            "datasets": [
                {"name": d.name, "path": d.path, "media_type": d.media_type}
                for d in video_datasets
            ],
        },
    }

    return summary
