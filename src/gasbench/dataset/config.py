from dataclasses import dataclass, replace
from typing import Dict, List, Optional
from pathlib import Path
import os
import yaml
from importlib.resources import files

from ..logger import get_logger

logger = get_logger(__name__)


# Dataset sampling configuration
REGULAR_DATASET_MIN_SAMPLES = 100
REGULAR_DATASET_MAX_SAMPLES = 2000
GASSTATION_DATASET_MIN_SAMPLES = 500
GASSTATION_DATASET_MAX_SAMPLES = 10000
GASSTATION_WEIGHT_MULTIPLIER = 5.0  # Gasstation datasets get 5x more samples
UNIFORM_SAMPLING_MULTIPLIER = 3  # Allow up to 3x the base allocation per dataset

# Total sample overrides for debug/small modes (for faster testing)
# In "full" mode, values are loaded from YAML (image_benchmark_size, video_benchmark_size)
BENCHMARK_TOTAL_OVERRIDES = {
    "debug": {"image": 100, "video": 50},
    "small": {"image": 1800, "video": 600},
}

# Per-dataset download limits (only applied in debug/small modes for faster testing)
# In "full" mode, YAML configs are respected
DOWNLOAD_SIZE_OVERRIDES = {
    "debug": {
        "media_per_archive": 100,
        "archives_per_dataset": 1,
    },
    "small": {
        "media_per_archive": 100,
        "archives_per_dataset": 1,
    },
}

@dataclass
class BenchmarkDatasetConfig:
    name: str
    path: str
    modality: str  # "image" or "video"
    media_type: str  # "real", "synthetic", or "semisynthetic"

    # Download parameters
    media_per_archive: int = 100
    archives_per_dataset: int = 5
    source_format: str = ""  # Auto-detected if empty
    source: str = "huggingface"  # "huggingface" or "modelscope"

    include_paths: Optional[List[str]] = None 
    exclude_paths: Optional[List[str]] = None


def get_benchmark_size(modality: str, mode: str = "full", yaml_path: Optional[str] = None) -> int:
    """Get the target benchmark size for a given modality and mode.
    
    Args:
        modality: "image" or "video"
        mode: "debug", "small", or "full"
        yaml_path: Optional path to custom yaml config (for full mode)
        
    Returns:
        Target number of samples for the benchmark
    """
    if mode in BENCHMARK_TOTAL_OVERRIDES:
        return BENCHMARK_TOTAL_OVERRIDES[mode][modality]

    # Full mode: load from YAML
    try:
        if yaml_path is not None:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = _load_bundled_config("benchmark_datasets.yaml")

        size_key = f"{modality}_benchmark_size"
        if size_key in data:
            return data[size_key]

        logger.warning(f"'{size_key}' not found in YAML, using default")
        return 10000 if modality == "image" else 5000

    except Exception as e:
        logger.warning(f"Failed to load benchmark size from YAML: {e}, using default")
        return 10000 if modality == "image" else 5000


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
        mode_config = DOWNLOAD_SIZE_OVERRIDES["debug"]
        modified = replace(
            dataset,
            media_per_archive=mode_config.get("media_per_archive", dataset.media_per_archive),
            archives_per_dataset=mode_config.get("archives_per_dataset", dataset.archives_per_dataset),
        )
        return [modified]
    
    elif mode == "small":
        modified_datasets = []
        for dataset in datasets:
            mode_config = DOWNLOAD_SIZE_OVERRIDES["small"]
            modified = replace(
                dataset,
                media_per_archive=mode_config.get("media_per_archive", dataset.media_per_archive),
                archives_per_dataset=mode_config.get("archives_per_dataset", dataset.archives_per_dataset),
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


def calculate_weighted_dataset_sampling(
    datasets: List[BenchmarkDatasetConfig], 
    target_total_samples: int, 
    gasstation_weight: float = None
) -> Dict[str, int]:
    """
    Calculate per-dataset sampling with higher weight for gasstation datasets.
    
    Args:
        datasets: List of dataset configurations
        target_total_samples: Total samples to aim for across all datasets
        gasstation_weight: Weight multiplier for gasstation datasets (default: GASSTATION_WEIGHT_MULTIPLIER)
    
    Returns:
        Dict mapping dataset names to their max_samples cap
    """
    if gasstation_weight is None:
        gasstation_weight = GASSTATION_WEIGHT_MULTIPLIER

    gasstation_datasets = [d for d in datasets if "gasstation" in d.name.lower()]
    regular_datasets = [d for d in datasets if "gasstation" not in d.name.lower()]
    num_gasstation = len(gasstation_datasets)
    num_regular = len(regular_datasets)
    
    if num_gasstation == 0:
        per_dataset_cap = target_total_samples // len(datasets) if datasets else 0
        return {
            d.name: min(REGULAR_DATASET_MAX_SAMPLES, per_dataset_cap * UNIFORM_SAMPLING_MULTIPLIER) 
            for d in datasets
        }
    
    # Currently each regular dataset gets weight=1, each gasstation gets weight=gasstation_weight
    total_weight = num_regular + (num_gasstation * gasstation_weight)
    samples_per_weight_unit = target_total_samples / total_weight

    regular_cap = int(samples_per_weight_unit)
    gasstation_cap = int(samples_per_weight_unit * gasstation_weight)

    regular_cap = min(REGULAR_DATASET_MAX_SAMPLES, max(REGULAR_DATASET_MIN_SAMPLES, regular_cap))
    gasstation_cap = min(GASSTATION_DATASET_MAX_SAMPLES, max(GASSTATION_DATASET_MIN_SAMPLES, gasstation_cap))

    sampling_dict = {}
    for dataset in datasets:
        if "gasstation" in dataset.name.lower():
            sampling_dict[dataset.name] = gasstation_cap
        else:
            sampling_dict[dataset.name] = regular_cap    
    return sampling_dict


def build_dataset_info(valid_datasets: List, dataset_sampling: Dict[str, int] = None) -> Dict:
    """Build dataset info dictionary for results.

    Args:
        valid_datasets: List of dataset configurations
        dataset_sampling: Optional dict mapping dataset names to their sample caps
    """
    info = {
        "datasets_used": [d.name for d in valid_datasets],
        "evaluation_type": "ai_generated_detection",
        "dataset_media_types": {d.name: d.media_type for d in valid_datasets},
    }
    if dataset_sampling:
        info["samples_per_dataset"] = dataset_sampling

    return info


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
        "media_per_archive",
        "archives_per_dataset",
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
                    media_per_archive=dataset_dict.get("media_per_archive", 100),
                    archives_per_dataset=dataset_dict.get("archives_per_dataset", 5),
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
                        media_per_archive=dataset_dict.get("media_per_archive", 100),
                        archives_per_dataset=dataset_dict.get("archives_per_dataset", 5),
                        source_format=dataset_dict.get("source_format", ""),
                        source=dataset_dict.get("source", "huggingface"),
                        include_paths=dataset_dict.get("include_paths"),
                        exclude_paths=dataset_dict.get("exclude_paths"),
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
