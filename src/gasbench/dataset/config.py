from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
import json
import yaml
from importlib.resources import files
from collections import defaultdict
import hashlib

from ..logger import get_logger

logger = get_logger(__name__)


# Dataset sampling configuration
REGULAR_DATASET_MIN_SAMPLES = 100
REGULAR_DATASET_MAX_SAMPLES = 2000
GASSTATION_DATASET_MIN_SAMPLES = 500
GASSTATION_DATASET_MAX_SAMPLES = 10000
GASSTATION_WEIGHT_MULTIPLIER = 5.0  # Gasstation datasets get 5x more samples
UNIFORM_SAMPLING_MULTIPLIER = 3  # Allow up to 3x the base allocation per dataset

# Total sample overrides for all benchmark modes.
# Full-mode sizes moved here from YAML configs — single source of truth.
BENCHMARK_TOTAL_OVERRIDES = {
    "debug": {"image": 100, "video": 50, "audio": 50},
    "small": {"image": 1800, "video": 600, "audio": 600},
    "full":  {"image": 55000, "video": 26000, "audio": 37000},
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
    modality: str  # "image", "video", or "audio"
    media_type: str  # "real", "synthetic", or "semisynthetic"

    # Download parameters
    media_per_archive: int = 100
    archives_per_dataset: int = 5
    source_format: str = ""  # Auto-detected if empty
    source: str = "huggingface"  # "huggingface", "modelscope", or "s3"

    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    
    # For parquet datasets: specify column name(s) containing media bytes
    # Works for any modality (image, audio, video)
    data_columns: Optional[List[str]] = None
    notes: Optional[str] = None

    # Row-level filter applied after downloading each parquet shard.
    # When set, gasbench downloads parquet files one at a time (sequentially)
    # and stops as soon as enough filtered samples have been collected,
    # avoiding unnecessary downloads from large datasets.
    # Example: filter_column: "label", filter_value: "fake"
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None

    # For holdout datasets: stores the original name before obfuscation
    original_name: Optional[str] = None

    generator_family: Optional[str] = None

    content_category: Optional[str] = None


def get_benchmark_size(
    modality: str, mode: str = "full", yaml_path: Optional[str] = None
) -> int:
    """Get the target benchmark size for a given modality and mode.

    Args:
        modality: "image", "video", or "audio"
        mode: "debug", "small", or "full"
        yaml_path: Optional path to custom yaml config. Only used in full mode
            when a custom config is provided; its benchmark_size field is used.

    Returns:
        Target number of samples for the benchmark
    """
    if mode in BENCHMARK_TOTAL_OVERRIDES:
        return BENCHMARK_TOTAL_OVERRIDES[mode][modality]

    # Custom config path: load benchmark_size from the YAML
    if yaml_path is not None:
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if "benchmark_size" in data:
                return data["benchmark_size"]
        except Exception as e:
            logger.warning(f"Failed to load benchmark size from {yaml_path}: {e}")

    # Fallback defaults
    logger.warning(f"No benchmark_size found for {modality} {mode}, using default")
    defaults = {"image": 10000, "video": 5000, "audio": 5000}
    return defaults.get(modality, 5000)


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
            media_per_archive=mode_config.get(
                "media_per_archive", dataset.media_per_archive
            ),
            archives_per_dataset=mode_config.get(
                "archives_per_dataset", dataset.archives_per_dataset
            ),
        )
        return [modified]

    elif mode == "small":
        modified_datasets = []
        for dataset in datasets:
            mode_config = DOWNLOAD_SIZE_OVERRIDES["small"]
            modified = replace(
                dataset,
                media_per_archive=mode_config.get(
                    "media_per_archive", dataset.media_per_archive
                ),
                archives_per_dataset=mode_config.get(
                    "archives_per_dataset", dataset.archives_per_dataset
                ),
            )
            modified_datasets.append(modified)
        return modified_datasets

    else:
        return datasets  # Full mode (or unknown): Use YAML configs as-is


def discover_benchmark_datasets(
    modality: str,
    mode: str = "full",
    gasstation_only: bool = False,
    no_gasstation: bool = False,
    yaml_path: Optional[str] = None,
    content_category: Optional[str] = None,
) -> List[BenchmarkDatasetConfig]:
    """Return list of available benchmark datasets for a given modality.

    Args:
        content_category: If set, only include datasets matching this
            content_category (e.g., "faces", "documents").
    """
    dataset_source = load_benchmark_datasets_from_yaml(yaml_path)
    if modality not in dataset_source:
        return []
    datasets = dataset_source[modality]

    if gasstation_only:
        datasets = [d for d in datasets if "gasstation" in d.name.lower()]

    if no_gasstation:
        datasets = [d for d in datasets if "gasstation" not in d.name.lower()]

    if content_category:
        datasets = [d for d in datasets if d.content_category == content_category]

    datasets = apply_mode_to_datasets(datasets, mode)

    return datasets


def discover_benchmark_audio_datasets(
    mode: str = "full",
    gasstation_only: bool = False,
    no_gasstation: bool = False,
    yaml_path: Optional[str] = None,
) -> List[BenchmarkDatasetConfig]:
    """Return list of available benchmark audio datasets.
    
    Args:
        mode: Benchmark mode - "debug", "small", or "full"
        gasstation_only: If True, only return gasstation datasets
        no_gasstation: If True, exclude gasstation datasets
        yaml_path: Optional path to custom yaml config
    """
    dataset_source = load_benchmark_datasets_from_yaml(yaml_path)
    datasets = dataset_source["audio"]

    datasets = apply_mode_to_datasets(datasets, mode)

    if gasstation_only:
        gasstation_datasets = [d for d in datasets if "gasstation" in d.name.lower()]
        return gasstation_datasets
    
    if no_gasstation:
        non_gasstation_datasets = [d for d in datasets if "gasstation" not in d.name.lower()]
        return non_gasstation_datasets

    return datasets


def calculate_weighted_dataset_sampling(
    datasets: List[BenchmarkDatasetConfig],
    target_total_samples: int,
    gasstation_weight: float = None,
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
        per_dataset_cap = min(REGULAR_DATASET_MAX_SAMPLES, per_dataset_cap)
        return {d.name: per_dataset_cap for d in datasets}

    # Currently each regular dataset gets weight=1, each gasstation gets weight=gasstation_weight
    total_weight = num_regular + (num_gasstation * gasstation_weight)
    samples_per_weight_unit = target_total_samples / total_weight

    regular_cap = int(samples_per_weight_unit)
    gasstation_cap = int(samples_per_weight_unit * gasstation_weight)

    regular_cap = min(
        REGULAR_DATASET_MAX_SAMPLES, max(REGULAR_DATASET_MIN_SAMPLES, regular_cap)
    )
    gasstation_cap = min(
        GASSTATION_DATASET_MAX_SAMPLES,
        max(GASSTATION_DATASET_MIN_SAMPLES, gasstation_cap),
    )

    sampling_dict = {}
    for dataset in datasets:
        if "gasstation" in dataset.name.lower():
            sampling_dict[dataset.name] = gasstation_cap
        else:
            sampling_dict[dataset.name] = regular_cap
    return sampling_dict


def build_dataset_info(
    valid_datasets: List, dataset_sampling: Dict[str, int] = None
) -> Dict:
    """Build dataset info dictionary for results.

    Args:
        valid_datasets: List of dataset configurations
        dataset_sampling: Optional dict mapping dataset names to their sample caps
    """
    info = {
        "datasets_used": [d.name for d in valid_datasets],
        "evaluation_type": "synthetic_detection",
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
        valid_modalities = ["image", "video", "audio"]
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

    if "source" in config_dict:
        valid_sources = ["huggingface", "modelscope", "s3"]
        if config_dict["source"] not in valid_sources:
            errors.append(
                f"Dataset '{dataset_name}': Invalid source '{config_dict['source']}'. "
                f"Must be one of {valid_sources}"
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


def _dataset_dict_to_config(d: dict, **overrides) -> BenchmarkDatasetConfig:
    """Build a BenchmarkDatasetConfig from a raw YAML dict.

    All fields are extracted from the dict with sensible defaults.
    Callers can pass keyword overrides for fields that differ per context
    (e.g. the holdout loader uses different source/media_per_archive defaults).
    """
    kwargs = {
        "name": d["name"],
        "path": d["path"],
        "modality": d.get("modality", "unknown"),
        "media_type": d["media_type"],
        "source_format": d.get("source_format", ""),
        "source": d.get("source", "huggingface"),
        "media_per_archive": d.get("media_per_archive", 100),
        "archives_per_dataset": d.get("archives_per_dataset", 5),
        "include_paths": d.get("include_paths"),
        "exclude_paths": d.get("exclude_paths"),
        "data_columns": d.get("data_columns"),
        "notes": d.get("notes"),
        "filter_column": d.get("filter_column"),
        "filter_value": d.get("filter_value"),
        "generator_family": d.get("generator_family"),
        "content_category": d.get("content_category"),
    }
    kwargs.update(overrides)
    return BenchmarkDatasetConfig(**kwargs)


def load_datasets_from_yaml(yaml_path: str) -> Dict[str, List[BenchmarkDatasetConfig]]:
    """Load benchmark dataset configurations from a YAML file.
    
    Supports two YAML formats:
    1. Flat format with 'datasets' list where each entry has a 'modality' field
    2. Grouped format with 'image', 'video', 'audio' top-level keys
    """
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

    all_errors = []
    result = {"image": [], "video": [], "audio": []}

    def process_dataset(idx: int, dataset_dict: dict, source_key: str):
        if not isinstance(dataset_dict, dict):
            all_errors.append(f"{source_key}[{idx}]: Must be a dictionary")
            return

        dataset_name = dataset_dict.get("name", f"dataset_{idx}")
        validation_errors = validate_dataset_config(dataset_dict, dataset_name)
        all_errors.extend(validation_errors)

        if not validation_errors:
            modality = dataset_dict["modality"].lower()
            config = _dataset_dict_to_config(dataset_dict, modality=modality)
            if modality in result:
                result[modality].append(config)

    if "datasets" in data and isinstance(data["datasets"], list):
        for idx, dataset_dict in enumerate(data["datasets"]):
            process_dataset(idx, dataset_dict, "datasets")
    else:
        for modality in ["image", "video", "audio"]:
            if modality not in data:
                continue
            if not isinstance(data[modality], list):
                raise ValueError(f"'{modality}' must be a list of dataset configurations")
            for idx, dataset_dict in enumerate(data[modality]):
                process_dataset(idx, dataset_dict, modality)

    if all_errors:
        error_msg = "Validation errors in YAML file:\n" + "\n".join(
            f"  - {e}" for e in all_errors
        )
        raise ValueError(error_msg)

    return result


def _obfuscate_holdout_names(
    datasets: List[BenchmarkDatasetConfig],
) -> Tuple[List[BenchmarkDatasetConfig], Dict[str, str]]:
    """
    Obfuscate dataset names for holdout datasets using a stable short hash so names
    do not change when items are inserted or reordered.
    Example: real-video-holdout-a1b2c3d4

    Returns:
        Tuple of (obfuscated_datasets, mapping_dict)
        mapping_dict maps obfuscated_name -> original_name
    """
    obfuscated: List[BenchmarkDatasetConfig] = []
    mapping: Dict[str, str] = {}

    for d in datasets:
        orig_name = d.name
        include_paths = ",".join(sorted(d.include_paths)) if d.include_paths else ""
        exclude_paths = ",".join(sorted(d.exclude_paths)) if d.exclude_paths else ""
        fingerprint = "|".join(
            [
                d.path or "",
                d.modality or "",
                d.media_type or "",
                (d.source_format or ""),
                include_paths,
                exclude_paths,
                (d.source or ""),
            ]
        )
        short_hash = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]
        new_name = f"{d.media_type}-{d.modality}-holdout-{short_hash}"
        obfuscated.append(replace(d, name=new_name, original_name=orig_name))
        mapping[new_name] = orig_name

    return obfuscated, mapping


def load_holdout_datasets_from_yaml(
    yaml_path: str, cache_dir: Optional[str] = None, modality: Optional[str] = None
) -> Dict[str, List[BenchmarkDatasetConfig]]:
    """
    Load holdout datasets from YAML and return dict with obfuscated names.
    
    Supports two formats:
    1. Single-modality format (new): Has 'datasets' key with list of datasets for one modality
    2. Multi-modality format (legacy): Has 'image', 'video', 'audio' keys

    Also saves a mapping file (<modality>-holdout-mappings.yaml) in the cache directory
    showing obfuscated_name -> original_name mappings.

    Args:
        yaml_path: Path to holdout YAML config
        cache_dir: Cache directory to save mapping file (optional)
        modality: Modality hint for single-modality configs (optional, auto-detected from datasets)
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Holdout config file not found: {yaml_path}")

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    result = {"image": [], "video": [], "audio": []}
    all_mappings = {"image": {}, "video": {}, "audio": {}}

    if "datasets" in data:
        datasets_list = data["datasets"]
        detected_modality = None
        
        for dataset_dict in datasets_list:
            if not isinstance(dataset_dict, dict):
                continue
            ds_modality = dataset_dict.get("modality")
            if ds_modality:
                detected_modality = ds_modality
                break
        
        effective_modality = modality or detected_modality
        if not effective_modality:
            raise ValueError("Could not determine modality from holdout config")
        
        configs = []
        for dataset_dict in datasets_list:
            if not isinstance(dataset_dict, dict):
                continue
            config = _dataset_dict_to_config(
                dataset_dict,
                modality=dataset_dict.get("modality", effective_modality),
                source="",
                media_per_archive=dataset_dict.get("media_per_archive", -1),
                archives_per_dataset=dataset_dict.get("archives_per_dataset", -1),
                include_paths=dataset_dict.get("include_paths", []),
                exclude_paths=dataset_dict.get("exclude_paths", []),
                notes=dataset_dict.get("notes", ""),
            )
            configs.append(config)
        
        obfuscated, mapping = _obfuscate_holdout_names(configs)
        result[effective_modality] = obfuscated
        all_mappings[effective_modality] = mapping
        
        if cache_dir:
            _save_holdout_mapping(cache_dir, effective_modality, mapping, yaml_path)
    else:
        base = load_datasets_from_yaml(yaml_path)
        
        for mod in ["image", "video", "audio"]:
            if mod in base and base[mod]:
                obfuscated, mapping = _obfuscate_holdout_names(base[mod])
                result[mod] = obfuscated
                all_mappings[mod] = mapping
                
                if cache_dir:
                    _save_holdout_mapping(cache_dir, mod, mapping, yaml_path)

    return result


def _save_holdout_mapping(cache_dir: str, modality: str, mapping: Dict[str, str], source_config: str):
    """Save holdout mapping to YAML file.

    The mapping filename is derived from the source config basename so that
    different verticals (e.g. image-holdouts.yaml vs image-human-holdouts.yaml)
    produce separate mapping files instead of overwriting each other.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        config_stem = Path(source_config).stem  # e.g. "image-human-holdouts"
        mapping_file = os.path.join(cache_dir, f"{config_stem}-mappings.yaml")
        full_mapping = {
            "modality": modality,
            "source_config": source_config,
            "mappings": mapping,
        }
        with open(mapping_file, "w") as f:
            yaml.dump(full_mapping, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved holdout name mapping to: {mapping_file}")
    except Exception as e:
        logger.warning(f"Failed to save holdout mapping file: {e}")


def _load_bundled_config(filename: str) -> Dict:
    """Load a config file from package data."""
    try:
        config_resource = files("gasbench.dataset") / "configs" / filename
        yaml_text = config_resource.read_text()
        return yaml.safe_load(yaml_text)
    except Exception as e:
        raise FileNotFoundError(f"Could not load bundled config {filename}: {e}")


def _load_modality_config(modality: str, custom_path: Optional[str] = None) -> list:
    """Load configuration for a specific modality.
    
    Args:
        modality: 'image', 'video', or 'audio'
        custom_path: Optional custom YAML file path
    
    Returns:
        List of dataset dicts (merged from real + synthetic split files)
    """
    if custom_path:
        with open(custom_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("datasets", [])
    
    # Load from split real + synthetic configs and merge
    SPLIT_CONFIGS = {
        "image": ("real_images.yaml", "synthetic_images.yaml"),
        "video": ("real_videos.yaml", "synthetic_videos.yaml"),
        "audio": ("real_audio.yaml", "synthetic_audio.yaml"),
    }
    
    pairs = SPLIT_CONFIGS.get(modality)
    if not pairs:
        return 10000, []
    
    all_datasets = []
    for filename in pairs:
        try:
            data = _load_bundled_config(filename)
            if isinstance(data, dict):
                if "datasets" in data and isinstance(data["datasets"], list):
                    all_datasets.extend(data["datasets"])
        except FileNotFoundError:
            logger.warning(f"Config file {filename} not found, skipping")
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}, skipping")
    
    return all_datasets


def load_benchmark_datasets_from_yaml(
    yaml_path: Optional[str] = None,
) -> Dict[str, List[BenchmarkDatasetConfig]]:
    """Load benchmark datasets from YAML files.
    
    Args:
        yaml_path: Optional path to custom yaml config. If None, loads bundled configs.
                   Custom path can be legacy single file or new modality-specific file.
    """
    result = {
        "image": [],
        "video": [],
        "audio": [],
    }
    
    try:
        # If custom yaml_path is provided, try to load it
        if yaml_path is not None:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            # Check if it's the old single-file format (has 'image'/'video'/'audio' keys at root)
            if "image" in data or "video" in data or "audio" in data:
                # Legacy single-file format
                all_errors = []
                for modality in ["image", "video", "audio"]:
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
                            config = _dataset_dict_to_config(dataset_dict)
                            result[modality].append(config)
                
                if all_errors:
                    raise ValueError("Validation errors:\n" + "\n".join(f"  - {e}" for e in all_errors))
                return result
            
            # New single-modality format (has 'datasets' key)
            elif "datasets" in data:
                datasets_list = data.get("datasets", [])
                # Infer modality from first dataset or fail
                if datasets_list:
                    modality = datasets_list[0].get("modality", "unknown")
                    for dataset_dict in datasets_list:
                        dataset_name = dataset_dict.get("name", "unknown")
                        validation_errors = validate_dataset_config(dataset_dict, dataset_name)
                        if not validation_errors:
                            config = _dataset_dict_to_config(dataset_dict)
                            result[modality].append(config)
                return result
        
        # Load from split real + synthetic config files (new default)
        all_errors = []
        for modality in ["image", "video", "audio"]:
            try:
                datasets_list = _load_modality_config(modality)
                
                for idx, dataset_dict in enumerate(datasets_list):
                    if not isinstance(dataset_dict, dict):
                        all_errors.append(f"{modality}[{idx}]: Must be a dictionary")
                        continue
                    
                    dataset_name = dataset_dict.get("name", f"{modality}_dataset_{idx}")
                    validation_errors = validate_dataset_config(dataset_dict, dataset_name)
                    all_errors.extend(validation_errors)
                    
                    if not validation_errors:
                        config = _dataset_dict_to_config(dataset_dict)
                        result[modality].append(config)
            except Exception as e:
                logger.warning(f"Failed to load {modality} datasets: {e}")
        
        if all_errors:
            raise ValueError("Validation errors:\n" + "\n".join(f"  - {e}" for e in all_errors))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}. Cannot load datasets.")
        return {
            "image": [],
            "video": [],
            "audio": [],
        }


def get_benchmark_dataset_summary() -> Dict:
    """Get a summary of available benchmark datasets by category."""
    datasets = load_benchmark_datasets_from_yaml()
    image_datasets = datasets["image"]
    video_datasets = datasets["video"]
    audio_datasets = datasets["audio"]

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
        "audio": {
            "total": len(audio_datasets),
            "active": len(
                [d for d in audio_datasets if d.media_type in ["real", "synthetic"]]
            ),
            "synthetic": len(
                [d for d in audio_datasets if d.media_type == "synthetic"]
            ),
            "real": len([d for d in audio_datasets if d.media_type == "real"]),
            "datasets": [
                {"name": d.name, "path": d.path, "media_type": d.media_type}
                for d in audio_datasets
            ],
        },
    }

    return summary
