from pathlib import Path
from typing import Dict, List, Optional


def generate_sample_id(
    sample: Dict,
    sample_index: int,
    iso_week: Optional[str] = None,
    source_file: Optional[str] = None,
    generator_hotkey: Optional[str] = None,
    generator_uid: Optional[str] = None,
) -> str:
    """Generate composite unique identifier for a gasstation sample.
    
    Args:
        sample: Sample dictionary
        sample_index: Index of the sample in the current iteration
        iso_week: ISO week string (extracted from sample if not provided)
        source_file: Source file name (extracted from sample if not provided)
        generator_hotkey: Generator hotkey (extracted from sample if not provided)
        generator_uid: Generator UID (extracted from sample if not provided)
    
    Returns:
        Composite ID string: {iso_week}_{source}_{sample_index}_{hotkey}_{uid}
    """
    # Extract values from sample if not provided
    iso_week = iso_week or sample.get("iso_week", "unknown")
    source_file = source_file or sample.get("source_file", "unknown")
    generator_hotkey = generator_hotkey or sample.get("generator_hotkey", "unknown")
    generator_uid = generator_uid or sample.get("generator_uid", "unknown")
    
    # Get just the filename stem (without extension) from source_file
    source_stem = Path(source_file).stem if source_file != "unknown" else "unknown"
    
    # Truncate hotkey for readability (keep first 12 chars)
    hotkey_short = (
        generator_hotkey[:12] if isinstance(generator_hotkey, str) and generator_hotkey != "unknown"
        else str(generator_hotkey)
    )
    
    return f"{iso_week}_{source_stem}_{sample_index:06d}_{hotkey_short}_{generator_uid}"


def should_track_sample(sample: Dict, dataset_name: str) -> bool:
    """Check if a sample should be tracked for misclassification.
    
    Only track gasstation samples that have both generator_hotkey and generator_uid.
    
    Args:
        sample: Sample dictionary
        dataset_name: Name of the dataset
    
    Returns:
        True if sample should be tracked, False otherwise
    """
    # Must be a gasstation dataset
    if "gasstation" not in dataset_name.lower():
        return False
    
    # Must have both generator fields and they must not be "unknown" or empty
    generator_hotkey = sample.get("generator_hotkey")
    generator_uid = sample.get("generator_uid")
    
    has_hotkey = (
        generator_hotkey is not None 
        and generator_hotkey != "unknown" 
        and generator_hotkey != ""
    )
    has_uid = (
        generator_uid is not None 
        and generator_uid != "unknown" 
        and generator_uid != ""
    )
    
    return has_hotkey and has_uid


def create_misclassification_record(
    sample: Dict,
    sample_index: int,
    true_label: int,
    predicted_label: int,
) -> Dict:
    """Create a misclassification record for an incorrectly classified sample.
    
    Args:
        sample: Sample dictionary with metadata
        sample_index: Index of the sample in the current iteration
        true_label: True label
        predicted_label: Predicted label
    
    Returns:
        Dictionary containing misclassification details
    """
    sample_id = generate_sample_id(sample, sample_index)
    
    return {
        "sample_id": sample_id,
        "generator_hotkey": sample.get("generator_hotkey", "unknown"),
        "generator_uid": sample.get("generator_uid", "unknown"),
        "iso_week": sample.get("iso_week", "unknown"),
        "source_file": sample.get("source_file", "unknown"),
        "dataset_name": sample.get("dataset_name", "unknown"),
        "media_type": sample.get("media_type", "unknown"),
        "model_name": sample.get("model_name", "unknown"),
        "true_label": true_label,
        "predicted_label": predicted_label,
    }


def aggregate_misclassification_stats(incorrect_samples: List[Dict]) -> Dict:
    """Aggregate statistics from misclassified samples.
    
    Args:
        incorrect_samples: List of misclassification records
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not incorrect_samples:
        return {
            "total_misclassified": 0,
            "by_generator": {},
            "by_week": {},
            "by_dataset": {},
        }
    
    by_generator = {}
    by_week = {}
    by_dataset = {}
    
    for record in incorrect_samples:
        # Aggregate by generator (hotkey)
        hotkey = record.get("generator_hotkey", "unknown")
        by_generator[hotkey] = by_generator.get(hotkey, 0) + 1
        
        # Aggregate by ISO week
        week = record.get("iso_week", "unknown")
        by_week[week] = by_week.get(week, 0) + 1
        
        # Aggregate by dataset
        dataset = record.get("dataset_name", "unknown")
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
    
    return {
        "total_misclassified": len(incorrect_samples),
        "by_generator": by_generator,
        "by_week": by_week,
        "by_dataset": by_dataset,
    }

