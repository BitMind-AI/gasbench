"""Metadata extraction and cleaning utilities for gasbench datasets."""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ...logger import get_logger

logger = get_logger(__name__)


def create_sample(
    dataset, media_obj, source_path: Path, iso_week: Optional[str] = None
) -> Dict[str, Any]:
    """Create a complete sample in the format expected by processing functions.
    
    Args:
        dataset: BenchmarkDatasetConfig object
        media_obj: PIL Image for images, or bytes for videos
        source_path: Path to the source file
        iso_week: Optional ISO week string (for gasstation datasets)
        
    Returns:
        Sample dictionary with media and metadata
    """
    base_sample = {
        "media_type": dataset.media_type,
        "dataset_name": dataset.name,
        "dataset_path": dataset.path,
        "source_file": source_path.name,
    }

    if iso_week:
        base_sample["iso_week"] = iso_week

    if dataset.modality == "image":
        base_sample["image"] = media_obj
    elif dataset.modality == "audio":
        base_sample["audio_bytes"] = media_obj
    else:
        base_sample["video_bytes"] = media_obj

    return base_sample


def clean_to_json_serializable(value: Any) -> Any:
    """Convert arbitrary values to JSON-serializable equivalents.
    
    Handles:
    - numpy scalars -> native python types
    - numpy arrays / lists / tuples / sets -> list of cleaned values
    - bytes -> base64 string
    - datetime -> isoformat string
    - dict -> recursively cleaned
    - NaN/Inf -> None
    - fallback -> str(value) if still not JSON-serializable
    
    Args:
        value: Value to convert
        
    Returns:
        JSON-serializable version of the value
    """
    try:
        if value is None or isinstance(value, (str, int, bool)):
            return value

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        # numpy scalar types
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        if isinstance(value, (np.bool_,)):
            return bool(value)

        # bytes -> base64
        if isinstance(value, (bytes, bytearray)):
            try:
                return base64.b64encode(bytes(value)).decode("ascii")
            except Exception:
                return None

        # datetime -> isoformat
        if isinstance(value, datetime):
            return value.isoformat()

        # numpy arrays and iterables
        if isinstance(value, (np.ndarray, list, tuple, set)):
            return [
                clean_to_json_serializable(v)
                for v in (
                    value.tolist() if isinstance(value, np.ndarray) else list(value)
                )
            ]

        # dict-like
        if isinstance(value, dict):
            return {str(k): clean_to_json_serializable(v) for k, v in value.items()}

        # Fallback: ensure JSON-serializable; else stringify
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)
    except Exception:
        return None


def extract_row_metadata(row: Any, media_col: str) -> Dict[str, Any]:
    """Extract non-media columns from a pandas Series row and clean to JSON-serializable dict.
    
    Args:
        row: Pandas Series row
        media_col: Name of the media column to exclude
        
    Returns:
        Dictionary with cleaned metadata
    """
    metadata: Dict[str, Any] = {}
    try:
        for col, val in row.items():
            if str(col) == str(media_col):
                continue
            cleaned = clean_to_json_serializable(val)
            col_str = str(col)
            metadata[col_str] = cleaned

            # Map common variations to standard field names for gasstation datasets
            col_lower = col_str.lower()
            if "hotkey" in col_lower and "generator_hotkey" not in metadata:
                metadata["generator_hotkey"] = cleaned
            if "uid" in col_lower and "generator_uid" not in metadata:
                metadata["generator_uid"] = cleaned
    except Exception:
        pass
    return metadata

