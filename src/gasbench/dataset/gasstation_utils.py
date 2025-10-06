"""Utilities for managing gasstation dataset caching with ISO week-based organization.

This module handles the special caching logic for gasstation datasets:
- ISO week-based hierarchical directory structure
- Per-week archive tracking to avoid re-downloads
- Detection of new archives for incremental updates
"""

import os
import json
from typing import List, Set, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from ..logger import get_logger

logger = get_logger(__name__)


def calculate_target_weeks(num_weeks: Optional[int] = None) -> List[str]:
    """Calculate target ISO weeks for download.
    
    Args:
        num_weeks: Number of recent weeks to include (None = current week only)
        
    Returns:
        List of ISO week strings (e.g., ["2025W39", "2025W40"])
    """
    target_weeks = []
    now = datetime.now()
    
    if num_weeks:
        for i in range(num_weeks):
            date_offset = now - timedelta(weeks=i)
            year, week, _ = date_offset.isocalendar()
            week_str = f"{year}W{week:02d}"
            target_weeks.append(week_str)
    else:
        # Default to current week only
        year, week, _ = now.isocalendar()
        week_str = f"{year}W{week:02d}"
        target_weeks = [week_str]
    
    target_weeks.sort()
    return target_weeks


def get_week_directories(dataset_name: str, cache_dir: str, target_weeks: List[str]) -> List[str]:
    """Get week-specific cache directory paths.
    
    Args:
        dataset_name: Name of the dataset
        cache_dir: Base cache directory
        target_weeks: List of ISO week strings
        
    Returns:
        List of week directory paths (e.g., [".../gasstation-generated-images/2025W40/"])
    """
    base_dir = f"{cache_dir}/datasets/{dataset_name}"
    return [f"{base_dir}/{week}" for week in target_weeks]


def load_downloaded_archives(week_dir: str) -> Set[str]:
    """Load the set of already-downloaded archives for a week.
    
    Args:
        week_dir: Path to the week's cache directory
        
    Returns:
        Set of downloaded archive basenames
    """
    archive_tracker_file = os.path.join(week_dir, "downloaded_archives.json")
    
    if not os.path.exists(archive_tracker_file):
        return set()
    
    try:
        with open(archive_tracker_file, "r") as f:
            return set(json.load(f))
    except Exception as e:
        logger.warning(f"Failed to load archive tracking: {e}")
        return set()


def save_downloaded_archives(week_dir: str, downloaded_archives: Set[str]):
    """Save the set of downloaded archives for a week.
    
    Args:
        week_dir: Path to the week's cache directory
        downloaded_archives: Set of downloaded archive basenames
    """
    archive_tracker_file = os.path.join(week_dir, "downloaded_archives.json")
    
    try:
        Path(week_dir).mkdir(parents=True, exist_ok=True)
        with open(archive_tracker_file, "w") as f:
            json.dump(list(downloaded_archives), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save archive tracking: {e}")


def check_for_new_archives(
    dataset_path: str,
    week_str: str,
    source_format: str,
    modality: str,
    downloaded_archives: Set[str]
) -> Tuple[bool, int]:
    """Check if there are new archives available for a week.
    
    Args:
        dataset_path: HuggingFace dataset path
        week_str: ISO week string (e.g., "2025W40")
        source_format: Source file format
        modality: Dataset modality ("image" or "video")
        downloaded_archives: Set of already-downloaded archive basenames
        
    Returns:
        Tuple of (has_new_archives, num_new_archives)
    """
    try:
        from .download import list_hf_files
        
        # Determine source format
        src_fmt = str(source_format).lower().lstrip(".")
        if not src_fmt:
            src_fmt = ".parquet" if modality == "image" else ".zip"
        else:
            src_fmt = "." + src_fmt if not src_fmt.startswith(".") else src_fmt
        
        # List files for this specific week
        all_files = list_hf_files(repo_id=dataset_path, extension=src_fmt)
        week_files = [f for f in all_files if week_str in f]
        
        # Check for new archives
        available_basenames = {os.path.basename(f) for f in week_files}
        new_archives = available_basenames - downloaded_archives
        
        return (len(new_archives) > 0, len(new_archives))
        
    except Exception as e:
        logger.debug(f"Failed to check for new archives: {e}")
        return (False, 0)


def is_week_cache_complete(
    week_dir: str,
    dataset_path: str,
    week_str: str,
    source_format: str,
    modality: str
) -> bool:
    """Check if a week's cache is complete (has data and no new archives available).
    
    Args:
        week_dir: Path to the week's cache directory
        dataset_path: HuggingFace dataset path
        week_str: ISO week string (e.g., "2025W40")
        source_format: Source file format
        modality: Dataset modality ("image" or "video")
        
    Returns:
        True if cache is complete, False if new data should be downloaded
    """
    metadata_file = os.path.join(week_dir, "sample_metadata.json")
    
    if not os.path.exists(metadata_file):
        return False
    
    # Load downloaded archives
    downloaded_archives = load_downloaded_archives(week_dir)
    if not downloaded_archives:
        return False
    
    has_new, num_new = check_for_new_archives(
        dataset_path, week_str, source_format, modality, downloaded_archives
    )
    
    if has_new:
        logger.info(
            f"🆕 Found {num_new} new archives for week {week_str} "
            f"(already have {len(downloaded_archives)} archived)"
        )
        return False
    
    return True


def get_week_sample_count(week_dir: str) -> int:
    """Get the number of samples cached for a week.
    
    Args:
        week_dir: Path to the week's cache directory
        
    Returns:
        Number of cached samples (0 if none)
    """
    metadata_file = os.path.join(week_dir, "sample_metadata.json")
    
    if not os.path.exists(metadata_file):
        return 0
    
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return len(metadata)
    except Exception:
        return 0


def get_total_cached_samples(week_dirs: List[str]) -> int:
    """Get total number of samples cached across all weeks.
    
    Args:
        week_dirs: List of week directory paths
        
    Returns:
        Total number of cached samples
    """
    total = 0
    for week_dir in week_dirs:
        total += get_week_sample_count(week_dir)
    return total
