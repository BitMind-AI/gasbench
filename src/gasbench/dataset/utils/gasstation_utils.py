"""Utilities for managing gasstation dataset caching with ISO week-based organization.

This module handles the special caching logic for gasstation datasets:
- ISO week-based hierarchical directory structure
- Per-week archive tracking to avoid re-downloads
- Detection of new archives for incremental updates
- Automatic fallback to previous week on first day of week (Monday)
"""

import os
import json
from typing import List, Set, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from ...logger import get_logger

logger = get_logger(__name__)


def calculate_target_weeks(
    num_weeks: Optional[int] = None,
    dataset_path: Optional[str] = None,
    source_format: Optional[str] = None,
    modality: Optional[str] = None,
) -> List[str]:
    """Calculate target ISO weeks for download.
    
    If the current week has no data available on HuggingFace, automatically
    falls back to include the previous week(s) until data is found.
    
    Args:
        num_weeks: Number of recent weeks to include (None = auto-detect)
        dataset_path: HuggingFace dataset path (for checking data availability)
        source_format: Source file format (e.g., "parquet")
        modality: Dataset modality ("image" or "video")
        
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
        current_year, current_week, _ = now.isocalendar()
        current_week_str = f"{current_year}W{current_week:02d}"
        target_weeks = [current_week_str]
        
        # Check if current week has data, if not fall back to previous weeks
        if dataset_path and source_format and modality:
            if not _week_has_data(dataset_path, current_week_str, source_format, modality):
                logger.info(
                    f"No data found for current week {current_week_str}, "
                    f"checking previous weeks..."
                )
                # Look back up to 4 weeks to find data
                for i in range(1, 5):
                    prev_date = now - timedelta(weeks=i)
                    prev_year, prev_week, _ = prev_date.isocalendar()
                    prev_week_str = f"{prev_year}W{prev_week:02d}"
                    
                    if _week_has_data(dataset_path, prev_week_str, source_format, modality):
                        target_weeks.append(prev_week_str)
                        logger.info(
                            f"Found data in week {prev_week_str}, "
                            f"including it along with current week {current_week_str}"
                        )
                        break
                    else:
                        logger.debug(f"No data in week {prev_week_str} either, continuing...")
    
    target_weeks.sort()
    return target_weeks


def _week_has_data(
    dataset_path: str,
    week_str: str,
    source_format: str,
    modality: str,
) -> bool:
    """Check if a week has any data files available on HuggingFace.
    
    Args:
        dataset_path: HuggingFace dataset path
        week_str: ISO week string (e.g., "2025W40")
        source_format: Source file format (e.g., "parquet")
        modality: Dataset modality ("image" or "video")
        
    Returns:
        True if files exist for this week, False otherwise
    """
    try:
        from ..download import list_hf_files
        
        src_fmt = str(source_format).lower().lstrip(".")
        if not src_fmt:
            src_fmt = ".parquet" if modality == "image" else ".zip"
        else:
            src_fmt = "." + src_fmt if not src_fmt.startswith(".") else src_fmt
        
        all_files = list_hf_files(repo_id=dataset_path, extension=src_fmt)
        week_files = [f for f in all_files if week_str in f]
        
        return len(week_files) > 0
        
    except Exception as e:
        logger.debug(f"Failed to check week data availability: {e}")
        # If we can't check, assume data exists to avoid blocking
        return True


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
    """Load the set of already-processed parquet/archive files for a week.
    
    For parquet-based gasstation datasets, this tracks parquet filenames.
    For legacy datasets, this tracks archive filenames.
    
    Args:
        week_dir: Path to the week's cache directory
        
    Returns:
        Set of downloaded parquet/archive basenames
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
    """Save the set of processed parquet/archive files for a week.
    
    For parquet-based gasstation datasets, this tracks parquet filenames.
    For legacy datasets, this tracks archive filenames.
    
    Args:
        week_dir: Path to the week's cache directory
        downloaded_archives: Set of processed parquet/archive basenames
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
    
    For parquet-based gasstation datasets, checks for new parquet files
    (which inherently means new tar archives are available, as they're uploaded together).
    
    Args:
        dataset_path: HuggingFace dataset path
        week_str: ISO week string (e.g., "2025W40")
        source_format: Source file format (e.g., "parquet", "tar.gz")
        modality: Dataset modality ("image" or "video")
        downloaded_archives: Set of already-downloaded parquet/archive basenames
        
    Returns:
        Tuple of (has_new_archives, num_new_archives)
    """
    try:
        from ..download import list_hf_files
        
        src_fmt = str(source_format).lower().lstrip(".")
        if not src_fmt:
            src_fmt = ".parquet" if modality == "image" else ".zip"
        else:
            src_fmt = "." + src_fmt if not src_fmt.startswith(".") else src_fmt
        
        all_files = list_hf_files(repo_id=dataset_path, extension=src_fmt)
        week_files = [f for f in all_files if week_str in f]
        
        available_basenames = {os.path.basename(f) for f in week_files}
        new_files = available_basenames - downloaded_archives
        
        return (len(new_files) > 0, len(new_files))
        
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
            f"Found {num_new} new archives for week {week_str} "
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


def extract_iso_week_from_path(file_path: str) -> Optional[str]:
    """Extract ISO week string from file path (e.g., '2025W40' from 'data_2025W40/file.parquet').
    
    Gasstation datasets are organized in weekly subdirectories like:
    - data_2025W38/
    - data_2025W40/
    - archives/2025W39/
    
    Args:
        file_path: Path containing ISO week pattern
    
    Returns:
        ISO week string like '2025W40', or None if not found
    """
    import re
    
    pattern = r'(\d{4}W\d{2})'
    match = re.search(pattern, file_path)
    if match:
        return match.group(1)
    return None


def filter_files_by_current_week(files: List[str]) -> List[str]:
    """Filter files to only include current ISO week's data for gasstation datasets.
    
    Gasstation datasets are organized in weekly subdirectories like:
    - data_2025W38/
    - data_2025W39/
    - data_2025W40/
    - archives/2025W38/
    - archives/2025W39/
    
    This function filters to only include files from the current ISO week.
    
    Args:
        files: List of file paths to filter
        
    Returns:
        Filtered list of files from current week
    """
    now = datetime.now()
    current_year, current_week, _ = now.isocalendar()
    current_week_str = f"{current_year}W{current_week:02d}"
    logger.info(f"Current ISO week: {current_week_str}")
    
    current_week_files = []
    for file_path in files:
        if current_week_str in file_path:
            current_week_files.append(file_path)
    
    logger.info(f"Found {len(current_week_files)} files for current week {current_week_str}")
    return current_week_files


def filter_files_by_recent_weeks(files: List[str], num_weeks: int) -> List[str]:
    """Filter files to only include the N most recent ISO weeks for gasstation datasets.
    
    Gasstation datasets are organized in weekly subdirectories like:
    - data_2025W38/
    - data_2025W39/
    - data_2025W40/
    - archives/2025W38/
    - archives/2025W39/
    
    This function filters to only include files from the N most recent ISO weeks.
    
    Args:
        files: List of file paths
        num_weeks: Number of most recent weeks to include
    
    Returns:
        Filtered list of files from recent weeks
    """
    now = datetime.now()
    
    recent_weeks = []
    for i in range(num_weeks):
        date_offset = now - timedelta(weeks=i)
        year, week, _ = date_offset.isocalendar()
        week_str = f"{year}W{week:02d}"
        recent_weeks.append(week_str)
    
    logger.info(f"Filtering to last {num_weeks} weeks: {', '.join(recent_weeks)}")
    
    recent_week_files = []
    for file_path in files:
        for week_str in recent_weeks:
            if week_str in file_path:
                recent_week_files.append(file_path)
                break
    
    logger.info(f"Found {len(recent_week_files)} files for last {num_weeks} weeks")
    return recent_week_files
