import base64
import json
import os
import random
import re
import shutil
import tarfile
import tempfile
import traceback
from contextlib import closing
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
import requests
import huggingface_hub as hf_hub
from PIL import Image
from modelscope.hub.api import HubApi as MSHubApi

from ..logger import get_logger

logger = get_logger(__name__)


IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
AUDIO_FILE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _calculate_files_to_download(
    dataset,
    source_format: str,
    media_per_archive: int,
    archives_per_dataset: int,
) -> int:
    """Calculate # files to download based on dataset modality and source format.
    Returns -1 to indicate "download all files", or a positive integer for the count.
    """
    src_fmt = source_format.lower().lstrip(".")

    # Direct media files (jpg, png, mp4, etc.) - download equivalent to archive extraction
    if dataset.modality == "image":
        is_direct_media = src_fmt in {ext.lstrip(".") for ext in IMAGE_FILE_EXTENSIONS}
    elif dataset.modality == "audio":
        is_direct_media = src_fmt in {ext.lstrip(".") for ext in AUDIO_FILE_EXTENSIONS}
    else:
        is_direct_media = src_fmt == "mp4"

    if is_direct_media:
        if media_per_archive == -1 or archives_per_dataset == -1:
            return -1
        return media_per_archive * archives_per_dataset

    # Archive-based media (parquet, zip, tar, etc.)
    return archives_per_dataset


def download_and_extract(
    dataset,  # BenchmarkDatasetConfig
    media_per_archive: int = 100,
    archives_per_dataset: int = 5,
    temp_dir: Optional[str] = None,
    force_download: bool = False,
    current_week_only: bool = False,
    num_weeks: int = None,
    cache_dir: str = "/.cache/gasbench",
    downloaded_archives: Optional[set] = None,
    target_week: Optional[str] = None,
    hf_token: Optional[str] = None,
    seed: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Download datasets and yield extracted media as a generator.

    Downloads files temporarily, extracts/processing content, then cleans up downloads.
    Processed content (videos, images) is cached persistently in other locations.

    Args:
        dataset: BenchmarkDatasetConfig object
        media_per_archive: Number of media items (images/videos) to extract per archive file
        archives_per_dataset: Number of archive files to download per dataset
        temp_dir: Temporary directory for downloads
        force_download: Force download even if dataset appears to be cached
        cache_dir: Base cache directory for persistent storage

    Yields:
        Complete sample dictionaries ready for processing functions
    """
    try:
        if not force_download and _is_dataset_cached(dataset, cache_dir):
            logger.info(
                f"Dataset {dataset.name} found in cache, loading from volume"
            )
            try:
                cached_samples_count = 0
                for sample in _load_dataset_from_cache(dataset, cache_dir):
                    cached_samples_count += 1
                    yield sample
                logger.info(
                    f"Successfully loaded {cached_samples_count} samples from cache for {dataset.name}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load {dataset.name} from cache: {e}")
                logger.info(
                    f"Fallback to download: {dataset.name} will be downloaded fresh"
                )
                # Fall through to download logic

        # Create temporary directory for downloads
        if temp_dir is not None:
            try:
                Path(temp_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        temp_dir_root = Path(tempfile.mkdtemp(dir=temp_dir))

        try:
            include_paths = getattr(dataset, "include_paths", None)
            exclude_paths = getattr(dataset, "exclude_paths", None)
            source = getattr(dataset, "source", "huggingface")

            filenames = _list_remote_dataset_files(
                dataset.path, dataset.source_format, current_week_only, num_weeks, target_week,
                include_paths, exclude_paths, source, hf_token
            )
            if not filenames:
                logger.warning(
                    f"No files found for {dataset.path} with format {dataset.source_format}"
                )

                src_fmt = str(getattr(dataset, "source_format", "")).lower().lstrip(".")
                fallback_formats = [".parquet", ".zip", ".tar", ".tar.gz"]
                for fallback_format in fallback_formats:
                    if fallback_format != dataset.source_format:
                        logger.info(
                            f"Trying fallback format {fallback_format} for {dataset.path}"
                        )
                        filenames = _list_remote_dataset_files(
                            dataset.path, fallback_format, current_week_only, num_weeks, target_week,
                            include_paths, exclude_paths, source, hf_token
                        )
                        if filenames:
                            logger.info(
                                f"Found {len(filenames)} files with format {fallback_format}"
                            )
                            break

                if not filenames:
                    logger.warning(f"No files found for {dataset.path} with any format")
                    return

            remote_paths = _get_download_urls(dataset.path, filenames, source)

            n_files = _calculate_files_to_download(
                dataset,
                dataset.source_format,
                media_per_archive,
                archives_per_dataset,
            )

            is_gasstation = "gasstation" in dataset.name.lower()
            to_download = _select_files_to_download(
                remote_paths, n_files, prioritize_recent=is_gasstation, seed=seed
            )
            
            if is_gasstation and downloaded_archives is not None:
                original_count = len(to_download)
                to_download = [
                    url for url in to_download 
                    if os.path.basename(url) not in downloaded_archives
                ]
                filtered_count = original_count - len(to_download)
                if filtered_count > 0:
                    logger.info(
                        f"Skipping {filtered_count} already-downloaded archives, "
                        f"downloading {len(to_download)} new archives"
                    )

            if len(to_download) == 0:
                return

            logger.info(
                f"Downloading {len(to_download)} files from {dataset.path} (dataset: {dataset.name})"
            )

            parquet_files_cache = None
            if dataset.modality == "video":
                try:
                    if source == "modelscope":
                        parquet_files_cache = list_modelscope_files(repo_id=dataset.path, extension=".parquet")
                    else:
                        parquet_files_cache = list_hf_files(repo_id=dataset.path, extension=".parquet", token=hf_token)
                    logger.info(f"Cached {len(parquet_files_cache)} parquet files for metadata lookup")
                except Exception as e:
                    logger.warning(f"Failed to cache parquet file list: {e}")
                    parquet_files_cache = []

            successfully_processed = 0
            for url in to_download:
                iso_week = _extract_iso_week_from_path(url)
                downloaded_file = download_single_file(url, temp_dir_root, 8192, hf_token)
                if not downloaded_file:
                    continue

                successfully_processed += 1
                try:
                    for sample in yield_media_from_source(
                        downloaded_file,
                        dataset,
                        media_per_archive,
                        iso_week,
                        hf_token,
                        seed,
                        parquet_files_cache
                    ):
                        yield sample
                finally:
                    try:
                        if downloaded_file.exists():
                            downloaded_file.unlink()
                    except Exception:
                        pass

            logger.info(
                f"Downloaded and processed {successfully_processed}/{len(to_download)} files "
                f"for {dataset.name}"
            )

        finally:
            if temp_dir_root.exists():
                shutil.rmtree(temp_dir_root)

    except Exception as e:
        logger.error(f"Error processing {dataset.path}: {e}")


def yield_media_from_source(
    source_path: Path,
    dataset,  # BenchmarkDatasetConfig
    num_items: int,
    iso_week: Optional[str] = None,
    hf_token: Optional[str] = None,
    seed: Optional[int] = None,
    parquet_files_cache: Optional[List[str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Unified media extractor for parquet, zip, and tar sources.

    Returns complete sample dictionaries ready for processing functions.
    Samples include: image/video_bytes, media_type, dataset_name, dataset_path, etc.
    """
    try:
        filename = str(source_path.name).lower()

        if _is_parquet_file(filename):
            yield from _process_parquet(source_path, dataset, num_items, iso_week, seed)
            return

        if _is_zip_file(filename) or _is_tar_file(filename):
            yield from _process_zip_or_tar(source_path, dataset, num_items, iso_week, hf_token, seed, parquet_files_cache)
            return

        if any(
            filename.endswith(ext)
            for ext in (IMAGE_FILE_EXTENSIONS | VIDEO_FILE_EXTENSIONS | AUDIO_FILE_EXTENSIONS)
        ):
            yield from _process_raw(source_path, dataset, iso_week)
            return

        logger.warning(f"Unsupported source format for {source_path}")
        return
    except Exception as e:
        logger.warning(f"Error in yield_media_from_source for {source_path}: {e}")
        return


def _extract_iso_week_from_path(file_path: str) -> Optional[str]:
    """Extract ISO week string from file path (e.g., '2025W40' from 'data_2025W40/file.parquet').

    Gasstation datasets are organized in weekly subdirectories like:
    - data_2025W38/
    - data_2025W40/
    - archives/2025W39/

    Returns:
        ISO week string like '2025W40', or None if not found
    """
    # Pattern to match ISO week format: YYYYWWW (e.g., 2025W40)
    pattern = r'(\d{4}W\d{2})'
    match = re.search(pattern, file_path)
    if match:
        return match.group(1)
    return None


def _select_files_to_download(urls: List[str], count: int, prioritize_recent: bool = False, seed: Optional[int] = None) -> List[str]:
    """Select files to download.

    Args:
        urls: List of file URLs to select from
        count: Number of files to select (-1 = all files)
        prioritize_recent: If True, sort by filename (descending) to get newest first.
                          Useful for gasstation datasets to get freshest data.
        seed: Random seed for reproducible sampling (non-gasstation only)

    Returns:
        List of selected URLs
    """
    if count == -1:
        return urls
    if count <= 0:
        return []
    
    if prioritize_recent:
        sorted_urls = sorted(urls, reverse=True)
        return sorted_urls[:min(count, len(urls))]
    
    if seed is not None:
        rng = random.Random(seed)
        return rng.sample(urls, min(count, len(urls)))
    return random.sample(urls, min(count, len(urls)))


def _list_remote_dataset_files(
    dataset_path: str, 
    source_format: str = ".parquet",
    current_week_only: bool = False,
    num_weeks: int = None,
    target_week: str = None,
    include_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    source: str = "huggingface",
    hf_token: Optional[str] = None,
) -> List[str]:
    """List available files in a dataset, filtered by source_format and path patterns.

    Supports single extensions (e.g., .parquet, .zip) and tar variants (.tar, .tar.gz, .tgz).
    For gasstation datasets with current_week_only=True, filters to only current ISO week's data.
    For gasstation datasets with num_weeks set, filters to only the N most recent weeks.
    
    Args:
        dataset_path: Dataset repository path (org/dataset-name)
        source_format: File extension to filter by
        current_week_only: For gasstation datasets, only get current week
        num_weeks: For gasstation datasets, get N most recent weeks
        target_week: For gasstation datasets, get specific week
        include_paths: Only include files containing one of these path segments
        exclude_paths: Exclude files containing any of these path segments
        source: Source platform ("huggingface" or "modelscope")
    """
    if not source_format.startswith("."):
        source_format = "." + source_format

    if source_format in [".tar", ".tar.gz", ".tgz"]:
        source_format = [".tar", ".tar.gz", ".tgz"]

    if source == "modelscope":
        files = list_modelscope_files(repo_id=dataset_path, extension=source_format)
    else:  # hf
        files = list_hf_files(repo_id=dataset_path, extension=source_format, token=hf_token)

    if include_paths:
        files = [f for f in files if any(path_seg in f for path_seg in include_paths)]

    if exclude_paths:
        files = [f for f in files if not any(path_seg in f for path_seg in exclude_paths)]

    if "gasstation" in dataset_path.lower():
        if target_week:
            files = [f for f in files if target_week in f]
            logger.info(f"Filtered to week {target_week} for {dataset_path}: {len(files)} files")
        elif num_weeks:
            files = _filter_files_by_recent_weeks(files, num_weeks)
            logger.info(f"Filtered to last {num_weeks} weeks for {dataset_path}: {len(files)} files")
        elif current_week_only:
            files = _filter_files_by_current_week(files)
            logger.info(f"Filtered to current week files for {dataset_path}: {len(files)} files")

    return files


def _filter_files_by_current_week(files: List[str]) -> List[str]:
    """Filter files to only include current ISO week's data for gasstation datasets.
    
    Gasstation datasets are organized in weekly subdirectories like:
    - data_2025W38/
    - data_2025W39/
    - data_2025W40/
    - archives/2025W38/
    - archives/2025W39/
    
    This function filters to only include files from the current ISO week.
    """
    from datetime import datetime
    
    now = datetime.now()
    current_year, current_week, _ = now.isocalendar()
    current_week_str = f"{current_year}W{current_week:02d}"
    logger.info(f"Current ISO week: {current_week_str}")
    
    current_week_files = []
    for file_path in files:
        # Check for patterns like data_2025W40/ or archives/2025W40/ or 2025W40 anywhere in path
        if current_week_str in file_path:
            current_week_files.append(file_path)
    
    logger.info(f"Found {len(current_week_files)} files for current week {current_week_str}")
    return current_week_files


def _filter_files_by_recent_weeks(files: List[str], num_weeks: int) -> List[str]:
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
        Filtered list of files
    """
    from datetime import datetime, timedelta
    
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
                break  # Only add file once even if it matches multiple weeks
    
    logger.info(f"Found {len(recent_week_files)} files for last {num_weeks} weeks")
    return recent_week_files


def _get_download_urls(
    dataset_path: str, 
    filenames: List[str], 
    source: str = "huggingface"
) -> List[str]:
    """Get download URLs for data files from the specified source.

    Args:
        dataset_path: Repository path (org/dataset-name)
        filenames: List of files to download
        source: Source platform ("huggingface" or "modelscope")

    Returns:
        List of download URLs
    """
    if source == "modelscope":
        return _get_modelscope_urls(dataset_path, filenames)
    else:
        return _get_huggingface_urls(dataset_path, filenames)


def _get_huggingface_urls(dataset_path: str, filenames: List[str]) -> List[str]:
    return [
        f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{f}"
        for f in filenames
    ]


def _get_modelscope_urls(dataset_path: str, filenames: List[str]) -> List[str]:
    return [
        f"https://www.modelscope.cn/api/v1/datasets/{dataset_path}/repo?Revision=master&FilePath={f}"
        for f in filenames
    ]


def _load_archive_metadata_map(dataset, archive_path: Path, iso_week: Optional[str] = None, hf_token: Optional[str] = None, parquet_files_cache: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Build a filename->metadata map for a given video archive using matching parquet shards.

    Matching strategy:
    - Use provided ISO week to filter metadata to same week
    - Strip archive suffix (e.g., .tar.gz, .tgz, .zip, .tar) to get the stem
    - Extract UID prefix (before timestamp) to match against parquet shards
    - Find parquets containing the week, UID prefix, and the word 'archive'
    - Download those small parquet files and collect metadata per filename column

    Note: Archives and their metadata parquets may have different timestamps,
    so we match on UID prefix only (e.g., '5EUQ8xz5' from '5EUQ8xz5_1760919909.tar.gz')
    """
    try:
        if not iso_week:
            return {}
        
        archive_stem = archive_path.name
        for suf in [".tar.gz", ".tgz", ".zip", ".tar"]:
            if archive_stem.endswith(suf):
                archive_stem = archive_stem[: -len(suf)]
                break

        parts = archive_stem.split("_")
        uid_prefix = parts[0] if parts else archive_stem
        timestamp = parts[1] if len(parts) > 1 else None
        alt_uid_prefix = uid_prefix.replace("_", "-")

        source = getattr(dataset, "source", "huggingface")

        # Use cached parquet files if available, otherwise fetch them (fallback for backwards compatibility)
        if parquet_files_cache is not None:
            parquet_files = parquet_files_cache
        else:
            if source == "modelscope":
                parquet_files = list_modelscope_files(repo_id=dataset.path, extension=".parquet")
            else:
                parquet_files = list_hf_files(repo_id=dataset.path, extension=".parquet", token=hf_token)

        # Match on week, UID, timestamp, 'archive'
        matching = [
            p for p in parquet_files 
            if iso_week in p 
            and alt_uid_prefix in p 
            and (timestamp in p if timestamp else True)
            and "archive" in p 
            and p.endswith(".parquet")
        ]

        if not matching:
            return {}

        temp_dir = Path(tempfile.mkdtemp())
        metadata_map: Dict[str, Dict[str, Any]] = {}
        try:
            urls = _get_download_urls(dataset.path, matching, source)
            for url in urls:
                pq_path = download_single_file(url, temp_dir, 8192, hf_token)
                if not pq_path:
                    continue
                try:
                    table = pq.read_table(pq_path)
                    df = table.to_pandas()
                    # Try to find a filename column (prioritize video_path, then more specific matches)
                    name_col = None
                    for keyword in ["video_path_in_archive", "video_path", "video_filename", "filename", "filepath", "file_path"]:
                        matching = [c for c in df.columns if keyword in str(c).lower()]
                        if matching:
                            name_col = matching[0]
                            break
                    for _, row in df.iterrows():
                        clean_meta = _extract_row_metadata(row, name_col or "")
                        if name_col:
                            key_name = str(row[name_col])
                            if key_name:
                                metadata_map[os.path.basename(key_name)] = clean_meta
                except Exception:
                    continue
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        return metadata_map
    except Exception:
        return {}


def _create_sample(
    dataset, media_obj, source_path: Path, iso_week: Optional[str] = None
) -> Dict[str, Any]:
    """Create a complete sample in the format expected by processing functions."""
    base_sample = {
        "media_type": dataset.media_type,
        "dataset_name": dataset.name,
        "dataset_path": dataset.path,
        "source_file": source_path.name,
    }

    # Add ISO week if available (for gasstation datasets)
    if iso_week:
        base_sample["iso_week"] = iso_week

    if dataset.modality == "image":
        base_sample["image"] = media_obj  # PIL Image
    elif dataset.modality == "audio":
        base_sample["audio_bytes"] = media_obj  # Raw audio bytes
    else:
        base_sample["video_bytes"] = media_obj  # Raw video bytes

    return base_sample


def _clean_to_json_serializable(value: Any) -> Any:
    """Convert arbitrary values to JSON-serializable equivalents.

    - numpy scalars -> native python
    - numpy arrays / lists / tuples / sets -> list of cleaned values
    - bytes -> base64 string
    - datetime -> isoformat string
    - dict -> recursively cleaned
    - NaN/Inf -> None
    - fallback -> str(value) if still not JSON-serializable
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
                _clean_to_json_serializable(v) for v in (value.tolist() if isinstance(value, np.ndarray) else list(value))
            ]

        # dict-like
        if isinstance(value, dict):
            return {str(k): _clean_to_json_serializable(v) for k, v in value.items()}

        # Fallback: ensure JSON-serializable; else stringify
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)
    except Exception:
        return None


def _extract_row_metadata(row: Any, media_col: str) -> Dict[str, Any]:
    """Extract non-media columns from a pandas Series row and clean to JSON-serializable dict."""
    metadata: Dict[str, Any] = {}
    try:
        for col, val in row.items():
            if str(col) == str(media_col):
                continue
            cleaned = _clean_to_json_serializable(val)
            col_str = str(col)
            metadata[col_str] = cleaned
            
            # Map common variations to standard field names for gasstation datasets
            col_lower = col_str.lower()
            if "hotkey" in col_lower and "generator" not in metadata:
                metadata["generator_hotkey"] = cleaned
            if "uid" in col_lower and "generator_uid" not in metadata:
                metadata["generator_uid"] = cleaned
    except Exception:
        # Best-effort extraction
        pass
    return metadata


def _process_parquet(
    source_path: Path, dataset, num_items: int, iso_week: Optional[str] = None, seed: Optional[int] = None
):  # BenchmarkDatasetConfig
    table = pq.read_table(source_path)
    df = table.to_pandas()
    if num_items == -1:
        sample_df = df
    else:
        sample_df = df.sample(n=min(num_items, len(df)), random_state=seed)

    if dataset.modality == "image":
        # First try exact match, then exclude _id columns, then any column with "image"
        media_col = (
            next((c for c in sample_df.columns if c.lower() == "image"), None)
            or next((c for c in sample_df.columns if "image" in c.lower() and "_id" not in c.lower()), None)
            or next((c for c in sample_df.columns if "image" in c.lower()), None)
        )
    elif dataset.modality == "audio":
        candidates = ["audio", "bytes", "content", "data", "wav", "mp3"]
        media_col = (
            next((c for c in sample_df.columns if c.lower() == "audio"), None)
            or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates) and "_id" not in c.lower()), None)
            or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates)), None)
        )
    else:
        candidates = ["video", "bytes", "content", "data"]
        # First try exact matches, then exclude _id columns, then fallback
        media_col = (
            next((c for c in sample_df.columns if c.lower() in candidates), None)
            or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates) and "_id" not in c.lower()), None)
            or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates)), None)
        )

    if not media_col:
        logger.warning(
            f"No media column found in {source_path} for modality {dataset.modality}"
        )
        return
    
    if "gasstation" in dataset.name.lower():
        cols = list(sample_df.columns)
        has_generator = any(
            "hotkey" in str(c).lower() or "generator" in str(c).lower()
            for c in cols
        )
        if not has_generator:
            logger.warning(f"Gasstation parquet {source_path.name} missing generator columns. Columns: {cols}")

    for _, row in sample_df.iterrows():
        try:
            media_data = row[media_col]
            if isinstance(media_data, dict):
                key = next(
                    (
                        k
                        for k in media_data
                        if any(
                            s in k.lower()
                            for s in ["bytes", "image", "video", "data", "content"]
                        )
                    ),
                    None,
                )
                media_data = media_data[key]

            if dataset.modality == "image":
                # Skip invalid media_data
                if media_data is None or isinstance(media_data, (int, float)):
                    continue

                try:
                    img = Image.open(BytesIO(media_data))
                except Exception:
                    if isinstance(media_data, str):
                        media_data = base64.b64decode(media_data)
                    img = Image.open(BytesIO(media_data))
                sample = _create_sample(dataset, img, source_path, iso_week)
            else:
                if media_data is None or isinstance(media_data, (int, float)):
                    continue

                if not isinstance(media_data, (bytes, bytearray)):
                    if isinstance(media_data, str):
                        media_data = base64.b64decode(media_data)
                    else:
                        continue
                sample = _create_sample(dataset, bytes(media_data), source_path, iso_week)

            # Merge parquet row metadata without overwriting base fields
            row_metadata = _extract_row_metadata(row, media_col)
            for k, v in row_metadata.items():
                if k not in sample:
                    sample[k] = v
            
            # Debug: Log extracted generator metadata for first few samples
            if "gasstation" in dataset.name.lower():
                gen_hotkey = sample.get("generator_hotkey")
                gen_uid = sample.get("generator_uid")
                if gen_hotkey and gen_hotkey != "unknown":
                    logger.debug(f"✅ Extracted generator metadata: hotkey={gen_hotkey[:8] if isinstance(gen_hotkey, str) else gen_hotkey}..., uid={gen_uid}")

            yield sample
        except Exception as e:
            logger.warning(f"Failed to extract row from {source_path}: {e}")
            continue


def _process_zip_or_tar(
    source_path: Path, dataset, num_items: int, iso_week: Optional[str] = None, hf_token: Optional[str] = None, seed: Optional[int] = None, parquet_files_cache: Optional[List[str]] = None
):  # BenchmarkDatasetConfig
    filename = str(source_path.name).lower()
    is_zip = _is_zip_file(filename)
    try:
        cm = ZipFile(source_path) if is_zip else tarfile.open(source_path, mode="r:*")
        with cm as archive:
            if dataset.modality == "image":
                valid_exts = IMAGE_FILE_EXTENSIONS
            elif dataset.modality == "audio":
                valid_exts = AUDIO_FILE_EXTENSIONS
            else:
                valid_exts = VIDEO_FILE_EXTENSIONS

            if is_zip:
                list_entries = archive.namelist()
                get_name = lambda e: e

                def open_entry(e):
                    return archive.open(e)

            else:
                list_entries = [m for m in archive.getmembers() if m.isreg()]
                get_name = lambda m: m.name

                def open_entry(m):
                    return archive.extractfile(m)

            candidates = [
                e
                for e in list_entries
                if any(get_name(e).lower().endswith(ext) for ext in valid_exts)
                and "MACOSX" not in get_name(e)
            ]
            if not candidates:
                logger.warning(f"No matching files found in {source_path}")
                return

            if num_items == -1:
                selected = candidates
            else:
                if seed is not None:
                    rng = random.Random(seed)
                    selected = rng.sample(candidates, min(num_items, len(candidates)))
                else:
                    selected = random.sample(candidates, min(num_items, len(candidates)))

            # Load associated metadata parquet(s) for video archives if available
            archive_metadata_map: Dict[str, Dict[str, Any]] = (
                _load_archive_metadata_map(dataset, source_path, iso_week, hf_token, parquet_files_cache)
                if dataset.modality == "video"
                else {}
            )

            for entry in selected:
                try:
                    src = open_entry(entry)
                    if src is None:
                        continue
                    with closing(src):
                        data_bytes = src.read()

                    if dataset.modality == "image":
                        try:
                            media_obj = Image.open(BytesIO(data_bytes))
                        except Exception:
                            logger.warning(
                                f"Failed to open image {get_name(entry)} from {source_path}"
                            )
                            continue
                    else:
                        media_obj = data_bytes

                    sample = _create_sample(dataset, media_obj, source_path, iso_week)
                    # Attach archive filename and row metadata if found
                    sample["archive_filename"] = source_path.name
                    try:
                        entry_name = os.path.basename(get_name(entry))
                        meta = archive_metadata_map.get(entry_name)
                        if meta:
                            for k, v in meta.items():
                                if k not in sample:
                                    sample[k] = v
                    except Exception:
                        pass

                    yield sample
                except Exception as e:
                    logger.warning(
                        f"Error extracting {get_name(entry)} from {source_path}: {e}"
                    )
                    continue
    except Exception as e:
        logger.warning(f"Error processing archive file {source_path}: {e}")
        return


def _process_raw(source_path: Path, dataset, iso_week: Optional[str] = None):
    filename = str(source_path.name).lower()
    try:
        data_bytes = source_path.read_bytes()
        if dataset.modality == "image" and any(
            filename.endswith(ext) for ext in IMAGE_FILE_EXTENSIONS
        ):
            media_obj = Image.open(BytesIO(data_bytes))
        elif dataset.modality == "audio" and any(
            filename.endswith(ext) for ext in AUDIO_FILE_EXTENSIONS
        ):
            media_obj = data_bytes
        elif dataset.modality == "video" and any(
            filename.endswith(ext) for ext in VIDEO_FILE_EXTENSIONS
        ):
            media_obj = data_bytes
        else:
            logger.warning(
                f"Direct file {source_path} does not match modality {dataset.modality}"
            )
            return

        yield _create_sample(dataset, media_obj, source_path, iso_week)
    except Exception as e:
        logger.warning(f"Error reading direct file {source_path}: {e}")
        return


def list_hf_files(repo_id, repo_type="dataset", extension=None, token=None):
    """List files from a Hugging Face repository.

    Args:
        repo_id: Repository ID
        repo_type: Type of repository ('dataset', 'model', etc.)
        extension: Filter files by extension
        token: Hugging Face API token for private datasets

    Returns:
        List of files in the repository
    """
    files = []
    try:
        files = list(hf_hub.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token))
        if extension:
            if isinstance(extension, (list, tuple, set)):
                exts = tuple(extension)
                files = [f for f in files if f.endswith(exts)]
            else:
                files = [f for f in files if f.endswith(extension)]
    except Exception as e:
        logger.error(f"Failed to list files of type {extension} in {repo_id}: {e}")
    return files


def list_modelscope_files(repo_id, extension=None):
    """List files from a ModelScope dataset repository.

    Args:
        repo_id: Repository ID (format: org/dataset-name)
        extension: Filter files by extension(s)

    Returns:
        List of files in the repository
    """
    files = []
    try:
        api = MSHubApi()
        file_info = api.get_dataset_files(repo_id, revision='master', recursive=True)
        
        # Extract file paths
        if isinstance(file_info, list):
            files = [f['Path'] if isinstance(f, dict) and 'Path' in f else str(f) for f in file_info]
        elif isinstance(file_info, dict):
            files = [f for f in file_info.keys()] if file_info else []

        if extension and files:
            if isinstance(extension, (list, tuple, set)):
                exts = tuple(extension)
                files = [f for f in files if f.endswith(exts)]
            else:
                files = [f for f in files if f.endswith(extension)]

    except Exception as e:
        logger.error(f"Failed to list files of type {extension} in ModelScope repo {repo_id}: {e}")
        logger.error(f"Make sure 'modelscope' package is installed: pip install modelscope")
    
    return files


def _is_zip_file(filename_lower: str) -> bool:
    """Return True if filename looks like a zip archive."""
    return filename_lower.endswith(".zip")


def _is_tar_file(filename_lower: str) -> bool:
    """Return True if filename looks like a tar archive (.tar, .tar.gz, .tgz)."""
    return (
        filename_lower.endswith(".tar")
        or filename_lower.endswith(".tar.gz")
        or filename_lower.endswith(".tgz")
    )


def _is_parquet_file(filename_lower: str) -> bool:
    """Return True if filename looks like a parquet file."""
    return filename_lower.endswith(".parquet")


def download_files(
    urls: List[str], output_dir: Path, chunk_size: int = 8192
) -> List[Path]:
    """Download multiple files synchronously.

    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        chunk_size: Size of chunks to download at a time

    Returns:
        List of successfully downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for url in urls:
        try:
            downloaded_file = download_single_file(url, output_dir, chunk_size)
            if downloaded_file:
                downloaded_files.append(downloaded_file)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")

    return downloaded_files


def download_single_file(
    url: str, 
    output_dir: Path, 
    chunk_size: int, 
    hf_token: Optional[str] = None
) -> Optional[Path]:
    """Download a single file synchronously

    Args:
        url: URL to download
        output_dir: Directory to save the file
        chunk_size: Size of chunks to download at a time
        hf_token: Hugging Face API token for private datasets

    Returns:
        Path to the downloaded file, or None if failed
    """
    try:
        filename = os.path.basename(url)
        filepath = output_dir / filename

        logger.info(f"Downloading {url}")
        headers = {}
        if hf_token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_token}"

        response = requests.get(url, stream=True, timeout=3600, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to download {url}: Status {response.status_code}")
            return None

        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Writing to {filepath} (size: {total_size/(1024*1024):.1f} MB)")

        downloaded = 0
        last_log_mb = 0
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=max(chunk_size, 1024*1024)):  # Use at least 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    current_mb = downloaded / (1024 * 1024)
                    if current_mb - last_log_mb >= 100:
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            logger.info(f"Progress: {current_mb:.0f}MB / {total_size/(1024*1024):.0f}MB ({pct:.1f}%)")
                        else:
                            logger.info(f"Downloaded: {current_mb:.0f}MB")
                        last_log_mb = current_mb

        # Verify download completeness
        if total_size > 0:
            actual_size = filepath.stat().st_size
            if actual_size != total_size:
                logger.error(f"❌ Download incomplete: expected {total_size} bytes, got {actual_size} bytes")
                return None
            logger.info(f"✅ Download verified: {filename} ({actual_size/(1024*1024):.1f} MB)")
        else:
            logger.info(f"✅ Downloaded: {filename}")

        return filepath

    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


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
    except Exception as e:
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
                    **metadata,  # Include all original metadata
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
                # Load video file as bytes
                with open(filepath, "rb") as f:
                    video_bytes = f.read()

                sample = {
                    "video_bytes": video_bytes,
                    "media_type": dataset.media_type,
                    "dataset_name": dataset.name,
                    "dataset_path": dataset.path,
                    "source_file": f"cached_{filename}",
                    **metadata,  # Include all original metadata
                }
                yield sample

        except Exception as e:
            logger.warning(
                f"Failed to load cached sample {filename} from {dataset.name}: {e}"
            )
            continue
