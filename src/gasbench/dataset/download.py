import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import tempfile
import traceback
import base64
from contextlib import closing
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from zipfile import ZipFile
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
import requests
import huggingface_hub as hf_hub
import numpy as np
from PIL import Image
from modelscope.hub.api import HubApi as MSHubApi

from ..logger import get_logger
from .utils.s3_utils import list_s3_files, download_s3_file, _get_s3_urls
from .utils.metadata_utils import create_sample, extract_row_metadata
from .utils.gasstation_utils import (
    extract_iso_week_from_path,
    filter_files_by_current_week,
    filter_files_by_recent_weeks,
)

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
            logger.info(f"Dataset {dataset.name} found in cache, loading from volume")
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
                dataset.path,
                dataset.source_format,
                current_week_only,
                num_weeks,
                target_week,
                include_paths,
                exclude_paths,
                source,
                hf_token,
            )
            if not filenames:
                is_gasstation = "gasstation" in dataset.name.lower()
                if is_gasstation:
                    logger.warning(
                        f"No files found for {dataset.path} with format {dataset.source_format}. "
                        f"Gasstation datasets require parquet metadata files."
                    )
                    return
                
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
                            dataset.path,
                            fallback_format,
                            current_week_only,
                            num_weeks,
                            target_week,
                            include_paths,
                            exclude_paths,
                            source,
                            hf_token,
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
                    url
                    for url in to_download
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

            if (
                is_gasstation
                and to_download
                and any(".parquet" in url for url in to_download)
            ):
                yield from _process_gasstation(
                    dataset=dataset,
                    to_download=to_download,
                    temp_dir_root=temp_dir_root,
                    media_per_archive=media_per_archive,
                    downloaded_archives=downloaded_archives,
                    source=source,
                    hf_token=hf_token,
                    seed=seed,
                )
            else:
                max_download_workers = min(10, len(to_download))

                downloaded_files = download_files(
                    to_download,
                    temp_dir_root,
                    chunk_size=8192,
                    max_workers=max_download_workers,
                    hf_token=hf_token,
                )

                if not downloaded_files:
                    logger.warning(
                        f"No files successfully downloaded for {dataset.name}"
                    )
                    return

                successfully_processed = 0
                for downloaded_file in downloaded_files:
                    if not downloaded_file or not downloaded_file.exists():
                        continue

                    try:
                        iso_week = extract_iso_week_from_path(str(downloaded_file))
                        successfully_processed += 1

                        for sample in yield_media_from_source(
                            downloaded_file,
                            dataset,
                            media_per_archive,
                            iso_week,
                            hf_token,
                            seed,
                        ):
                            yield sample
                    finally:
                        try:
                            if downloaded_file.exists():
                                downloaded_file.unlink()
                        except Exception:
                            pass

                logger.info(
                    f"Downloaded and processed {successfully_processed}/{len(downloaded_files)} files "
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
) -> Generator[Dict[str, Any], None, None]:
    """
    Unified media extractor for parquet, zip, tar sources, frame directories, and raw media files.

    Returns complete sample dictionaries ready for processing functions.
    Samples include: image/video_bytes, media_type, dataset_name, dataset_path, etc.
    """
    try:
        if source_path.is_dir():
            yield from _process_frame_directory(source_path, dataset, iso_week)
            return

        filename = str(source_path.name).lower()

        if _is_parquet_file(filename):
            yield from _process_parquet(source_path, dataset, num_items, iso_week, seed)
            return

        if _is_zip_file(filename) or _is_tar_file(filename):
            yield from _process_zip_or_tar(
                source_path,
                dataset,
                num_items,
                iso_week,
                seed,
            )
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


def _select_files_to_download(
    urls: List[str],
    count: int,
    prioritize_recent: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
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
        return sorted_urls[: min(count, len(urls))]

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
    # Don't add "." prefix for special formats like "frames"
    if source_format != "frames" and not source_format.startswith("."):
        source_format = "." + source_format

    if source_format in [".tar", ".tar.gz", ".tgz"]:
        source_format = [".tar", ".tar.gz", ".tgz"]

    if source == "modelscope":
        files = list_modelscope_files(repo_id=dataset_path, extension=source_format)
    elif source == "s3":
        files = list_s3_files(path=dataset_path, extension=source_format)
    else:  # hf
        files = list_hf_files(
            repo_id=dataset_path, extension=source_format, token=hf_token
        )

    if include_paths:
        files = [f for f in files if any(path_seg in f for path_seg in include_paths)]

    if exclude_paths:
        files = [
            f for f in files if not any(path_seg in f for path_seg in exclude_paths)
        ]

    if "gasstation" in dataset_path.lower():
        if target_week:
            files = [f for f in files if target_week in f]
            logger.info(
                f"Filtered to week {target_week} for {dataset_path}: {len(files)} files"
            )
        elif num_weeks:
            files = filter_files_by_recent_weeks(files, num_weeks)
            logger.info(
                f"Filtered to last {num_weeks} weeks for {dataset_path}: {len(files)} files"
            )
        elif current_week_only:
            files = filter_files_by_current_week(files)
            logger.info(
                f"Filtered to current week files for {dataset_path}: {len(files)} files"
            )

    return files


def _get_download_urls(
    dataset_path: str, filenames: List[str], source: str = "huggingface"
) -> List[str]:
    """Get download URLs for data files from the specified source.

    Args:
        dataset_path: Repository path (org/dataset-name) or S3 path (bucket/prefix)
        filenames: List of files to download
        source: Source platform ("huggingface", "modelscope", or "s3")

    Returns:
        List of download URLs
    """
    if source == "modelscope":
        return _get_modelscope_urls(dataset_path, filenames)
    elif source == "s3":
        return _get_s3_urls(dataset_path, filenames)
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


def _process_gasstation(
    dataset,
    to_download: List[str],
    temp_dir_root: Path,
    media_per_archive: int,
    downloaded_archives: Optional[set],
    source: str,
    hf_token: Optional[str],
    seed: Optional[int],
) -> Generator[Dict[str, Any], None, None]:
    """Process gasstation datasets.

    Steps:
    1. Download parquet metadata files
    2. Extract unique archive filenames from parquet files
    3. Download those tar archives
    4. Extract media from archives using metadata mapping

    Args:
        dataset: BenchmarkDatasetConfig
        to_download: List of URLs to download
        temp_dir_root: Temporary directory for downloads
        media_per_archive: Number of items to extract per archive
        downloaded_archives: Set of already-processed parquet basenames
        source: Source platform (huggingface, modelscope, s3)
        hf_token: HuggingFace API token
        seed: Random seed for reproducibility

    Yields:
        Sample dictionaries with media and metadata
    """
    parquet_urls = [url for url in to_download if ".parquet" in url]

    if not parquet_urls:
        logger.warning(f"No parquet files found in download list for {dataset.name}")
        return

    # Track parquet files we've already processed
    if downloaded_archives is not None:
        parquet_basenames = {os.path.basename(url) for url in parquet_urls}
        parquets_to_download = [
            url
            for url in parquet_urls
            if os.path.basename(url) not in downloaded_archives
        ]

        skipped = len(parquet_urls) - len(parquets_to_download)
        if skipped > 0:
            logger.info(
                f"Skipping {skipped} already-processed parquet files, "
                f"downloading {len(parquets_to_download)} new parquet files"
            )
        parquet_urls = parquets_to_download

    if not parquet_urls:
        logger.info("No new parquet files to download")
        return

    logger.info(f"Downloading {len(parquet_urls)} parquet metadata files")

    max_download_workers = min(10, len(parquet_urls))
    downloaded_parquets = download_files(
        parquet_urls,
        temp_dir_root,
        chunk_size=8192,
        max_workers=max_download_workers,
        hf_token=hf_token,
    )

    if not downloaded_parquets:
        logger.warning(f"Failed to download parquet files for {dataset.name}")
        return

    logger.info(
        f"Extracting archive filenames from {len(downloaded_parquets)} parquet files"
    )

    all_archive_filenames = set()
    parquet_metadata_maps = {}
    processed_parquet_basenames = []

    for parquet_path in downloaded_parquets:
        if not parquet_path or not parquet_path.exists():
            continue

        try:
            processed_parquet_basenames.append(parquet_path.name)

            archive_filenames = _extract_unique_archive_filenames(parquet_path)
            all_archive_filenames.update(archive_filenames)

            metadata_map = _build_parquet_metadata_map(parquet_path)
            for key, value in metadata_map.items():
                parquet_metadata_maps[key] = value

            iso_week = extract_iso_week_from_path(str(parquet_path))
            if iso_week:
                for key in metadata_map.keys():
                    if key in parquet_metadata_maps:
                        parquet_metadata_maps[key]["iso_week"] = iso_week
        except Exception as e:
            logger.warning(f"Failed to process parquet {parquet_path}: {e}")
            continue

    logger.info(
        f"Found {len(all_archive_filenames)} unique archive files referenced in parquet metadata"
    )
    logger.info(f"Built metadata map with {len(parquet_metadata_maps)} entries")

    if not all_archive_filenames:
        logger.warning("No archive filenames found in parquet metadata")
        return

    # Get ISO week from original parquet URLs (not temp download paths)
    iso_week = None
    for url in parquet_urls:
        iso_week = extract_iso_week_from_path(url)
        if iso_week:
            break
    
    if not iso_week:
        logger.warning("Could not determine ISO week for archive paths from parquet URLs")
        return
    
    # Prepend archives/{week}/ to basenames
    archive_paths = [f"archives/{iso_week}/{basename}" for basename in all_archive_filenames]
    
    logger.info(f"Downloading {len(archive_paths)} tar archives from archives/{iso_week}/")

    archive_urls = _get_download_urls(dataset.path, archive_paths, source)

    max_workers = min(10, len(archive_urls))
    downloaded_archive_files = download_files(
        archive_urls,
        temp_dir_root,
        chunk_size=8192,
        max_workers=max_workers,
        hf_token=hf_token,
    )

    if not downloaded_archive_files:
        logger.warning(f"Failed to download archive files for {dataset.name}")
        return

    logger.info(f"Extracting media from {len(downloaded_archive_files)} archives")

    total_samples = 0
    for archive_path in downloaded_archive_files:
        if not archive_path or not archive_path.exists():
            continue

        try:
            iso_week = extract_iso_week_from_path(str(archive_path))

            for sample in _process_tar_with_metadata(
                archive_path,
                dataset,
                parquet_metadata_maps,
                media_per_archive,
                iso_week,
                seed,
            ):
                total_samples += 1
                yield sample
        except Exception as e:
            logger.warning(f"Failed to process archive {archive_path}: {e}")
            continue
        finally:
            try:
                if archive_path.exists():
                    archive_path.unlink()
            except Exception:
                pass

    # Track the processed parquet files (not tar archives)
    if downloaded_archives is not None:
        for parquet_basename in processed_parquet_basenames:
            downloaded_archives.add(parquet_basename)

    # Clean up parquet files
    for parquet_path in downloaded_parquets:
        try:
            if parquet_path and parquet_path.exists():
                parquet_path.unlink()
        except Exception:
            pass

    logger.info(
        f"Extracted {total_samples} samples from {len(downloaded_archive_files)} archives"
    )


def _extract_unique_archive_filenames(parquet_path: Path) -> set:
    """Extract unique archive filenames from gasstation parquet metadata file.

    Args:
        parquet_path: Path to the parquet metadata file

    Returns:
        Set of unique archive filenames referenced in the parquet
    """
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "archive_filename" not in df.columns:
            logger.warning(f"No archive_filename column in {parquet_path}")
            return set()

        archive_filenames = set(df["archive_filename"].dropna().unique())
        return archive_filenames
    except Exception as e:
        logger.warning(f"Failed to extract archive filenames from {parquet_path}: {e}")
        return set()


def _build_parquet_metadata_map(
    parquet_path: Path,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Build a mapping from (archive_filename, file_path_in_archive) to metadata.

    Args:
        parquet_path: Path to the parquet metadata file

    Returns:
        Dictionary mapping (archive_filename, file_path_in_archive) tuples to metadata dicts
    """
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if (
            "archive_filename" not in df.columns
            or "file_path_in_archive" not in df.columns
        ):
            logger.warning(f"Missing required columns in {parquet_path}")
            return {}

        metadata_map = {}
        for _, row in df.iterrows():
            archive_filename = row.get("archive_filename")
            file_path = row.get("file_path_in_archive")

            if pd.isna(archive_filename) or pd.isna(file_path):
                continue

            key = (str(archive_filename), str(file_path))
            metadata_map[key] = extract_row_metadata(row, "")

        return metadata_map
    except Exception as e:
        logger.warning(f"Failed to build metadata map from {parquet_path}: {e}")
        return {}


def _process_tar_with_metadata(
    archive_path: Path,
    dataset,
    metadata_map: Dict[Tuple[str, str], Dict[str, Any]],
    num_items: int,
    iso_week: Optional[str],
    seed: Optional[int],
) -> Generator[Dict[str, Any], None, None]:
    """Extract media from tar archive using parquet metadata mapping.

    Args:
        archive_path: Path to tar archive
        dataset: BenchmarkDatasetConfig
        metadata_map: Mapping from (archive_filename, file_path_in_archive) to metadata
        num_items: Number of items to extract (-1 for all)
        iso_week: ISO week string
        seed: Random seed

    Yields:
        Sample dictionaries with media and metadata
    """
    archive_basename = archive_path.name

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            valid_exts = (
                IMAGE_FILE_EXTENSIONS
                if dataset.modality == "image"
                else (AUDIO_FILE_EXTENSIONS if dataset.modality == "audio" else VIDEO_FILE_EXTENSIONS)
            )

            all_members = [m for m in archive.getmembers() if m.isreg()]
            candidates = [
                m
                for m in all_members
                if any(m.name.lower().endswith(ext) for ext in valid_exts)
                and "MACOSX" not in m.name
            ]

            if not candidates:
                logger.warning(f"No matching media files found in {archive_path}")
                return

            if num_items == -1:
                selected = candidates
            else:
                if seed is not None:
                    rng = random.Random(seed)
                    selected = rng.sample(candidates, min(num_items, len(candidates)))
                else:
                    selected = random.sample(
                        candidates, min(num_items, len(candidates))
                    )

            for member in selected:
                try:
                    src = archive.extractfile(member)
                    if src is None:
                        continue

                    with closing(src):
                        data_bytes = src.read()

                    if dataset.modality == "image":
                        try:
                            media_obj = Image.open(BytesIO(data_bytes))
                        except Exception as e:
                            logger.warning(f"Failed to open image {member.name}: {e}")
                            continue
                    else:
                        media_obj = data_bytes

                    sample = create_sample(dataset, media_obj, archive_path, iso_week)
                    sample["archive_filename"] = archive_basename
                    sample["member_path"] = member.name

                    # Look up metadata from parquet
                    metadata_key = (archive_basename, member.name)
                    if metadata_key in metadata_map:
                        meta = metadata_map[metadata_key]
                        for k, v in meta.items():
                            if k not in sample:
                                sample[k] = v

                    yield sample

                except Exception as e:
                    logger.warning(
                        f"Error extracting {member.name} from {archive_path}: {e}"
                    )
                    continue

    except Exception as e:
        logger.warning(f"Error opening tar archive {archive_path}: {e}")
        return


def _process_zip_or_tar(
    source_path: Path,
    dataset,
    num_items: int,
    iso_week: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Extract media from zip/tar archives (non-gasstation datasets)."""
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
                    selected = random.sample(
                        candidates, min(num_items, len(candidates))
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

                    sample = create_sample(dataset, media_obj, source_path, iso_week)
                    sample["archive_filename"] = source_path.name
                    sample["member_path"] = get_name(entry)

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

        yield create_sample(dataset, media_obj, source_path, iso_week)
    except Exception as e:
        logger.warning(f"Error reading direct file {source_path}: {e}")
        return


def _clean_to_json_serializable(value: Any) -> Any:
    """Convert arbitrary values to JSON-serializable equivalents."""
    try:
        if value is None or isinstance(value, (str, int, bool)):
            return value

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        if isinstance(value, (np.bool_,)):
            return bool(value)

        if isinstance(value, (bytes, bytearray)):
            try:
                return base64.b64encode(bytes(value)).decode("ascii")
            except Exception:
                return None

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, (np.ndarray, list, tuple, set)):
            return [
                _clean_to_json_serializable(v) for v in (value.tolist() if isinstance(value, np.ndarray) else list(value))
            ]

        if isinstance(value, dict):
            return {str(k): _clean_to_json_serializable(v) for k, v in value.items()}

        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)
    except Exception:
        return None


def _extract_parquet_row_metadata(row: Any, media_col: str) -> Dict[str, Any]:
    """Extract non-media columns from a pandas Series row and clean to JSON-serializable dict."""
    metadata: Dict[str, Any] = {}
    try:
        for col, val in row.items():
            if str(col) == str(media_col):
                continue
            cleaned = _clean_to_json_serializable(val)
            col_str = str(col)
            metadata[col_str] = cleaned
            
            col_lower = col_str.lower()
            if "hotkey" in col_lower and "generator_hotkey" not in metadata:
                metadata["generator_hotkey"] = cleaned
            if "uid" in col_lower and "generator_uid" not in metadata:
                metadata["generator_uid"] = cleaned
    except Exception:
        pass
    return metadata


def _process_parquet(
    source_path: Path, 
    dataset, 
    num_items: int, 
    iso_week: Optional[str] = None, 
    seed: Optional[int] = None
):
    """Process parquet files with embedded media bytes (e.g., PICA-100K).
    
    Supports datasets with single or multiple image columns.
    For datasets with multiple columns (e.g., src_img and tgt_img), yields one sample per column.
    """
    try:
        table = pq.read_table(source_path)
        df = table.to_pandas()
        
        if num_items == -1:
            sample_df = df
        else:
            sample_df = df.sample(n=min(num_items, len(df)), random_state=seed)

        data_columns = getattr(dataset, 'data_columns', None)
        if data_columns:
            media_cols = [c for c in data_columns if c in sample_df.columns]
            if not media_cols:
                logger.warning(
                    f"Specified data_columns {data_columns} not found in {source_path}"
                )
                return
        elif dataset.modality == "image":
            media_col = (
                next((c for c in sample_df.columns if c.lower() == "image"), None)
                or next((c for c in sample_df.columns if "image" in c.lower() and "_id" not in c.lower()), None)
                or next((c for c in sample_df.columns if "image" in c.lower()), None)
            )
            media_cols = [media_col] if media_col else []
        elif dataset.modality == "audio":
            candidates = ["audio", "bytes", "content", "data", "wav", "mp3"]
            exact = [c for c in sample_df.columns if c.lower() == "audio"]
            if exact:
                media_cols = exact
            else:
                media_cols = [
                    c
                    for c in sample_df.columns
                    if "_id" not in c.lower()
                    and any(k in c.lower() for k in candidates)
                ]
        else:
            candidates = ["video", "bytes", "content", "data"]
            media_col = (
                next((c for c in sample_df.columns if c.lower() in candidates), None)
                or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates) and "_id" not in c.lower()), None)
                or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates)), None)
            )
            media_cols = [media_col] if media_col else []

        if not media_cols:
            logger.warning(
                f"No media column found in {source_path} for modality {dataset.modality}"
            )
            return

        for _, row in sample_df.iterrows():
            for col in media_cols:
                try:
                    media_data = row[col]
                    audio_sampling_rate = None  # Track sampling rate for audio dicts
                    
                    if isinstance(media_data, dict):
                        # For audio, try to extract sampling_rate before getting the array
                        if dataset.modality == "audio":
                            audio_sampling_rate = media_data.get("sampling_rate") or media_data.get("sample_rate")
                        
                        key = next(
                            (
                                k
                                for k in media_data
                                if any(
                                    s in k.lower()
                                    for s in ["bytes", "image", "video", "audio", "array", "data", "content"]
                                )
                            ),
                            None,
                        )
                        if key:
                            media_data = media_data[key]
                        else:
                            logger.warning(f"No valid key found in dict media_data for {source_path}: {list(media_data.keys())}")
                            continue

                    if dataset.modality == "image":
                        if media_data is None or isinstance(media_data, (int, float)):
                            continue

                        try:
                            img = Image.open(BytesIO(media_data))
                        except Exception:
                            if isinstance(media_data, str):
                                media_data = base64.b64decode(media_data)
                            img = Image.open(BytesIO(media_data))
                        sample = create_sample(dataset, img, source_path, iso_week)
                    else:
                        if media_data is None or isinstance(media_data, (int, float)):
                            continue

                        # Handle audio array format (common in HF audio datasets)
                        if dataset.modality == "audio" and isinstance(media_data, (list, np.ndarray)):
                            # Audio data is a numpy array - convert to WAV bytes
                            try:
                                import io
                                import soundfile as sf
                                audio_array = np.array(media_data, dtype=np.float32)
                                if audio_array.ndim == 1:
                                    audio_array = audio_array.reshape(-1, 1)
                                # Use extracted sampling rate or default to 16000
                                sr = audio_sampling_rate if audio_sampling_rate else 16000
                                buffer = io.BytesIO()
                                sf.write(buffer, audio_array, sr, format='WAV')
                                media_data = buffer.getvalue()
                            except Exception as e:
                                logger.warning(f"Failed to convert audio array to bytes: {e}")
                                continue
                        elif not isinstance(media_data, (bytes, bytearray)):
                            if isinstance(media_data, str):
                                media_data = base64.b64decode(media_data)
                            else:
                                continue
                        sample = create_sample(dataset, bytes(media_data), source_path, iso_week)

                    row_metadata = _extract_parquet_row_metadata(row, col)
                    for k, v in row_metadata.items():
                        if k not in sample:
                            sample[k] = v
                    
                    if len(media_cols) > 1:
                        sample["source_column"] = col

                    yield sample
                except Exception as e:
                    logger.warning(f"Failed to extract row from {source_path} (column {col}): {e}")
                    continue
    except Exception as e:
        logger.warning(f"Error processing parquet file {source_path}: {e}")
        return


def _process_frame_directory(
    frame_dir: Path, dataset, iso_week: Optional[str] = None
) -> Generator[Dict[str, Any], None, None]:
    """Process a directory containing pre-extracted video frames.

    Args:
        frame_dir: Directory containing frame files (e.g., bm-videos/dfb/DFDC/test/frames/amwhgrjvkw/)
        dataset: BenchmarkDatasetConfig
        iso_week: Optional ISO week string

    Yields:
        Sample dict with 'video_frames' key containing list of frame paths
    """
    try:
        if not frame_dir.is_dir():
            logger.warning(f"{frame_dir} is not a directory")
            return

        # Get all image files in the directory
        frame_files = []
        for ext in IMAGE_FILE_EXTENSIONS:
            frame_files.extend(sorted(frame_dir.glob(f"*{ext}")))
            frame_files.extend(sorted(frame_dir.glob(f"*{ext.upper()}")))

        # Sort frames numerically if they have numeric names
        def extract_number(filepath):
            """Extract numeric part from filename for sorting (e.g., '000.png' -> 0)"""
            name = filepath.stem
            match = re.match(r"(\d+)", name)
            if match:
                return int(match.group(1))
            return name

        frame_files = sorted(set(frame_files), key=extract_number)

        if not frame_files:
            logger.warning(f"No frame files found in {frame_dir}")
            return

        logger.debug(f"Found {len(frame_files)} frames in {frame_dir.name}")

        # Create sample with frame paths
        sample = {
            "video_frames": frame_files,
            "media_type": dataset.media_type,
            "dataset_name": dataset.name,
            "dataset_path": dataset.path,
            "source_file": frame_dir.name,
        }

        if iso_week:
            sample["iso_week"] = iso_week

        yield sample

    except Exception as e:
        logger.warning(f"Error processing frame directory {frame_dir}: {e}")
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
        files = list(
            hf_hub.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
        )
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
        file_info = api.get_dataset_files(repo_id, revision="master", recursive=True)

        # Extract file paths
        if isinstance(file_info, list):
            files = [
                f["Path"] if isinstance(f, dict) and "Path" in f else str(f)
                for f in file_info
            ]
        elif isinstance(file_info, dict):
            files = [f for f in file_info.keys()] if file_info else []

        if extension and files:
            if isinstance(extension, (list, tuple, set)):
                exts = tuple(extension)
                files = [f for f in files if f.endswith(exts)]
            else:
                files = [f for f in files if f.endswith(extension)]

    except Exception as e:
        logger.error(
            f"Failed to list files of type {extension} in ModelScope repo {repo_id}: {e}"
        )
        logger.error(
            f"Make sure 'modelscope' package is installed: pip install modelscope"
        )

    return files


def _is_zip_file(filename_lower: str) -> bool:
    """Return True if filename looks like a zip archive."""
    return filename_lower.endswith(".zip")


def _is_tar_file(filename_lower: str) -> bool:
    """Return True if filename looks like a tar archive (.tar, .tar.gz, .tgz).
    
    Also handles hash-suffixed filenames like 'file.tar_abc123.gz' that result
    from download_single_file adding URL hashes to avoid collisions.
    """
    import re
    
    if filename_lower.endswith(".tar") or filename_lower.endswith(".tgz"):
        return True
    if filename_lower.endswith(".tar.gz"):
        return True
    # Handle hash-suffixed tar.gz files: file.tar_<hash>.gz
    if re.search(r"\.tar_[a-f0-9]{8}\.gz$", filename_lower):
        return True
    return False


def _is_parquet_file(filename_lower: str) -> bool:
    """Return True if filename looks like a parquet file."""
    return filename_lower.endswith(".parquet")


def download_files(
    urls: List[str],
    output_dir: Path,
    chunk_size: int = 8192,
    max_workers: int = 10,
    hf_token: Optional[str] = None,
) -> List[Path]:
    """Download multiple files in parallel.

    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        chunk_size: Size of chunks to download at a time
        max_workers: Maximum number of parallel downloads (default: 10)
        hf_token: Hugging Face API token for private datasets

    Returns:
        List of successfully downloaded file paths
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    def download_url(url):
        try:
            return download_single_file(url, output_dir, chunk_size, hf_token)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    max_workers = min(max_workers, len(urls)) if urls else 1

    if len(urls) > 1:
        logger.info(
            f"Downloading {len(urls)} files with {max_workers} parallel workers..."
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_url, url): url for url in urls}

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded_files.append(result)
            completed += 1
            if len(urls) > 10 and completed % 10 == 0:
                logger.info(f"  Progress: {completed}/{len(urls)} files downloaded")

    return downloaded_files


def download_single_file(
    url: str, output_dir: Path, chunk_size: int, hf_token: Optional[str] = None
) -> Optional[Path]:
    """Download a single file synchronously

    Args:
        url: URL to download (supports http/https and s3: schemes)
        output_dir: Directory to save the file
        chunk_size: Size of chunks to download at a time
        hf_token: Hugging Face API token for private datasets

    Returns:
        Path to the downloaded file, or None if failed
    """
    try:
        # Handle S3 URLs (format: s3:bucket-name/key/path or s3:bucket-name/frame/directory/)
        if url.startswith("s3:"):
            from .utils.s3_utils import download_s3_frame_directory

            s3_path = url[3:]  # Remove "s3:" prefix
            parts = s3_path.split("/", 1)
            if len(parts) != 2:
                logger.error(
                    f"Invalid S3 URL format: {url}. Expected: s3:bucket/key/path"
                )
                return None

            bucket = parts[0]
            key = parts[1]

            # Check if this is a frame directory (no file extension or ends with /)
            if key.endswith("/") or ("." not in os.path.basename(key)):
                # This is a frame directory - download all frames
                return download_s3_frame_directory(bucket, key, output_dir)
            else:
                # This is a single file - use unique filename to avoid collisions
                url_hash = hashlib.md5(key.encode()).hexdigest()[:8]
                base_filename = os.path.basename(key)
                name, ext = os.path.splitext(base_filename)
                filename = f"{name}_{url_hash}{ext}"
                filepath = output_dir / filename
                return download_s3_file(bucket, key, filepath)

        # Handle regular HTTP/HTTPS URLs
        # Use hash of full URL path to avoid filename collisions when multiple
        # files from different directories have the same basename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        base_filename = os.path.basename(url)
        name, ext = os.path.splitext(base_filename)
        filename = f"{name}_{url_hash}{ext}"
        filepath = output_dir / filename

        logger.info(f"Downloading {url}")
        headers = {}
        if hf_token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_token}"

        response = requests.get(url, stream=True, timeout=3600, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to download {url}: Status {response.status_code}")
            return None

        total_size = int(response.headers.get("content-length", 0))
        logger.info(f"Writing to {filepath} (size: {total_size/(1024*1024):.1f} MB)")

        downloaded = 0
        last_log_mb = 0
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=max(chunk_size, 1024 * 1024)
            ):  # Use at least 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    current_mb = downloaded / (1024 * 1024)
                    if current_mb - last_log_mb >= 100:
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            logger.info(
                                f"Progress: {current_mb:.0f}MB / {total_size/(1024*1024):.0f}MB ({pct:.1f}%)"
                            )
                        else:
                            logger.info(f"Downloaded: {current_mb:.0f}MB")
                        last_log_mb = current_mb

        # Verify download completeness
        if total_size > 0:
            actual_size = filepath.stat().st_size
            if actual_size != total_size:
                logger.error(
                    f"❌ Download incomplete: expected {total_size} bytes, got {actual_size} bytes"
                )
                return None
            logger.info(
                f"✅ Download verified: {filename} ({actual_size/(1024*1024):.1f} MB)"
            )
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

        except Exception as e:
            logger.warning(
                f"Failed to load cached sample {filename} from {dataset.name}: {e}"
            )
            continue
