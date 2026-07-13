"""Top-level download orchestration: download_and_extract, gasstation, filtered-sequential."""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pyarrow

from ...logger import get_logger
from ..utils.gasstation_utils import (
    extract_iso_week_from_path,
)

from .constants import (
    DatasetAccessError,
    IMAGE_FILE_EXTENSIONS,
    VIDEO_FILE_EXTENSIONS,
    AUDIO_FILE_EXTENSIONS,
)
from .cache_io import _is_dataset_cached, _load_dataset_from_cache
from .listing import (
    _list_remote_dataset_files,
    _get_download_urls,
    _select_files_to_download,
)
from .fetch import _stream_downloads, download_files, _get_expected_download_filename
from .extract import (
    yield_media_from_source,
    _extract_unique_archive_filenames,
    _build_parquet_metadata_map,
    _process_tar_with_metadata,
)

logger = get_logger(__name__)


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
        is_direct_media = src_fmt in {ext.lstrip(".") for ext in VIDEO_FILE_EXTENSIONS}

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
            except (OSError, json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Failed to load {dataset.name} from cache: {e}")
                logger.info(
                    f"Fallback to download: {dataset.name} will be downloaded fresh"
                )
                # Fall through to download logic

        # Create temporary directory for downloads
        if temp_dir is not None:
            try:
                Path(temp_dir).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        temp_dir_root = Path(tempfile.mkdtemp(dir=temp_dir))

        try:
            include_paths = getattr(dataset, "include_paths", None)
            exclude_paths = getattr(dataset, "exclude_paths", None)
            source = getattr(dataset, "source", "huggingface")

            n_files = _calculate_files_to_download(
                dataset,
                dataset.source_format,
                media_per_archive,
                archives_per_dataset,
            )
            max_files_to_list = n_files if n_files > 0 else None

            # ── Filtered sequential mode ─────────────────────────────────────
            # When filter_column/filter_value are set, download parquet shards
            # one at a time and stop as soon as the sample target is hit.
            filter_column = getattr(dataset, "filter_column", None)
            filter_value = getattr(dataset, "filter_value", None)
            if filter_column and filter_value:
                try:
                    all_filenames = _list_remote_dataset_files(
                        dataset.path,
                        dataset.source_format or "parquet",
                        False, None, None,
                        include_paths, exclude_paths,
                        source, hf_token,
                        max_files=None,  # need full list to shuffle and pick from
                    )
                except DatasetAccessError as e:
                    logger.warning(f"Skipping {dataset.name}: {e}")
                    return

                if not all_filenames:
                    logger.warning(f"No parquet files found for {dataset.path}")
                    return

                target_total = media_per_archive * archives_per_dataset
                logger.info(
                    f"Filtered mode: {dataset.name} — {filter_column}=={filter_value}, "
                    f"target {target_total} samples from {len(all_filenames)} shards"
                )
                yield from _download_filtered_sequential(
                    dataset=dataset,
                    filenames=all_filenames,
                    target_total=target_total,
                    temp_dir_root=temp_dir_root,
                    source=source,
                    hf_token=hf_token,
                    seed=seed,
                )
                return
            # ── Standard (non-filtered) download path ────────────────────────

            try:
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
                    max_files=max_files_to_list,
                )
            except DatasetAccessError as e:
                logger.warning(
                    f"Skipping dataset {dataset.name}: {e}"
                )
                return

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

                fallback_formats = [".parquet", ".zip", ".tar", ".tar.gz"]
                for fallback_format in fallback_formats:
                    if fallback_format != dataset.source_format:
                        logger.info(
                            f"Trying fallback format {fallback_format} for {dataset.path}"
                        )
                        try:
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
                                max_files=max_files_to_list,
                            )
                        except DatasetAccessError as e:
                            logger.warning(
                                f"Skipping dataset {dataset.name}: {e}"
                            )
                            return
                        if filenames:
                            logger.info(
                                f"Found {len(filenames)} files with format {fallback_format}"
                            )
                            # Recalculate n_files for the actual format found — the
                            # original n_files was computed for source_format (e.g. mp4)
                            # which treats each file as one media item, but an archive
                            # format (parquet/zip/tar) should only download
                            # archives_per_dataset files, not media_per_archive.
                            n_files = _calculate_files_to_download(
                                dataset, fallback_format, media_per_archive, archives_per_dataset
                            )
                            break

                if not filenames:
                    logger.warning(f"No files found for {dataset.path} with any format")
                    return

            remote_paths = _get_download_urls(dataset.path, filenames, source)

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

                successfully_processed = 0
                any_downloaded = False
                for downloaded_file in _stream_downloads(
                    to_download,
                    temp_dir_root,
                    chunk_size=8192,
                    max_workers=max_download_workers,
                    hf_token=hf_token,
                ):
                    if not downloaded_file or not downloaded_file.exists():
                        continue

                    any_downloaded = True
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
                        except OSError:
                            pass

                if not any_downloaded:
                    logger.warning(
                        f"No files successfully downloaded for {dataset.name}"
                    )
                    return

                logger.info(
                    f"Downloaded and processed {successfully_processed} files "
                    f"for {dataset.name}"
                )

        finally:
            if temp_dir_root.exists():
                shutil.rmtree(temp_dir_root)

    except Exception as e:
        logger.error(f"Error processing {dataset.path}: {e}")



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
        f"Processing {len(downloaded_parquets)} parquet files (per-parquet download & extract)"
    )

    # Build reverse mapping: downloaded filename -> original URL.
    # download_single_file() appends a URL hash to filenames (e.g., "file_abc12345.parquet"),
    # but downloaded_archives.json must store original basenames (e.g., "file.parquet")
    # to match HuggingFace file listings for cache completeness checks.
    _downloaded_name_to_url = {}
    for url in parquet_urls:
        expected_name = _get_expected_download_filename(url)
        _downloaded_name_to_url[expected_name] = url

    total_samples = 0
    total_archives = 0

    # Process each parquet independently: extract metadata → download its tars →
    # extract samples → mark parquet done. This ensures a parquet is only marked
    # as processed after ALL of its tar archives have been successfully downloaded
    # and extracted, and preserves progress incrementally across timeouts.
    for parquet_path in downloaded_parquets:
        if not parquet_path or not parquet_path.exists():
            continue

        try:
            # Resolve original URL basename (not hash-appended) for archive tracking
            original_url = _downloaded_name_to_url.get(parquet_path.name)
            original_basename = os.path.basename(original_url) if original_url else parquet_path.name

            iso_week = extract_iso_week_from_path(original_url or str(parquet_path))

            # Step 1: Extract archive filenames referenced by this parquet
            archive_filenames = _extract_unique_archive_filenames(parquet_path)
            if not archive_filenames:
                logger.warning(f"No archive filenames found in {parquet_path}")
                # Still mark as done - the parquet was processed, it just had no archives
                if downloaded_archives is not None:
                    downloaded_archives.add(original_basename)
                continue

            archive_to_week = {}
            for archive_filename in archive_filenames:
                if iso_week:
                    archive_to_week[archive_filename] = iso_week

            # Step 2: Build metadata map from this parquet
            metadata_map = _build_parquet_metadata_map(parquet_path)
            if iso_week:
                for key in metadata_map.keys():
                    if key in metadata_map:
                        metadata_map[key]["iso_week"] = iso_week

            # Step 3: Download this parquet's tar archives
            archive_paths = [
                f"archives/{week}/{basename}"
                for basename, week in archive_to_week.items()
            ]
            archive_urls = _get_download_urls(dataset.path, archive_paths, source)

            if not archive_urls:
                logger.warning(f"No archive URLs resolved for {original_basename}")
                continue

            max_workers = min(10, len(archive_urls))
            downloaded_archive_files = download_files(
                archive_urls,
                temp_dir_root,
                chunk_size=8192,
                max_workers=max_workers,
                hf_token=hf_token,
            )

            if not downloaded_archive_files:
                logger.warning(
                    f"Failed to download tar archives for parquet {original_basename}"
                )
                continue

            # Step 4: Extract samples from this parquet's tar archives
            parquet_samples = 0
            for archive_path in downloaded_archive_files:
                if not archive_path or not archive_path.exists():
                    continue

                try:
                    tar_iso_week = extract_iso_week_from_path(str(archive_path))

                    for sample in _process_tar_with_metadata(
                        archive_path,
                        dataset,
                        metadata_map,
                        media_per_archive,
                        tar_iso_week,
                        seed,
                    ):
                        parquet_samples += 1
                        total_samples += 1
                        yield sample
                except (OSError, pyarrow.ArrowInvalid) as e:
                    logger.warning(f"Failed to process archive {archive_path}: {e}")
                    continue
                finally:
                    try:
                        if archive_path.exists():
                            archive_path.unlink()
                    except OSError:
                        pass

            total_archives += len(downloaded_archive_files)

            # Step 5: Mark parquet as done ONLY after all its tars are processed.
            # This is the last step so if anything above fails or times out,
            # the parquet will be retried on the next run.
            if downloaded_archives is not None:
                downloaded_archives.add(original_basename)
            logger.info(
                f"✅ Parquet {original_basename}: {parquet_samples} samples "
                f"from {len(downloaded_archive_files)} archives"
            )
        except (OSError, pyarrow.ArrowInvalid) as e:
            logger.warning(f"Failed to process parquet {parquet_path}: {e}")
            continue
        finally:
            # Clean up this parquet file immediately
            try:
                if parquet_path and parquet_path.exists():
                    parquet_path.unlink()
            except (OSError, pyarrow.ArrowInvalid):
                pass

    logger.info(
        f"Extracted {total_samples} samples from {total_archives} archives"
    )



def _download_filtered_sequential(
    dataset,
    filenames: List[str],
    target_total: int,
    temp_dir_root: Path,
    source: str,
    hf_token: Optional[str],
    seed: Optional[int],
) -> Generator[Dict[str, Any], None, None]:
    """Download parquet shards one at a time, stopping once target_total samples are collected.

    Filtering is handled inside _process_parquet via dataset.filter_column/filter_value,
    so all media decoding follows the exact same path as regular parquet downloads.
    """
    remote_paths = _get_download_urls(dataset.path, filenames, source)
    if seed is not None:
        random.Random(seed).shuffle(remote_paths)
    else:
        random.shuffle(remote_paths)

    collected = 0
    for url in remote_paths:
        if collected >= target_total:
            break

        logger.info(
            f"Filtered download [{dataset.name}]: {collected}/{target_total} collected, "
            f"fetching next shard"
        )
        downloaded = download_files([url], temp_dir_root, hf_token=hf_token)
        if not downloaded:
            continue

        parquet_path = downloaded[0]
        try:
            remaining = target_total - collected
            for sample in yield_media_from_source(
                parquet_path, dataset, remaining, iso_week=None,
                hf_token=hf_token, seed=seed,
            ):
                yield sample
                collected += 1
                if collected >= target_total:
                    break
        finally:
            try:
                parquet_path.unlink()
            except (OSError, pyarrow.ArrowInvalid):
                pass

    logger.info(
        f"Filtered download complete: {collected}/{target_total} samples for {dataset.name}"
    )



