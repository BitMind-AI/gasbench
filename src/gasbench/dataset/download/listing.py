"""Remote dataset file listing and download-URL resolution (HF/ModelScope/S3)."""

import random
from urllib.parse import quote
from typing import List, Optional, Sequence, Union

import huggingface_hub as hf_hub
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from modelscope.hub.api import HubApi as MSHubApi

from ...logger import get_logger
from ..utils.s3_utils import list_s3_files, _get_s3_urls
from ..utils.gasstation_utils import (
    filter_files_by_current_week,
    filter_files_by_recent_weeks,
)

from .constants import DatasetAccessError, MAX_FILES_DEFAULT

logger = get_logger(__name__)


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
    source_format: Union[str, Sequence[str]] = ".parquet",
    current_week_only: bool = False,
    num_weeks: int = None,
    target_week: str = None,
    include_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    source: str = "huggingface",
    hf_token: Optional[str] = None,
    max_files: Optional[int] = None,
    hf_revision: Optional[str] = None,
    hf_subfolders: Optional[List[str]] = None,
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
        max_files: Stop after collecting this many files (None uses MAX_FILES_DEFAULT)
    """
    if max_files is None:
        max_files = MAX_FILES_DEFAULT
    formats = [source_format] if isinstance(source_format, str) else list(source_format)
    formats = [
        fmt if fmt == "frames" or fmt.startswith(".") else f".{fmt}" for fmt in formats
    ]
    if any(fmt in {".tar", ".tar.gz", ".tgz"} for fmt in formats):
        formats = list(dict.fromkeys([*formats, ".tar", ".tar.gz", ".tgz"]))
    preferred_format = formats[0]
    preferred_extensions = (
        (".tar", ".tar.gz", ".tgz")
        if preferred_format in {".tar", ".tar.gz", ".tgz"}
        else preferred_format
    )
    source_format = preferred_format if len(formats) == 1 else formats

    # For gasstation datasets with week filtering, we must fetch ALL files before
    # applying max_files. Otherwise early termination picks up files from older
    # weeks (alphabetically first) and the subsequent week filter discards them all,
    # resulting in zero files.
    is_gasstation = "gasstation" in dataset_path.lower()
    needs_week_filter = is_gasstation and (
        target_week or num_weeks or current_week_only
    )
    listing_max = None if needs_week_filter else max_files

    if source == "modelscope":
        files = list_modelscope_files(repo_id=dataset_path, extension=source_format)
        if include_paths:
            files = [
                f for f in files if any(path_seg in f for path_seg in include_paths)
            ]
        if exclude_paths:
            files = [
                f for f in files if not any(path_seg in f for path_seg in exclude_paths)
            ]
        if not needs_week_filter:
            files = files[:max_files]
    elif source == "s3":
        files = list_s3_files(path=dataset_path, extension=source_format)
        if include_paths:
            files = [
                f for f in files if any(path_seg in f for path_seg in include_paths)
            ]
        if exclude_paths:
            files = [
                f for f in files if not any(path_seg in f for path_seg in exclude_paths)
            ]
        if not needs_week_filter:
            files = files[:max_files]
    else:  # hf - supports early termination natively
        files = list_hf_files(
            repo_id=dataset_path,
            extension=source_format,
            preferred_extension=preferred_extensions if len(formats) > 1 else None,
            token=hf_token,
            max_files=listing_max,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            revision=hf_revision,
            subfolders=hf_subfolders,
        )

    if is_gasstation:
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

        # Apply max_files limit AFTER week filtering
        if needs_week_filter and max_files:
            files = files[:max_files]

    return files


def _get_download_urls(
    dataset_path: str,
    filenames: List[str],
    source: str = "huggingface",
    hf_revision: Optional[str] = None,
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
        return _get_huggingface_urls(dataset_path, filenames, hf_revision)


def _get_huggingface_urls(
    dataset_path: str, filenames: List[str], revision: Optional[str] = None
) -> List[str]:
    revision = quote(revision or "main", safe="")
    return [
        f"https://huggingface.co/datasets/{dataset_path}/resolve/{revision}/{quote(f, safe='/')}"
        for f in filenames
    ]


def _get_modelscope_urls(dataset_path: str, filenames: List[str]) -> List[str]:
    return [
        f"https://www.modelscope.cn/api/v1/datasets/{dataset_path}/repo?Revision=master&FilePath={f}"
        for f in filenames
    ]


def list_hf_files(
    repo_id,
    repo_type="dataset",
    extension=None,
    token=None,
    max_files=None,
    include_paths=None,
    exclude_paths=None,
    preferred_extension=None,
    revision=None,
    subfolders=None,
):
    """List files from a Hugging Face repository with early termination support.

    Args:
        repo_id: Repository ID
        repo_type: Type of repository ('dataset', 'model', etc.)
        extension: Filter files by extension
        token: Hugging Face API token for private datasets
        max_files: Stop after collecting this many matching files (None = no limit)
        include_paths: Only include files containing one of these path segments
        exclude_paths: Exclude files containing any of these path segments

    Returns:
        List of files in the repository

    Raises:
        DatasetAccessError: If the repository is gated/private without access, or not found
    """
    files = []
    preferred_matches = 0
    extension_counts = {}
    if extension:
        if isinstance(extension, (list, tuple, set)):
            exts = tuple(extension)
        else:
            exts = (extension,)
    else:
        exts = None

    try:
        api = hf_hub.HfApi()
        roots = subfolders or [None]
        entries = (
            entry
            for root in roots
            for entry in api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=root,
                recursive=True,
                expand=False,
                revision=revision,
                repo_type=repo_type,
                token=token,
            )
        )
        for entry in entries:
            f = getattr(entry, "path", None)
            if not f:
                continue
            if exts and not f.endswith(exts):
                continue
            if include_paths and not any(path_seg in f for path_seg in include_paths):
                continue
            if exclude_paths and any(path_seg in f for path_seg in exclude_paths):
                continue
            is_preferred = preferred_extension is None or f.endswith(preferred_extension)
            if is_preferred:
                preferred_matches += 1
            # Retain enough candidates for each format without allowing a
            # fallback extension to crowd the preferred extension out.
            matched_extension = next((ext for ext in exts or () if f.endswith(ext)), None)
            retained_for_extension = extension_counts.get(matched_extension, 0)
            if not max_files or retained_for_extension < max_files:
                files.append(f)
                extension_counts[matched_extension] = retained_for_extension + 1
            if max_files and preferred_matches >= max_files:
                logger.info(
                    f"Early termination: collected {max_files} files from {repo_id}"
                )
                break
    except GatedRepoError:
        raise DatasetAccessError(
            f"Dataset {repo_id} is gated and requires access approval. "
            "Visit the dataset page on HuggingFace to request access."
        )
    except RepositoryNotFoundError:
        raise DatasetAccessError(
            f"Dataset {repo_id} not found. It may not exist or may be private."
        )
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
            "Make sure 'modelscope' package is installed: pip install modelscope"
        )

    return files
