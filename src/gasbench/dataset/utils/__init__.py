"""Dataset utilities for gasbench.

This package contains helper modules for dataset operations:
- s3_utils: S3/object storage integration
- gasstation_utils: Gasstation dataset management
- metadata_utils: Sample metadata extraction and cleaning
"""

from .s3_utils import (
    list_s3_files,
    download_s3_file,
    download_s3_frame_directory,
)

from .gasstation_utils import (
    calculate_target_weeks,
    get_week_directories,
    load_downloaded_archives,
    save_downloaded_archives,
    check_for_new_archives,
    is_week_cache_complete,
    get_week_sample_count,
    get_total_cached_samples,
    extract_iso_week_from_path,
    filter_files_by_current_week,
    filter_files_by_recent_weeks,
)

from .metadata_utils import (
    clean_to_json_serializable,
    extract_row_metadata,
    create_sample,
)

__all__ = [
    # S3
    "list_s3_files",
    "download_s3_file",
    "download_s3_frame_directory",
    # Gasstation
    "calculate_target_weeks",
    "get_week_directories",
    "load_downloaded_archives",
    "save_downloaded_archives",
    "check_for_new_archives",
    "is_week_cache_complete",
    "get_week_sample_count",
    "get_total_cached_samples",
    "extract_iso_week_from_path",
    "filter_files_by_current_week",
    "filter_files_by_recent_weeks",
    # Metadata
    "clean_to_json_serializable",
    "extract_row_metadata",
    "create_sample",
]

