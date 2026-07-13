"""Shared constants, the dataset-access error, and archive-type predicates."""

import re

class DatasetAccessError(Exception):
    """Raised when a dataset cannot be accessed (gated, not found, or permission denied)."""

    pass


IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"}
AUDIO_FILE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

MAX_FILES_DEFAULT = 5000

def _is_zip_file(filename_lower: str) -> bool:
    """Return True if filename looks like a zip archive."""
    return filename_lower.endswith(".zip")


def _is_tar_file(filename_lower: str) -> bool:
    """Return True if filename looks like a tar archive (.tar, .tar.gz, .tgz).
    
    Also handles hash-suffixed filenames like 'file.tar_abc123.gz' that result
    from download_single_file adding URL hashes to avoid collisions.
    """
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



