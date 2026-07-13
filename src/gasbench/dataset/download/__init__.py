"""Dataset download and extraction.

Previously a single ~1900-line module; split into a package with the same
public API (all prior `from ...dataset.download import X` imports still work):

  constants  - shared constants, DatasetAccessError, archive-type predicates
  fetch      - low-level file downloading (HTTP/HF streaming)
  listing    - remote file enumeration and download-URL resolution
  extract    - decode media samples from parquet/tar/zip/raw sources
  cache_io   - load samples from the persistent cache volume
  core       - download_and_extract orchestration (incl. gasstation)
"""

from .constants import (
    DatasetAccessError,
    IMAGE_FILE_EXTENSIONS,
    VIDEO_FILE_EXTENSIONS,
    AUDIO_FILE_EXTENSIONS,
    MAX_FILES_DEFAULT,
)
from .fetch import download_files, download_single_file
from .listing import list_hf_files, list_modelscope_files
from .extract import yield_media_from_source, _process_parquet
from .core import download_and_extract

__all__ = [
    "download_and_extract",
    "DatasetAccessError",
    "IMAGE_FILE_EXTENSIONS",
    "VIDEO_FILE_EXTENSIONS",
    "AUDIO_FILE_EXTENSIONS",
    "MAX_FILES_DEFAULT",
    "download_files",
    "download_single_file",
    "list_hf_files",
    "list_modelscope_files",
    "yield_media_from_source",
    "_process_parquet",
]
