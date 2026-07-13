"""Low-level file downloading (HTTP/HF streaming) and output-name helpers."""

import hashlib
import os
import time
import traceback
from pathlib import Path
from typing import Generator, List, Optional

import requests

from ...logger import get_logger
from ..utils.s3_utils import download_s3_file

logger = get_logger(__name__)

def _stream_downloads(
    urls: List[str],
    output_dir: Path,
    chunk_size: int = 8192,
    max_workers: int = 10,
    hf_token: Optional[str] = None,
) -> Generator[Optional[Path], None, None]:
    """Like download_files but yields each path as soon as its download completes."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir.mkdir(parents=True, exist_ok=True)
    if not urls:
        return

    def download_url(url):
        try:
            return download_single_file(url, output_dir, chunk_size, hf_token)
        except (requests.RequestException, OSError) as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    workers = min(max_workers, len(urls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_url, url): url for url in urls}
        for future in as_completed(futures):
            yield future.result()



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
        except (requests.RequestException, OSError) as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    max_workers = min(max_workers, len(urls)) if urls else 1

    logger.debug(f"Downloading {len(urls)} files with {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_url, url): url for url in urls}

        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded_files.append(result)

    return downloaded_files



def download_single_file(
    url: str, output_dir: Path, chunk_size: int, hf_token: Optional[str] = None
) -> Optional[Path]:
    """Download a single file synchronously with resume and retry support

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
        partial_filepath = Path(str(filepath) + ".partial")

        logger.debug(f"Downloading {url}")

        max_retries = 5
        effective_chunk_size = max(chunk_size, 1024 * 1024)

        for attempt in range(max_retries):
            try:
                headers = {}
                if hf_token and "huggingface.co" in url:
                    headers["Authorization"] = f"Bearer {hf_token}"

                downloaded_size = 0
                if partial_filepath.exists():
                    downloaded_size = partial_filepath.stat().st_size
                    headers["Range"] = f"bytes={downloaded_size}-"
                    logger.info(f"Resuming download from {downloaded_size} bytes")

                response = requests.get(
                    url, stream=True, timeout=(30, 300), headers=headers
                )

                if response.status_code == 416:
                    logger.info("Server returned 416 (range not satisfiable), restarting download")
                    partial_filepath.unlink(missing_ok=True)
                    downloaded_size = 0
                    del headers["Range"]
                    response = requests.get(
                        url, stream=True, timeout=(30, 300), headers=headers
                    )

                if response.status_code not in (200, 206):
                    logger.error(f"Failed to download {url}: Status {response.status_code}")
                    return None

                if response.status_code == 200 and downloaded_size > 0:
                    logger.info("Server doesn't support resume, restarting download")
                    partial_filepath.unlink(missing_ok=True)
                    downloaded_size = 0

                total_size = int(response.headers.get("content-length", 0))
                if response.status_code == 200:
                    expected_total = total_size
                else:
                    expected_total = downloaded_size + total_size

                mode = "ab" if downloaded_size > 0 else "wb"
                with open(partial_filepath, mode) as f:
                    for chunk in response.iter_content(chunk_size=effective_chunk_size):
                        if chunk:
                            f.write(chunk)

                actual_size = partial_filepath.stat().st_size
                if expected_total > 0 and actual_size != expected_total:
                    raise IOError(
                        f"Download incomplete: expected {expected_total} bytes, got {actual_size} bytes"
                    )

                partial_filepath.rename(filepath)
                return filepath

            except (requests.exceptions.RequestException, IOError) as e:
                wait_time = min(30 * (2 ** attempt), 300)
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise

        return None

    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        logger.error(traceback.format_exc())
        return None



def _get_expected_download_filename(url: str) -> str:
    """Compute the filename that download_single_file() would produce for a URL.

    download_single_file() appends a URL hash to filenames to avoid collisions
    (e.g., "file.parquet" becomes "file_abc12345.parquet"). This function mirrors
    that naming convention to enable reverse-mapping from downloaded filenames
    back to original URLs.
    """
    if url.startswith("s3:"):
        s3_path = url[3:]
        parts = s3_path.split("/", 1)
        key = parts[1] if len(parts) == 2 else s3_path
        url_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        base_filename = os.path.basename(key)
    else:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        base_filename = os.path.basename(url)
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{url_hash}{ext}"



