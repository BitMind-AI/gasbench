"""S3-compatible object storage utilities for gasbench.

Supports any S3-compatible service including:
- AWS S3
- Wasabi
- MinIO
- DigitalOcean Spaces
- Backblaze B2
- Cloudflare R2
- etc.

Environment variables:
    S3_ACCESS_KEY: S3 access key ID (required)
    S3_SECRET_KEY: S3 secret access key (required)
    S3_ENDPOINT: S3 endpoint URL (optional, defaults to AWS)
    S3_REGION: S3 region (optional, defaults to us-east-1)
"""

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

from ...logger import get_logger

logger = get_logger(__name__)

# File extensions for frame detection
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# Global S3 client cache to reuse connection pool
_s3_client_cache = None


def _get_s3_client():
    """Initialize boto3 S3 client using environment credentials.
    
    Supports any S3-compatible service (AWS S3, MinIO, DigitalOcean Spaces, etc.)
    Caches the client to reuse connection pool across calls.
    
    Environment variables:
        S3_ACCESS_KEY: S3 access key ID
        S3_SECRET_KEY: S3 secret access key
        S3_ENDPOINT: S3 endpoint URL (optional, defaults to AWS)
        S3_REGION: S3 region (optional, defaults to us-east-1)
    
    Returns:
        boto3.client: Configured S3 client
        
    Raises:
        RuntimeError: If boto3 is not available or credentials are missing
    """
    global _s3_client_cache
    
    # Return cached client if available
    if _s3_client_cache is not None:
        return _s3_client_cache
    
    if not BOTO3_AVAILABLE:
        raise RuntimeError(
            "boto3 is not installed. Install it with: pip install boto3"
        )
    
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    endpoint_url = os.environ.get("S3_ENDPOINT")
    region = os.environ.get("S3_REGION", "us-east-1")
    
    if not access_key or not secret_key:
        raise RuntimeError(
            "S3 credentials not found. Set S3_ACCESS_KEY and S3_SECRET_KEY "
            "environment variables."
        )
    
    try:
        from botocore.config import Config
        # With 4 concurrent datasets and 10 workers each, we need ~40 connections
        config = Config(
            max_pool_connections=50,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        kwargs = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
            "config": config,
        }
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        
        _s3_client_cache = boto3.client("s3", **kwargs)
        return _s3_client_cache
    except Exception as e:
        raise RuntimeError(f"Failed to create S3 client: {e}")


def _parse_s3_path(path: str) -> Tuple[str, str]:
    """Parse S3 path into bucket and prefix.
    
    Args:
        path: Path in format 'bucket-name/prefix/path' or 'bucket-name'
        
    Returns:
        Tuple of (bucket, prefix). Prefix may be empty string.
        
    Examples:
        'my-bucket/datasets/images' -> ('my-bucket', 'datasets/images')
        'my-bucket' -> ('my-bucket', '')
    """
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def list_s3_files(path: str, extension=None) -> List[str]:
    """List files from an S3 bucket.
    
    Args:
        path: Path in format 'bucket-name/prefix/path'
        extension: Filter files by extension(s) (e.g., '.parquet' or ['.tar', '.tar.gz'])
                  Special value 'frames' to detect frame directories
        
    Returns:
        List of file keys (paths within the bucket) or directory paths if extension='frames'
    """
    try:
        bucket, prefix = _parse_s3_path(path)
        client = _get_s3_client()
        
        files = []
        paginator = client.get_paginator("list_objects_v2")
        
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
                
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                files.append(key)
        
        if extension == "frames":
            frame_dirs = set()
            for f in files:
                if any(f.lower().endswith(ext) for ext in IMAGE_FILE_EXTENSIONS):
                    parent = os.path.dirname(f)
                    if parent:
                        frame_dirs.add(parent)
            
            logger.info(f"Found {len(frame_dirs)} frame directories in S3 bucket {bucket}/{prefix}")
            return sorted(list(frame_dirs))
        
        if extension and files:
            if isinstance(extension, (list, tuple, set)):
                exts = tuple(extension)
                files = [f for f in files if f.endswith(exts)]
            else:
                files = [f for f in files if f.endswith(extension)]
        
        logger.info(f"Found {len(files)} files in S3 bucket {bucket}/{prefix}")
        return files
        
    except NoCredentialsError:
        logger.error("S3 credentials not found or invalid")
        return []
    except ClientError as e:
        logger.error(f"Failed to list S3 files from {path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error listing S3 files from {path}: {e}")
        return []


def _get_s3_urls(path: str, filenames: List[str]) -> List[str]:
    """Generate S3 URLs for files.
    
    Args:
        path: Path in format 'bucket-name/prefix'
        filenames: List of file keys to generate URLs for
        
    Returns:
        List of S3 URLs in format 's3:bucket-name/key/path'
    """
    bucket, prefix = _parse_s3_path(path)
    return [f"s3:{bucket}/{filename}" for filename in filenames]


def download_s3_file(
    bucket: str, key: str, output_path: Path
) -> Optional[Path]:
    """Download a single file from S3 bucket.
    
    Args:
        bucket: S3 bucket name
        key: File key (path) in the bucket
        output_path: Local path to save the file
        
    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        client = _get_s3_client()
        
        logger.info(f"Downloading s3://{bucket}/{key}")
        
        try:
            head = client.head_object(Bucket=bucket, Key=key)
            total_size = head.get("ContentLength", 0)
            logger.info(f"File size: {total_size/(1024*1024):.1f} MB")
        except Exception:
            total_size = 0
        
        client.download_file(bucket, key, str(output_path))
        
        if output_path.exists():
            actual_size = output_path.stat().st_size
            if total_size > 0 and actual_size != total_size:
                logger.error(
                    f"Download incomplete: expected {total_size} bytes, got {actual_size} bytes"
                )
                return None
            logger.info(f"✅ Downloaded: {output_path.name} ({actual_size/(1024*1024):.1f} MB)")
            return output_path
        else:
            logger.error(f"Download failed: file not found at {output_path}")
            return None
            
    except NoCredentialsError:
        logger.error("S3 credentials not found or invalid")
        return None
    except ClientError as e:
        logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error downloading s3://{bucket}/{key}: {e}")
        logger.error(traceback.format_exc())
        return None


def download_s3_frame_directory(
    bucket: str, frame_dir_key: str, output_dir: Path
) -> Optional[Path]:
    """Download all frames from an S3 frame directory.
    
    Args:
        bucket: S3 bucket name
        frame_dir_key: Directory path in bucket (e.g., 'dfb/DFDC/test/frames/amwhgrjvkw')
        output_dir: Local directory to save frames
        
    Returns:
        Path to the directory containing downloaded frames, or None if failed
    """
    try:
        client = _get_s3_client()
        
        frame_output_dir = output_dir / Path(frame_dir_key).name
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading frames from s3:{bucket}/{frame_dir_key}/")
        
        prefix = frame_dir_key if frame_dir_key.endswith("/") else frame_dir_key + "/"
        paginator = client.get_paginator("list_objects_v2")
        
        frame_files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if any(key.lower().endswith(ext) for ext in IMAGE_FILE_EXTENSIONS):
                    frame_files.append(key)
        
        if not frame_files:
            logger.warning(f"No frame files found in s3:{bucket}/{frame_dir_key}/")
            return None
        
        logger.info(f"Downloading {len(frame_files)} frames (using parallel downloads)...")
        
        def download_frame(frame_key):
            """Download a single frame."""
            frame_name = os.path.basename(frame_key)
            frame_path = frame_output_dir / frame_name
            try:
                client.download_file(bucket, frame_key, str(frame_path))
                return True
            except Exception as e:
                logger.warning(f"Failed to download frame {frame_key}: {e}")
                return False
        
        downloaded_count = 0
        # When multiple datasets download in parallel, too many workers causes connection pool exhaustion
        max_workers = min(10, len(frame_files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_frame, frame_key): frame_key for frame_key in frame_files}
            
            for future in as_completed(futures):
                if future.result():
                    downloaded_count += 1
                    if downloaded_count % 50 == 0:
                        logger.info(f"  Downloaded {downloaded_count}/{len(frame_files)} frames...")
        
        if downloaded_count == 0:
            logger.error(f"Failed to download any frames from {frame_dir_key}")
            return None
        
        logger.info(f"✅ Downloaded {downloaded_count}/{len(frame_files)} frames to {frame_output_dir}")
        return frame_output_dir
        
    except Exception as e:
        logger.error(f"Error downloading frame directory s3:{bucket}/{frame_dir_key}: {e}")
        logger.error(traceback.format_exc())
        return None

