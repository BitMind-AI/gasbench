"""Extract and decode media samples from parquet, tar, zip, and raw sources."""

import base64
import io
import json
import random
import re
import tarfile
from contextlib import closing
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple
from zipfile import ZipFile
from datetime import datetime

import soundfile as sf
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import numpy as np
from PIL import Image

from ...logger import get_logger
from ..utils.metadata_utils import create_sample, extract_row_metadata

from .constants import (
    IMAGE_FILE_EXTENSIONS,
    VIDEO_FILE_EXTENSIONS,
    AUDIO_FILE_EXTENSIONS,
    _is_parquet_file,
    _is_zip_file,
    _is_tar_file,
)

logger = get_logger(__name__)


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
    except (OSError, pyarrow.ArrowInvalid) as e:
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
    except (OSError, json.JSONDecodeError, FileNotFoundError) as e:
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

    archive_basename_normalized = re.sub(r"\.tar_[a-f0-9]{8}\.gz$", ".tar.gz", archive_basename)

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
                        except (OSError, FileNotFoundError) as e:
                            logger.warning(f"Failed to open image {member.name}: {e}")
                            continue
                    else:
                        media_obj = data_bytes

                    sample = create_sample(dataset, media_obj, archive_path, iso_week)
                    sample["archive_filename"] = archive_basename_normalized
                    sample["member_path"] = member.name

                    metadata_key = (archive_basename_normalized, member.name)
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

                def get_name(e):
                    return e

                def open_entry(e):
                    return archive.open(e)

            else:
                list_entries = [m for m in archive.getmembers() if m.isreg()]

                def get_name(m):
                    return m.name

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
                        except (OSError, FileNotFoundError):
                            logger.warning(
                                f"Failed to open image {get_name(entry)} from {source_path}"
                            )
                            continue
                    else:
                        media_obj = data_bytes

                    sample = create_sample(dataset, media_obj, source_path, iso_week)
                    # Normalize hash-appended filename back to original basename
                    # (download_single_file appends _<hash> before extension)
                    _normalized = re.sub(r"\.tar_[a-f0-9]{8}\.gz$", ".tar.gz", source_path.name)
                    if _normalized == source_path.name:
                        _normalized = re.sub(r"_[a-f0-9]{8}(\.\w+)$", r"\1", source_path.name)
                    sample["archive_filename"] = _normalized
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
        except (TypeError, ValueError):
            return str(value)
    except (TypeError, ValueError):
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
    except (OSError, json.JSONDecodeError, FileNotFoundError):
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

        filter_column = getattr(dataset, "filter_column", None)
        filter_value = getattr(dataset, "filter_value", None)
        if filter_column and filter_value:
            if filter_column not in df.columns:
                logger.warning(f"filter_column '{filter_column}' not in {source_path.name}; skipping")
                return
            df = df[df[filter_column] == filter_value]
            if df.empty:
                logger.info(f"No rows matching {filter_column}=={filter_value} in {source_path.name}")
                return
            logger.info(f"  {source_path.name}: {len(df)} rows after filter ({filter_column}=={filter_value})")

        # Frames-parquet: video dataset stored as one row per frame, grouped by video_id
        if dataset.modality == "video" and "video_id" in df.columns:
            yield from _process_frames_parquet(
                df, dataset, source_path, num_items, iso_week, seed
            )
            return

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
                        except (OSError, FileNotFoundError):
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
                                audio_array = np.array(media_data, dtype=np.float32)
                                if audio_array.ndim == 1:
                                    audio_array = audio_array.reshape(-1, 1)
                                sr = audio_sampling_rate if audio_sampling_rate else 16000
                                buffer = io.BytesIO()
                                sf.write(buffer, audio_array, sr, format='WAV')
                                media_data = buffer.getvalue()
                            except (ValueError, TypeError, OSError) as e:
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



def _process_frames_parquet(
    df, dataset, source_path: Path, num_items: int,
    iso_week: Optional[str] = None, seed: Optional[int] = None
):
    """Process a frames-parquet shard: one row per frame, one video per video_id group.

    Expected columns: video_id (str), image (frame bytes), optional frame_idx for
    ordering. Yields the same sample shape as _process_frame_directory, with
    'video_frames' holding a list of frame bytes (num_items counts videos).
    """
    media_col = (
        next((c for c in df.columns if c.lower() == "image"), None)
        or next(
            (c for c in df.columns
             if ("image" in c.lower() or "frame" in c.lower())
             and "_id" not in c.lower() and "idx" not in c.lower()),
            None,
        )
    )
    if not media_col:
        logger.warning(f"No frame media column found in {source_path}")
        return

    groups = list(df.groupby("video_id", sort=False))
    if num_items != -1 and len(groups) > num_items:
        rng = random.Random(seed)
        groups = rng.sample(groups, num_items)

    for video_id, group in groups:
        try:
            if "frame_idx" in group.columns:
                group = group.sort_values("frame_idx")
            frames = []
            for val in group[media_col]:
                if isinstance(val, dict):
                    val = next(
                        (v for v in val.values() if isinstance(v, (bytes, bytearray))),
                        None,
                    )
                if isinstance(val, (bytes, bytearray)):
                    frames.append(bytes(val))
            if not frames:
                logger.warning(f"No frame bytes for video {video_id} in {source_path.name}")
                continue

            sample = {
                "video_frames": frames,
                "media_type": dataset.media_type,
                "dataset_name": dataset.name,
                "dataset_path": dataset.path,
                "source_file": str(video_id),
            }
            if iso_week:
                sample["iso_week"] = iso_week
            yield sample
        except Exception as e:
            logger.warning(f"Failed to build video {video_id} from {source_path.name}: {e}")
            continue


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



