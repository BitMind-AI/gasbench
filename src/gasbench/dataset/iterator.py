"""Dataset iterator for unified access to benchmark datasets."""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict

from PIL import Image

from ..logger import get_logger
from .config import BenchmarkDatasetConfig
from .download import download_and_extract
from .cache import save_sample_to_cache, save_dataset_cache_files
from . import gasstation_utils

logger = get_logger(__name__)

DEFAULT_MAX_SAMPLES = 10000


class DatasetIterator:
    """Unified iterator for benchmark datasets with caching support."""

    def __init__(
        self,
        dataset_config: BenchmarkDatasetConfig,
        max_samples: int = None,
        cache_dir: str = "/.cache/gasbench",
        download: bool = True,
        num_weeks: int = None,
    ):
        self.config = dataset_config
        self.max_samples = max_samples or DEFAULT_MAX_SAMPLES
        self.samples_yielded = 0
        self.cache_dir = cache_dir
        self.num_weeks = num_weeks
        
        self.is_gasstation = "gasstation" in dataset_config.name.lower()
        if self.is_gasstation:
            # Gasstation datasets use week-based subdirectories
            self.target_weeks = gasstation_utils.calculate_target_weeks(num_weeks)
            self.week_dirs = gasstation_utils.get_week_directories(
                dataset_config.name, cache_dir, self.target_weeks
            )
            self.dataset_base_dir = f"{cache_dir}/datasets/{dataset_config.name}"
            self.dataset_dir = None
        else:
            # Non-gasstation datasets use simple directory
            self.dataset_dir = f"{cache_dir}/datasets/{dataset_config.name}"
            self.target_weeks = None
            self.week_dirs = None
            self.dataset_base_dir = self.dataset_dir

        if download:
            self.ensure_cached()

    def __iter__(self):
        return self

    def __next__(self):
        if self.samples_yielded >= self.max_samples:
            raise StopIteration

        if not hasattr(self, "_generator"):
            self._generator = self.get_samples()

        try:
            sample = next(self._generator)
            self.samples_yielded += 1
            return sample
        except StopIteration:
            raise StopIteration

    def ensure_cached(self):
        """Ensure dataset is fully cached (download if needed or incomplete).
        
        For gasstation datasets, ensures each target week is cached separately.
        """
        if self.is_gasstation:
            # Process each week separately
            for week_str, week_dir in zip(self.target_weeks, self.week_dirs):
                self._ensure_week_cached(week_str, week_dir)
        else:
            # Standard caching for non-gasstation datasets
            if self._is_cache_complete():
                logger.info(f"Cache complete for {self.config.name} ({self._get_cached_count()} samples)")
                return
            self._download_and_cache()

    def get_samples(self):
        """Generator that yields samples from cached data.
        
        For gasstation datasets, loads from all target week directories.
        """
        try:
            if self.is_gasstation:
                # Load from all week directories
                total_samples = 0
                for week_str, week_dir in zip(self.target_weeks, self.week_dirs):
                    if not os.path.exists(week_dir):
                        continue

                    logger.info(f"Loading cached data for {self.config.name} week {week_str}")
                    for sample in self._load_from_cache_dir(week_dir):
                        if total_samples >= self.max_samples:
                            return
                        yield sample
                        total_samples += 1
            else:
                # Non-gasstation datasets: standard loading
                if not self._has_cached_dataset():
                    logger.warning(f"No cached data found for {self.config.name}. Call ensure_cached() first.")
                    return

                cached_count = self._get_cached_count()
                samples_to_load = min(cached_count, self.max_samples)
                logger.info(f"Loading {samples_to_load} cached samples for {self.config.name}")
                
                total_samples = 0
                for sample in self._load_from_cache_dir(self.dataset_dir):
                    if total_samples >= self.max_samples:
                        return
                    yield sample
                    total_samples += 1

        except Exception as e:
            logger.error(f"Failed to get samples from {self.config.name}: {e}")
            return

    def _ensure_week_cached(self, week_str: str, week_dir: str):
        """Ensure a specific week is cached for gasstation datasets.
        
        Args:
            week_str: ISO week string (e.g., "2025W40")
            week_dir: Directory path for this week's cache
        """
        if gasstation_utils.is_week_cache_complete(
            week_dir,
            self.config.path,
            week_str,
            getattr(self.config, "source_format", ""),
            self.config.modality
        ):
            sample_count = gasstation_utils.get_week_sample_count(week_dir)
            logger.info(f"Cache complete for {self.config.name} week {week_str} ({sample_count} samples)")
            return
        
        downloaded_archives = gasstation_utils.load_downloaded_archives(week_dir)
        self._download_and_cache_week(week_str, week_dir, downloaded_archives)
    
    def _extract_sample_metadata(self, sample: Dict) -> Dict:
        """Extract metadata from sample for caching.
        
        Args:
            sample: Sample dictionary with media data and metadata
            
        Returns:
            Dictionary with extracted metadata fields
        """
        metadata = {
            "source_file": sample.get("source_file", ""),
            "model_name": sample.get("model_name", ""),
            "media_type": sample.get("media_type", ""),
        }
        
        # gasstation-specific fields
        for field in ["iso_week", "generator_hotkey", "generator_uid"]:
            if field in sample:
                metadata[field] = sample.get(field)
        
        # Debug: Log if we're missing generator info for gasstation datasets
        if "gasstation" in self.config.name.lower():
            has_generator = "generator_hotkey" in metadata or "generator_hotkey" in sample
            if not has_generator:
                logger.debug(f"⚠️  No generator_hotkey found in sample. Available keys: {list(sample.keys())[:10]}")
        
        return metadata
    
    def _find_oldest_sample(self, sample_metadata: Dict, samples_dir: str) -> Optional[str]:
        """Find the oldest sample file to evict (based on filename index).
        
        Args:
            sample_metadata: Dictionary mapping filenames to metadata
            samples_dir: Directory containing sample files
            
        Returns:
            Filename of oldest sample, or None if none found
        """
        if not sample_metadata:
            return None
        
        # Samples are named like img_000000.jpg, img_000001.jpg, vid_000000.mp4, etc.
        # Find the one with the lowest index number
        oldest_file = None
        oldest_index = float('inf')
        
        for filename in sample_metadata.keys():
            # Extract numeric index from filename (e.g., "000123" from "img_000123.jpg")
            match = re.search(r'_(\d+)', filename)
            if match:
                index = int(match.group(1))
                if index < oldest_index:
                    oldest_index = index
                    oldest_file = filename
        
        return oldest_file
    
    def _download_and_cache_week(self, week_str: str, week_dir: str, downloaded_archives: set):
        """Download and cache data for a specific ISO week (gasstation datasets only).
        
        Args:
            week_str: ISO week string (e.g., "2025W40")
            week_dir: Directory path for this week's cache
            downloaded_archives: Set of already-downloaded archive basenames
        """
        samples_dir = os.path.join(week_dir, "samples")
        metadata_file = os.path.join(week_dir, "sample_metadata.json")
        archive_tracker_file = os.path.join(week_dir, "downloaded_archives.json")
        
        sample_metadata = {}
        sample_count = 0
        next_index = 0  # Track the next index to use for new files
        
        # Load existing cache
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    sample_metadata = json.load(f)
                sample_count = len(sample_metadata)
                
                # Find the highest index used so far to continue from there
                max_index = -1
                for filename in sample_metadata.keys():
                    match = re.search(r'_(\d+)', filename)
                    if match:
                        max_index = max(max_index, int(match.group(1)))
                next_index = max_index + 1
                
                logger.info(f"Found {sample_count} existing samples for week {week_str}, next index: {next_index}")
            except Exception as e:
                logger.warning(f"Failed to load partial metadata for week {week_str}: {e}")
                sample_metadata = {}
                sample_count = 0
                next_index = 0
        
        # If we're at max_samples and there are new archives, we'll do sample replacement
        # Keep the most recent samples by evicting oldest ones
        replacing_samples = sample_count >= self.max_samples
        if replacing_samples:
            logger.info(
                f"Week {week_str} has {sample_count} samples (at max_samples). "
                f"Will replace oldest samples with fresh data from new archives"
            )
        
        Path(samples_dir).mkdir(parents=True, exist_ok=True)
        
        initial_sample_count = sample_count

        # Download with week filter (num_weeks=None to download just this week via week filtering in download_and_extract)
        for sample in download_and_extract(
            self.config,
            images_per_parquet=self.config.images_per_parquet,
            videos_per_zip=self.config.videos_per_zip,
            parquet_per_dataset=self.config.parquet_per_dataset,
            zips_per_dataset=self.config.zips_per_dataset,
            temp_dir=f"{self.cache_dir}/temp_downloads",
            force_download=False,
            cache_dir=self.cache_dir,
            num_weeks=None,
            downloaded_archives=downloaded_archives,
            target_week=week_str
        ):
            archive_name = sample.get("archive_filename") or sample.get("source_file", "")
            if archive_name:
                downloaded_archives.add(archive_name)
            
            # If we're replacing samples, find and remove the oldest sample
            if replacing_samples and sample_count >= self.max_samples:
                oldest_sample = self._find_oldest_sample(sample_metadata, samples_dir)
                if oldest_sample:
                    # Remove the oldest sample
                    old_file = os.path.join(samples_dir, oldest_sample)
                    try:
                        if os.path.exists(old_file):
                            os.remove(old_file)
                        del sample_metadata[oldest_sample]
                        sample_count -= 1
                        logger.debug(f"Evicted oldest sample: {oldest_sample}")
                    except Exception as e:
                        logger.warning(f"Failed to evict old sample {oldest_sample}: {e}")
            
            # Use next_index for the filename to avoid overwriting existing files
            filename = save_sample_to_cache(
                sample, self.config, samples_dir, next_index
            )
            if filename:
                sample_metadata[filename] = self._extract_sample_metadata(sample)
                sample_count += 1
                next_index += 1

                if sample_count % 50 == 0:
                    save_dataset_cache_files(
                        self.config, week_dir, sample_metadata, sample_count
                    )
                    gasstation_utils.save_downloaded_archives(week_dir, downloaded_archives)
                    logger.info(f"Downloaded {sample_count} samples for week {week_str}")

        new_samples = sample_count - initial_sample_count

        if new_samples > 0:
            save_dataset_cache_files(
                self.config, week_dir, sample_metadata, sample_count
            )
            gasstation_utils.save_downloaded_archives(week_dir, downloaded_archives)
            
            if replacing_samples:
                logger.info(
                    f"Week {week_str}: Downloaded {new_samples} new samples (replaced old samples). "
                    f"Total: {sample_count} samples, {len(downloaded_archives)} archives"
                )
            else:
                logger.info(
                    f"Week {week_str}: Downloaded {new_samples} new samples. "
                    f"Total: {sample_count} samples, {len(downloaded_archives)} archives"
                )
        elif sample_count < 0:
            logger.warning(f"No samples cached for week {week_str}")
    
    def _download_and_cache(self):
        """Download samples and save them to cache with incremental checkpointing.
        
        Used for non-gasstation datasets.
        """
        samples_dir = os.path.join(self.dataset_dir, "samples")
        metadata_file = os.path.join(self.dataset_dir, "sample_metadata.json")

        sample_metadata = {}
        sample_count = 0
        next_index = 0  # Track the next index to use for new files

        # Load existing cache
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    sample_metadata = json.load(f)
                sample_count = len(sample_metadata)
                
                # Find the highest index used so far to continue from there
                max_index = -1
                for filename in sample_metadata.keys():
                    match = re.search(r'_(\d+)', filename)
                    if match:
                        max_index = max(max_index, int(match.group(1)))
                next_index = max_index + 1
                
                logger.info(
                    f"Found partial cache with {sample_count} samples, next index: {next_index}, resuming download"
                )
            except Exception as e:
                logger.warning(f"Failed to load partial metadata, starting fresh: {e}")
                sample_metadata = {}
                sample_count = 0
                next_index = 0
        else:
            logger.info(
                f"No cached datasets found.\nDownloading {self.config.name} (max {self.max_samples} samples)"
            )

        if sample_count >= self.max_samples:
            logger.info(f"Cache complete with {sample_count} samples")
            return

        Path(samples_dir).mkdir(parents=True, exist_ok=True)

        for sample in download_and_extract(
            self.config,
            images_per_parquet=self.config.images_per_parquet,
            videos_per_zip=self.config.videos_per_zip,
            parquet_per_dataset=self.config.parquet_per_dataset,
            zips_per_dataset=self.config.zips_per_dataset,
            temp_dir=f"{self.cache_dir}/temp_downloads",
            force_download=False,
            cache_dir=self.cache_dir,
            num_weeks=self.num_weeks,
            downloaded_archives=None,  # Not used for non-gasstation datasets
        ):
            if sample_count >= self.max_samples:
                break

            filename = save_sample_to_cache(
                sample, self.config, samples_dir, next_index
            )
            if filename:
                sample_metadata[filename] = self._extract_sample_metadata(sample)
                sample_count += 1
                next_index += 1  # Always increment to use new indices

                # Save metadata incrementally every 50 samples
                if sample_count % 50 == 0:
                    save_dataset_cache_files(
                        self.config, self.dataset_dir, sample_metadata, sample_count
                    )
                    logger.info(
                        f"Downloaded {sample_count}/{self.max_samples} samples (checkpoint saved)"
                    )

        # Save final metadata
        if sample_count > 0:
            save_dataset_cache_files(
                self.config, self.dataset_dir, sample_metadata, sample_count
            )
            logger.info(
                f"Download complete: Saved {sample_count} samples to cache for {self.config.name}"
            )
        else:
            logger.warning(f"No samples were downloaded for {self.config.name}")

    def _has_cached_dataset(self) -> bool:
        """Check if dataset has any cached data (even if incomplete)."""
        try:
            dataset_info_file = os.path.join(self.dataset_dir, "dataset_info.json")
            samples_dir = os.path.join(self.dataset_dir, "samples")
            metadata_file = os.path.join(self.dataset_dir, "sample_metadata.json")

            return (
                os.path.exists(dataset_info_file)
                and os.path.exists(samples_dir)
                and os.path.exists(metadata_file)
                and len(os.listdir(samples_dir)) > 0
            )
        except Exception:
            return False

    def _is_cache_complete(self) -> bool:
        """Check if cached dataset has enough samples for max_samples.
        
        For gasstation datasets, checks all week directories.
        For non-gasstation datasets, checks the single dataset directory.
        """
        try:
            if self.is_gasstation:
                # For gasstation, check all week directories
                # (Individual weeks are checked in _ensure_week_cached)
                total_cached = gasstation_utils.get_total_cached_samples(self.week_dirs)
                return total_cached >= self.max_samples
            else:
                if not self._has_cached_dataset():
                    return False

                metadata_file = os.path.join(self.dataset_dir, "sample_metadata.json")
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                cached_count = len(metadata)
                return cached_count >= self.max_samples

        except Exception:
            return False

    def _get_cached_count(self) -> int:
        """Get the number of samples currently cached."""
        try:
            metadata_file = os.path.join(self.dataset_dir, "sample_metadata.json")
            if not os.path.exists(metadata_file):
                return 0

            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return len(metadata)
        except Exception:
            return 0
    
    def get_total_cached_count(self) -> int:
        """Get total number of samples cached across all directories.
        
        For gasstation datasets, counts across all week directories.
        For non-gasstation datasets, counts from the single dataset directory.
        
        This reads metadata files directly without loading actual samples.
        """
        try:
            if self.is_gasstation:
                return gasstation_utils.get_total_cached_samples(self.week_dirs)
            else:
                return self._get_cached_count()
        except Exception as e:
            logger.warning(f"Failed to get cached count for {self.config.name}: {e}")
            return 0

    def _load_from_cache_dir(self, cache_dir: str):
        """Load samples from a specific cache directory.
        
        Args:
            cache_dir: Path to the cache directory to load from
        """
        try:
            samples_dir = os.path.join(cache_dir, "samples")
            metadata_file = os.path.join(cache_dir, "sample_metadata.json")
            
            if not os.path.exists(metadata_file) or not os.path.exists(samples_dir):
                return
            
            with open(metadata_file, "r") as f:
                metadata_map = json.load(f)

            sample_files = [
                f
                for f in os.listdir(samples_dir)
                if os.path.isfile(os.path.join(samples_dir, f))
            ]
            
            # Sort by numeric index in reverse order (newest first)
            # Files are named like img_000000.jpg, img_000001.jpg, etc.
            # Higher indices = newer files
            def extract_index(filename):
                match = re.search(r'_(\d+)', filename)
                return int(match.group(1)) if match else -1
            
            sample_files.sort(key=extract_index, reverse=True)

            for filename in sample_files:
                if self.samples_yielded >= self.max_samples:
                    break

                file_path = os.path.join(samples_dir, filename)
                metadata = metadata_map.get(filename, {})

                if self.config.modality == "image":
                    try:
                        image = Image.open(file_path)
                        sample = {
                            "image": image,
                            "dataset_name": self.config.name,
                            "media_type": self.config.media_type,
                            **metadata,
                        }
                        yield sample
                        self.samples_yielded += 1
                    except Exception as e:
                        logger.warning(f"Failed to load cached image {filename}: {e}")
                        continue

                elif self.config.modality == "video":
                    try:
                        with open(file_path, "rb") as f:
                            video_bytes = f.read()
                        sample = {
                            "video_bytes": video_bytes,
                            "dataset_name": self.config.name,
                            "media_type": self.config.media_type,
                            **metadata,
                        }
                        yield sample
                        self.samples_yielded += 1
                    except Exception as e:
                        logger.warning(f"Failed to load cached video {filename}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to load cached dataset from {cache_dir}: {e}")
            raise
