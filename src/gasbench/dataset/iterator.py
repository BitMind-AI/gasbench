"""Dataset iterator for unified access to benchmark datasets."""

import os
import json
from pathlib import Path
from typing import Optional

from PIL import Image

from ..logger import get_logger
from .config import BenchmarkDatasetConfig, IMAGE_BENCHMARK_SIZE, VIDEO_BENCHMARK_SIZE
from .download import download_and_extract
from .cache import save_sample_to_cache, save_dataset_cache_files

logger = get_logger(__name__)


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
        self.max_samples = max_samples or min(
            IMAGE_BENCHMARK_SIZE, VIDEO_BENCHMARK_SIZE
        )
        self.samples_yielded = 0
        self.cache_dir = cache_dir
        self.dataset_dir = f"{cache_dir}/datasets/{dataset_config.name}"
        self.num_weeks = num_weeks

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
        """
        if self._is_cache_complete():
            logger.info(f"✅ Cache already complete for {self.config.name} with {self._get_cached_count()} samples")
            return

        self._download_and_cache()

    def get_samples(self):
        """Generator that yields samples from cached data.
        """
        try:
            if not self._has_cached_dataset():
                logger.warning(f"⚠️  No cached data found for {self.config.name}. Call ensure_cached() first.")
                return

            logger.info(f"📂 Using cached dataset: {self.config.name}")
            yield from self._load_from_cache()

        except Exception as e:
            logger.error(f"❌ Failed to get samples from {self.config.name}: {e}")
            return

    def _download_and_cache(self):
        """Download samples and save them to cache with incremental checkpointing."""
        samples_dir = os.path.join(self.dataset_dir, "samples")
        metadata_file = os.path.join(self.dataset_dir, "sample_metadata.json")

        sample_metadata = {}
        sample_count = 0

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    sample_metadata = json.load(f)
                sample_count = len(sample_metadata)
                logger.info(
                    f"📂 Found partial cache with {sample_count} samples, resuming download..."
                )
            except Exception as e:
                logger.warning(f"Failed to load partial metadata, starting fresh: {e}")
                sample_metadata = {}
                sample_count = 0
        else:
            logger.info(
                f"No cached datasets found.\nDownloading {self.config.name} (max {self.max_samples} samples)"
            )

        if sample_count >= self.max_samples:
            logger.info(f"✅ Cache already complete with {sample_count} samples")
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
        ):
            if sample_count >= self.max_samples:
                break

            filename = save_sample_to_cache(
                sample, self.config, samples_dir, sample_count
            )
            if filename:
                sample_metadata[filename] = {
                    "source_file": sample.get("source_file", ""),
                    "model_name": sample.get("model_name", ""),
                    "media_type": sample.get("media_type", ""),
                }
                sample_count += 1

                # Save metadata incrementally every 50 samples
                if sample_count % 50 == 0:
                    save_dataset_cache_files(
                        self.config, self.dataset_dir, sample_metadata, sample_count
                    )
                    logger.info(
                        f"📥 Downloaded {sample_count}/{self.max_samples} samples (checkpoint saved)..."
                    )

        # Save final metadata
        if sample_count > 0:
            save_dataset_cache_files(
                self.config, self.dataset_dir, sample_metadata, sample_count
            )
            logger.info(
                f"✅ DOWNLOAD COMPLETE: Saved {sample_count} samples to cache for {self.config.name}"
            )
        else:
            logger.warning(f"⚠️  No samples were downloaded for {self.config.name}")

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
        """Check if cached dataset has enough samples for max_samples."""
        try:
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

    def _load_from_cache(self):
        """Load samples from cached dataset locally."""
        try:
            samples_dir = os.path.join(self.dataset_dir, "samples")
            metadata_file = os.path.join(self.dataset_dir, "sample_metadata.json")

            with open(metadata_file, "r") as f:
                metadata_map = json.load(f)

            sample_files = [
                f
                for f in os.listdir(samples_dir)
                if os.path.isfile(os.path.join(samples_dir, f))
            ]

            if self.max_samples:
                sample_files = sample_files[: self.max_samples]

            logger.info(
                f"📂 Loading {len(sample_files)} cached samples for {self.config.name}"
            )

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
            logger.error(f"Failed to load cached dataset {self.config.name}: {e}")
            raise
