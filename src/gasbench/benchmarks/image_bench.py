import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_IMAGE_BATCH_SIZE,
)
from ..constants import media_type_to_label
from ..processing.media import process_image_sample
from ..processing.transforms import (
    apply_random_augmentations,
    apply_robustness_augmentations,
)
from .aug_cache import img_aug_cache_path, write_aug_cache
from .common import (
    BenchmarkRunConfig,
    use_augmentation_cache,
    verify_sample,
)
from .prefetch import PrefetchPipeline as BasePrefetchPipeline
from .recording import build_sample_id
from .runner import run_modality_benchmark

_HEAVY_SAMPLE_KEYS = frozenset(("image", "image_bytes", "image_path"))


class PrefetchPipeline(BasePrefetchPipeline):
    def __init__(self, *args, num_workers=8, max_queue_size=8, **kwargs):
        super().__init__(
            *args, num_workers=num_workers, max_queue_size=max_queue_size, **kwargs
        )

    def _read_and_preprocess(self, sample, sample_index, dataset_name):
        """Read file from disk (if lazy), decode, and augment. Runs in worker thread."""
        verify_sample(sample)
        sample_seed = None if self.seed is None else self.seed + sample_index
        cache_path = None
        if self.robustness_pass and self.aug_cache_dir:
            cache_path = img_aug_cache_path(
                self.aug_cache_dir, build_sample_id(sample), self.target_size
            )
        if cache_path is not None and use_augmentation_cache(sample, cache_path):
            aug_hwc = np.load(cache_path, allow_pickle=False)
            if aug_hwc.shape != (*self.target_size, 3):
                raise ValueError(
                    "Cached image shape differs from the model preprocessing contract"
                )
            label = media_type_to_label(sample.get("media_type", "synthetic"), "image")
        else:
            image_path = sample.get("image_path")
            if image_path:
                with open(image_path, "rb") as stream:
                    sample = {**sample, "image": stream.read()}
            image_array, label = process_image_sample(sample)
            if image_array is None or label is None:
                return None
            if self.robustness_pass:
                aug_hwc, _, _, _ = apply_robustness_augmentations(
                    image_array, self.target_size, seed=sample_seed
                )
                if cache_path is not None and not self.aug_cache_readonly:
                    write_aug_cache(cache_path, aug_hwc)
            else:
                aug_hwc, _, _, _ = apply_random_augmentations(
                    image_array,
                    self.target_size,
                    seed=sample_seed,
                    level=self.augment_level,
                    crop_prob=self.crop_prob,
                )
        aug_chw = np.transpose(aug_hwc, (2, 0, 1))

        sample_meta = {k: v for k, v in sample.items() if k not in _HEAVY_SAMPLE_KEYS}

        return {
            "data": aug_chw,
            "label": label,
            "sample": sample_meta,
            "sample_index": sample_index,
            "dataset_name": dataset_name,
            "sample_seed": sample_seed,
        }


async def run_image_benchmark(
    session,
    input_specs,
    benchmark_results: Dict,
    mode: str = "full",
    gasstation_only: bool = False,
    cache_dir: str = "/.cache/gasbench",
    download_latest_gasstation_data: bool = False,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataset_config: Optional[str] = None,
    holdout_config: Optional[str] = None,
    augment_level: Optional[int] = 0,
    crop_prob: float = 0.0,
    records_parquet_path: Optional[str] = None,
    run_id: Optional[str] = None,
    dataset_filters: Optional[list] = None,
    skip_missing: bool = False,
    holdout_weight: float = 1.0,
    holdouts_only: bool = False,
    content_category: Optional[str] = None,
    score_composition: dict = None,
    multiclass_scoring: bool = False,
    n_aug_per_dataset: int = 0,
    aug_weight: float = 0.2,
    aug_cache_dir: Optional[str] = None,
    aug_cache_readonly: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_persist=None,
) -> pd.DataFrame:
    """Compatibility entry point; all modalities share one run lifecycle."""
    batch_size = DEFAULT_IMAGE_BATCH_SIZE if batch_size is None else batch_size
    config = BenchmarkRunConfig(
        modality="image",
        mode=mode,
        gasstation_only=gasstation_only,
        dataset_config_path=dataset_config,
        holdout_config_path=holdout_config,
        cache_dir=cache_dir,
        hf_token=os.environ.get("HF_TOKEN"),
        batch_size=batch_size,
        augment_level=augment_level or 0,
        crop_prob=crop_prob or 0.0,
        records_parquet_path=records_parquet_path,
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        checkpoint_persist=checkpoint_persist,
        dataset_filters=dataset_filters,
        holdout_weight=holdout_weight,
        holdouts_only=holdouts_only,
        content_category=content_category,
        score_composition=score_composition,
        multiclass_scoring=multiclass_scoring,
        n_aug_per_dataset=n_aug_per_dataset,
        aug_weight=aug_weight,
        aug_cache_dir=aug_cache_dir,
        aug_cache_readonly=aug_cache_readonly,
    )
    return await run_modality_benchmark(
        session,
        input_specs,
        benchmark_results,
        config=config,
        seed=seed,
        skip_missing=skip_missing,
        download_latest_gasstation_data=download_latest_gasstation_data,
    )


def preprocessing_options(session, input_specs):
    return {}


PIPELINE = PrefetchPipeline
