import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import DEFAULT_VIDEO_BATCH_SIZE
from ..constants import media_type_to_label
from ..processing.media import process_video_bytes_sample, process_video_frames_sample
from ..processing.transforms import (
    apply_random_augmentations,
    apply_video_robustness_augmentations,
)
from .aug_cache import vid_aug_cache_path, write_aug_cache
from .common import (
    BenchmarkRunConfig,
    use_augmentation_cache,
    verify_sample,
)
from .prefetch import PrefetchPipeline as BasePrefetchPipeline
from .recording import build_sample_id
from .runner import run_modality_benchmark

_HEAVY_VIDEO_KEYS = frozenset(("video_bytes", "video_path"))


class VideoPrefetchPipeline(BasePrefetchPipeline):
    def __init__(self, *args, num_frames=16, frame_rate=None, **kwargs):
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        super().__init__(*args, **kwargs)

    def _read_and_preprocess(self, sample, sample_index, dataset_name):
        """Read file from disk (if lazy), decode, and augment. Runs in worker thread."""
        verify_sample(sample)
        sample_seed = None if self.seed is None else (self.seed + sample_index)
        cache_path = None
        if self.robustness_pass and self.aug_cache_dir:
            cache_path = vid_aug_cache_path(
                self.aug_cache_dir,
                build_sample_id(sample),
                self.target_size,
                num_frames=self.num_frames,
                frame_rate=self.frame_rate,
            )

        # Validate the frozen source and cache before loading cached frames.
        # A cache hit already contains the frames needed for inference.
        if cache_path is not None and use_augmentation_cache(sample, cache_path):
            aug_thwc = np.load(cache_path, allow_pickle=False)
            if aug_thwc.shape != (self.num_frames, *self.target_size, 3):
                raise ValueError(
                    "Cached video shape differs from the model preprocessing contract"
                )
            label = media_type_to_label(sample.get("media_type", "synthetic"), "video")
        else:
            video_path = sample.get("video_path")
            if video_path:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                sample = {**sample, "video_bytes": video_bytes}

            if "video_frames" in sample:
                video_array, label = process_video_frames_sample(
                    sample, num_frames=self.num_frames
                )
            else:
                video_array, label = process_video_bytes_sample(
                    sample, num_frames=self.num_frames, frame_rate=self.frame_rate
                )

            if video_array is None or label is None:
                return None

            if self.robustness_pass:
                aug_thwc, _, _, _ = apply_video_robustness_augmentations(
                    video_array, self.target_size, seed=sample_seed
                )
                if cache_path is not None and not self.aug_cache_readonly:
                    write_aug_cache(cache_path, aug_thwc)
            else:
                aug_thwc, _, _, _ = apply_random_augmentations(
                    video_array,
                    self.target_size,
                    seed=sample_seed,
                    level=self.augment_level,
                    crop_prob=self.crop_prob,
                )

        aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))

        sample_meta = {k: v for k, v in sample.items() if k not in _HEAVY_VIDEO_KEYS}

        return {
            "data": aug_tchw,
            "label": label,
            "sample": sample_meta,
            "sample_index": sample_index,
            "dataset_name": dataset_name,
            "sample_seed": sample_seed,
        }


async def run_video_benchmark(
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
    batch_size = DEFAULT_VIDEO_BATCH_SIZE if batch_size is None else batch_size
    config = BenchmarkRunConfig(
        modality="video",
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


PIPELINE = VideoPrefetchPipeline


def preprocessing_options(session, input_specs):
    from .inputs import video_preprocessing

    return video_preprocessing(session, input_specs)
