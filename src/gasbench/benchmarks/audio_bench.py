"""Audio sample preparation; orchestration is shared with image and video."""

import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import DEFAULT_AUDIO_BATCH_SIZE
from ..constants import AUDIO_NUM_SAMPLES, AUDIO_SAMPLE_RATE, media_type_to_label
from ..dataset.cache import load_audio_sample
from ..processing.media import process_audio_sample
from ..processing.transforms import apply_audio_robustness_augmentations
from .aug_cache import aud_aug_cache_path, write_aug_cache
from .common import BenchmarkRunConfig, use_augmentation_cache, verify_sample
from .inputs import validate_audio_preprocessing
from .prefetch import PrefetchPipeline
from .recording import build_sample_id
from .runner import run_modality_benchmark


def preprocessing_options(session, input_specs):
    config = (
        session.get_preprocessing_config()
        if hasattr(session, "get_preprocessing_config")
        else {}
    )
    validate_audio_preprocessing(config)
    if len(input_specs[0].shape) != 2 or input_specs[0].shape[1] != AUDIO_NUM_SAMPLES:
        raise ValueError(
            "Audio input must contain 96000 samples (mono 16 kHz, six seconds)"
        )
    return {}


class AudioPrefetchPipeline(PrefetchPipeline):
    def __init__(self, *args, num_workers=1, max_queue_size=1, **kwargs):
        super().__init__(
            *args, num_workers=num_workers, max_queue_size=max_queue_size, **kwargs
        )

    def _read_and_preprocess(self, sample, sample_index, dataset_name):
        verify_sample(sample)
        sample_seed = self.seed + sample_index
        path = (
            aud_aug_cache_path(self.aug_cache_dir, build_sample_id(sample), sample_seed)
            if self.robustness_pass and self.aug_cache_dir
            else None
        )
        label = media_type_to_label(sample["media_type"], "audio")
        if path and use_augmentation_cache(sample, path):
            array = np.load(path, allow_pickle=False)
        else:
            try:
                loaded = load_audio_sample(sample)
            except (
                pickle.UnpicklingError,
                EOFError,
                RuntimeError,
                KeyError,
                ValueError,
            ):
                return None
            if loaded["is_preprocessed"]:
                array = loaded["preprocessed_waveform"]
            else:
                array, label = process_audio_sample(
                    loaded, target_sr=AUDIO_SAMPLE_RATE, seed=sample_seed
                )
                if array is None or label is None:
                    return None
            array = np.asarray(array).squeeze()
            self._validate(array)
            if self.robustness_pass:
                array, _, _, _ = apply_audio_robustness_augmentations(
                    array,
                    target_sr=AUDIO_SAMPLE_RATE,
                    seed=sample_seed,
                )
                if path and not self.aug_cache_readonly:
                    write_aug_cache(path, array)
        array = np.asarray(array, dtype=np.float32)
        self._validate(array)
        return {
            "data": array,
            "label": label,
            "sample": sample,
            "sample_index": sample_index,
            "dataset_name": dataset_name,
            "sample_seed": sample_seed,
        }

    @staticmethod
    def _validate(array):
        if array.shape != (AUDIO_NUM_SAMPLES,) or not np.isfinite(array).all():
            raise ValueError(
                "Preprocessed audio must contain 96000 finite mono samples"
            )


PIPELINE = AudioPrefetchPipeline


async def run_audio_benchmark(
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
    records_parquet_path: Optional[str] = None,
    run_id: Optional[str] = None,
    dataset_filters: Optional[list] = None,
    skip_missing: bool = False,
    holdout_weight: float = 1.0,
    holdouts_only: bool = False,
    content_category: Optional[str] = None,
    score_composition: dict = None,
    multiclass_scoring: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_persist=None,
    n_aug_per_dataset: int = 0,
    aug_weight: float = 0.2,
    aug_cache_dir: Optional[str] = None,
    aug_cache_readonly: bool = False,
) -> pd.DataFrame:
    """Compatibility entry point; all modalities share one run lifecycle."""
    batch_size = DEFAULT_AUDIO_BATCH_SIZE if batch_size is None else batch_size
    config = BenchmarkRunConfig(
        modality="audio",
        mode=mode,
        gasstation_only=gasstation_only,
        dataset_config_path=dataset_config,
        holdout_config_path=holdout_config,
        cache_dir=cache_dir,
        hf_token=os.environ.get("HF_TOKEN"),
        batch_size=batch_size,
        augment_level=0,
        crop_prob=0.0,
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
