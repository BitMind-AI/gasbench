import os
import numpy as np
from typing import Dict, Optional

from ..logger import get_logger
from ..processing.media import process_audio_sample
from ..processing.transforms import apply_audio_robustness_augmentations
from ..dataset.iterator import load_audio_sample

from ._checkpoint import CheckpointError
from .recording import BenchmarkRunRecorder, log_dataset_summary, build_sample_id
from .aug_cache import aud_aug_cache_path, write_aug_cache
from .common import (
    BenchmarkRunConfig,
    build_plan,
    create_tracker,
    create_dataset_iterator,
    verify_sample,
    use_augmentation_cache,
    finalize_run,
    run_batch_and_record,
)
import pandas as pd

logger = get_logger(__name__)

DEFAULT_AUDIO_BATCH_SIZE = 16


def process_batch(
    session,
    input_specs,
    batch_audio,
    batch_metadata,
    tracker: BenchmarkRunRecorder,
    batch_id: int,
    aug_pass: bool = False,
):
    """Push a batch of audio samples through the model and record rows in tracker."""
    if not batch_audio:
        return

    # Stack audio tensors into batch array
    try:
        first = batch_audio[0]
        if hasattr(first, "numpy"):
            batch_audio_np = [b.numpy() for b in batch_audio]
        else:
            batch_audio_np = batch_audio

        # Squeeze any extra dimensions (e.g., (1, 96000) -> (96000,))
        batch_audio_np = [b.squeeze() if b.ndim > 1 else b for b in batch_audio_np]

        batch_array = np.stack(batch_audio_np)
    except CheckpointError:
        raise
    except Exception as e:
        logger.error(f"Failed to stack audio batch: {e}")
        for label, sample, sample_index, dataset_name, sample_seed in batch_metadata:
            tracker.add_error(
                dataset_name=dataset_name,
                sample_index=sample_index,
                sample=sample,
                error_message=f"stack-failed: {str(e)[:160]}",
                aug_pass=aug_pass,
            )
        tracker.checkpoint()
        return

    run_batch_and_record(
        session, input_specs, batch_array, batch_metadata, tracker, batch_id, aug_pass
    )


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
    """Test model on benchmark audio datasets for AI-generated content detection.
    
    Uses binary classification: 0=real, 1=synthetic (semisynthetic treated as synthetic).
    """

    seed = 42 if seed is None else seed

    if batch_size is None:
        batch_size = DEFAULT_AUDIO_BATCH_SIZE

    try:
        hf_token = os.environ.get("HF_TOKEN")

        if gasstation_only:
            logger.info("Loading gasstation audio datasets only")
        else:
            logger.info("Loading benchmark audio datasets")

        run_config = BenchmarkRunConfig(
            modality="audio",
            mode=mode,
            gasstation_only=gasstation_only,
            dataset_config_path=dataset_config,
            holdout_config_path=holdout_config,
            cache_dir=cache_dir,
            hf_token=hf_token,
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

        plan = build_plan(logger, run_config, input_specs)
        if not plan:
            logger.error("No benchmark audio datasets configured")
            benchmark_results["audio_results"] = {"error": "No datasets available"}
            return 0.0

        tracker = create_tracker(
            run_config, plan, input_specs, session=session, seed=seed,
            skip_missing=skip_missing,
            download_latest_gasstation_data=download_latest_gasstation_data,
        )
        benchmark_results["run_id"] = run_config.run_id
        benchmark_results["checkpoint_dir"] = run_config.checkpoint_dir

        # Target sample rate for audio processing
        target_sr = 16000

        benchmark_results.setdefault("errors", [])
        logger.info(
            f"Sampling plan targets {plan.sampling_summary.actual_total_samples} samples across {plan.sampling_summary.num_datasets} datasets"
        )
        for aug_pass in ([False, True] if n_aug_per_dataset > 0 else [False]):
            for dataset_idx, dataset_cfg in enumerate(plan.available_datasets):
                dataset_cap = n_aug_per_dataset if aug_pass else plan.sampling_plan[dataset_cfg.name]
                logger.info(
                    f"{'Robustness pass' if aug_pass else 'Processing dataset'} "
                    f"{dataset_idx + 1}/{len(plan.available_datasets)}: "
                    f"{dataset_cfg.name} ({dataset_cap} samples)"
                )

                try:
                    dataset_iterator = create_dataset_iterator(
                        run_config, plan, dataset_cfg, aug_pass=aug_pass,
                    )

                    if skip_missing and dataset_iterator.get_total_cached_count() == 0:
                        logger.warning(f"Skipping {dataset_cfg.name} (not cached, --skip-missing enabled)")
                        continue

                    batch_audio = []
                    batch_metadata = []
                    batch_id = 0
                    sample_index = 0

                    for sample in dataset_iterator:
                        sample_index += 1
                        if tracker.is_checkpointed(
                            dataset_name=dataset_cfg.name, sample_index=sample_index, sample=sample,
                            aug_pass=aug_pass,
                        ):
                            continue
                        verify_sample(sample)
                        try:
                            audio_sample = load_audio_sample(sample)
                            # Check if sample is already preprocessed
                            if audio_sample.get("is_preprocessed", False):
                                audio_array = audio_sample.get("preprocessed_waveform")
                                label = audio_sample.get("label")

                                if audio_array is None or label is None:
                                    continue

                                if hasattr(audio_array, "numpy"):
                                    audio_array = audio_array.numpy()
                            else:
                                # Process raw audio bytes
                                sample_seed_val = None if seed is None else (seed + sample_index)
                                audio_array, label = process_audio_sample(
                                    audio_sample,
                                    target_sr=target_sr,
                                    seed=sample_seed_val,
                                )

                                if audio_array is None or label is None:
                                    continue

                                if hasattr(audio_array, "numpy"):
                                    audio_array = audio_array.numpy()

                            sample_seed_val = None if seed is None else (seed + sample_index)
                            if aug_pass:
                                cache_path = (
                                    aud_aug_cache_path(
                                        aug_cache_dir, build_sample_id(sample), sample_seed_val
                                    )
                                    if aug_cache_dir else None
                                )
                                if cache_path and use_augmentation_cache(sample, cache_path):
                                    audio_array = np.load(cache_path, allow_pickle=False)
                                else:
                                    audio_array, _, _, _ = apply_audio_robustness_augmentations(
                                        np.asarray(audio_array).squeeze(),
                                        target_sr=target_sr,
                                        seed=sample_seed_val,
                                    )
                                    if cache_path and not aug_cache_readonly:
                                        write_aug_cache(cache_path, audio_array)
                            batch_audio.append(audio_array)
                            batch_metadata.append(
                                (label, sample, sample_index, dataset_cfg.name, sample_seed_val)
                            )

                            if len(batch_audio) >= batch_size:
                                batch_id += 1
                                process_batch(
                                    session,
                                    input_specs,
                                    batch_audio,
                                    batch_metadata,
                                    tracker,
                                    batch_id,
                                    aug_pass=aug_pass,
                                )
                                batch_audio = []
                                batch_metadata = []

                                if tracker.count % 500 == 0:
                                    logger.info(f"Progress: {tracker.count} samples")

                        except CheckpointError:
                            raise
                        except Exception as e:
                            logger.warning(
                                f"Failed to process audio sample from {dataset_cfg.name}: {e}"
                            )
                            benchmark_results["errors"].append(
                                f"Audio processing error: {str(e)[:100]}"
                            )

                    # Process remaining samples
                    if batch_audio:
                        batch_id += 1
                        process_batch(
                            session,
                            input_specs,
                            batch_audio,
                            batch_metadata,
                            tracker,
                            batch_id,
                            aug_pass=aug_pass,
                        )

                    log_dataset_summary(
                        logger, tracker, dataset_cfg.name, include_skipped=False
                    )

                except CheckpointError:
                    raise
                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_cfg.name}: {e}")
                    benchmark_results["errors"].append(
                        f"Dataset error for {dataset_cfg.name}: {str(e)[:100]}"
                    )

        df = finalize_run(
            config=run_config,
            plan=plan,
            tracker=tracker,
            benchmark_results=benchmark_results,
            results_key="audio_results",
            extra_fields=None,
        )
        return df

    except CheckpointError:
        raise
    except Exception as e:
        logger.error(f"Benchmark audio testing failed: {e}")
        benchmark_results["audio_results"] = {"error": str(e)}
        raise e
