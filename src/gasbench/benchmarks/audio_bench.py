import os
import time
import traceback
import numpy as np
from typing import Dict, Optional

from ..logger import get_logger
from ..processing.media import process_audio_sample
from ..dataset.iterator import DatasetIterator

from .utils.inference import process_model_output
from .recording import BenchmarkRunRecorder, log_dataset_summary
from .common import BenchmarkRunConfig, build_plan, create_tracker, finalize_run
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
    except Exception as e:
        logger.error(f"Failed to stack audio batch: {e}")
        return

    start = time.time()
    outputs = None
    try:
        outputs = session.run(None, {input_specs[0].name: batch_array})
    except Exception as e:
        logger.error(f"Inference failed: {e} (batch shape: {batch_array.shape})")
        for i, (label, sample, sample_index, dataset_name, sample_seed) in enumerate(
            batch_metadata
        ):
            tracker.add_error(
                dataset_name=dataset_name,
                sample_index=sample_index,
                sample=sample,
                error_message=f"inference-failed: {str(e)[:160]}",
            )
        return
    batch_inference_time = (time.time() - start) * 1000
    per_sample_time = batch_inference_time / len(batch_audio)

    for i, (label, sample, sample_index, dataset_name, sample_seed) in enumerate(
        batch_metadata
    ):
        predicted, pred_probs = process_model_output(outputs[0][i])

        tracker.add_ok(
            dataset_name=dataset_name,
            sample_index=sample_index,
            sample=sample,
            label=label,
            predicted=predicted,
            probs=pred_probs,
            inference_time_ms=per_sample_time,
            batch_inference_time_ms=batch_inference_time,
            batch_id=batch_id,
            batch_size=len(batch_audio),
            sample_seed=sample_seed,
        )


async def run_audio_benchmark(
    session,
    input_specs,
    benchmark_results: Dict,
    mode: str = "full",
    gasstation_only: bool = False,
    cache_dir: str = "/.cache/gasbench",
    download_latest_gasstation_data: bool = False,
    cache_policy: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataset_config: Optional[str] = None,
    holdout_config: Optional[str] = None,
    records_parquet_path: Optional[str] = None,
) -> pd.DataFrame:
    """Test model on benchmark audio datasets for AI-generated content detection.
    
    Uses binary classification: 0=real, 1=synthetic (semisynthetic treated as synthetic).
    """

    if batch_size is None:
        batch_size = DEFAULT_AUDIO_BATCH_SIZE

    try:
        hf_token = os.environ.get("HF_TOKEN")

        if gasstation_only:
            logger.info("Loading gasstation audio datasets only")
        else:
            logger.info("Loading benchmark audio datasets")

        # Build run config - audio doesn't use augmentation or crop
        run_config = BenchmarkRunConfig(
            modality="audio",
            mode=mode,
            gasstation_only=gasstation_only,
            dataset_config_path=dataset_config,
            holdout_config_path=holdout_config,
            cache_dir=cache_dir,
            cache_policy_path=cache_policy,
            hf_token=hf_token,
            batch_size=batch_size,
            augment_level=0,
            crop_prob=0.0,
            records_parquet_path=records_parquet_path,
        )

        plan = build_plan(logger, run_config, input_specs)
        if not plan:
            logger.error("No benchmark audio datasets configured")
            benchmark_results["audio_results"] = {"error": "No datasets available"}
            return 0.0

        tracker = create_tracker(run_config, plan, input_specs)

        # Target sample rate for audio processing
        target_sr = 16000

        benchmark_results.setdefault("errors", [])
        logger.info(
            f"Sampling plan targets {plan.sampling_summary.actual_total_samples} samples across {plan.sampling_summary.num_datasets} datasets"
        )
        for dataset_idx, dataset_cfg in enumerate(plan.available_datasets):
            dataset_cap = plan.sampling_plan[dataset_cfg.name]
            logger.info(
                f"Processing dataset {dataset_idx + 1}/{len(plan.available_datasets)}: "
                f"{dataset_cfg.name} ({dataset_cap} samples)"
            )

            try:
                # Download gasstation datasets only if flag is set
                is_gasstation = "gasstation" in dataset_cfg.name.lower()
                should_download = (
                    download_latest_gasstation_data if is_gasstation else True
                )

                dataset_iterator = DatasetIterator(
                    dataset_cfg,
                    max_samples=dataset_cap,
                    cache_dir=cache_dir,
                    download=should_download,
                    cache_policy=cache_policy,
                    hf_token=hf_token,
                    seed=seed,
                )

                batch_audio = []
                batch_metadata = []
                batch_id = 0
                sample_index = 0

                for sample in dataset_iterator:
                    sample_index += 1
                    try:
                        # Check if sample is already preprocessed
                        if sample.get("is_preprocessed", False):
                            audio_array = sample.get("preprocessed_waveform")
                            label = sample.get("label")

                            if audio_array is None or label is None:
                                continue

                            if hasattr(audio_array, "numpy"):
                                audio_array = audio_array.numpy()
                        else:
                            # Process raw audio bytes
                            sample_seed_val = None if seed is None else (seed + sample_index)
                            audio_array, label = process_audio_sample(
                                sample, 
                                target_sr=target_sr,
                                seed=sample_seed_val,
                            )

                            if audio_array is None or label is None:
                                continue

                            if hasattr(audio_array, "numpy"):
                                audio_array = audio_array.numpy()

                        sample_seed_val = None if seed is None else (seed + sample_index)
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
                            )
                            batch_audio = []
                            batch_metadata = []

                            if tracker.count % 500 == 0:
                                logger.info(f"Progress: {tracker.count} samples")

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
                    )

                log_dataset_summary(
                    logger, tracker, dataset_cfg.name, include_skipped=False
                )

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

    except Exception as e:
        logger.error(f"Benchmark audio testing failed: {e}")
        benchmark_results["audio_results"] = {"error": str(e)}
        raise e
