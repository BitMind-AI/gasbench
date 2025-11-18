import os
import time
import traceback
import numpy as np
from typing import Dict, Optional

from ..logger import get_logger
from ..processing.media import process_image_sample
from ..processing.transforms import (
    apply_random_augmentations,
)
from ..config import (
    DEFAULT_IMAGE_BATCH_SIZE,
)
 
from ..dataset.iterator import DatasetIterator
 
from ..model.inference import process_model_output
from .recording import BenchmarkRunRecorder, log_dataset_summary
from .common import BenchmarkRunConfig, build_plan, create_tracker, finalize_run
import pandas as pd

logger = get_logger(__name__)


def process_batch(
    session,
    input_specs,
    batch_images,
    batch_metadata,
    tracker: BenchmarkRunRecorder,
    batch_id: int,
):
    """push a batch of images through the model and record rows in tracker."""
    if not batch_images:
        return

    first = batch_images[0]
    batch_array = np.empty((len(batch_images),) + first.shape, dtype=first.dtype)
    for i, img in enumerate(batch_images):
        batch_array[i] = img

    start = time.time()
    outputs = session.run(None, {input_specs[0].name: batch_array})
    batch_inference_time = (time.time() - start) * 1000
    per_sample_time = batch_inference_time / len(batch_images)

    for i, (label, sample, sample_index, dataset_name, sample_seed) in enumerate(batch_metadata):
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
            batch_size=len(batch_images),
            sample_seed=sample_seed,
        )


async def run_image_benchmark(
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
    augment_level: Optional[int] = 0,
    crop_prob: float = 0.0,
    records_parquet_path: Optional[str] = None,
) -> pd.DataFrame:
    """Test model on benchmark image datasets for AI-generated content detection."""
    
    if batch_size is None:
        batch_size = DEFAULT_IMAGE_BATCH_SIZE

    try:
        hf_token = os.environ.get("HF_TOKEN")

        if gasstation_only:
            logger.info("Loading gasstation image datasets only")
        else:
            logger.info("Loading benchmark image datasets")

        run_config = BenchmarkRunConfig(
            modality="image",
            mode=mode,
            gasstation_only=gasstation_only,
            dataset_config_path=dataset_config,
            holdout_config_path=holdout_config,
            cache_dir=cache_dir,
            cache_policy_path=cache_policy,
            hf_token=hf_token,
            batch_size=batch_size,
            augment_level=augment_level or 0,
            crop_prob=crop_prob or 0.0,
            records_parquet_path=records_parquet_path,
        )

        plan = build_plan(logger, run_config, input_specs)
        if not plan:
            logger.error("No benchmark image datasets configured")
            benchmark_results["image_results"] = {"error": "No datasets available"}
            return 0.0

        tracker = create_tracker(run_config, plan, input_specs)

        benchmark_results.setdefault("errors", [])
        for dataset_idx, dataset_config in enumerate(plan.available_datasets):
            dataset_cap = plan.sampling_plan[dataset_config.name]
            logger.info(
                f"Processing dataset {dataset_idx + 1}/{len(plan.available_datasets)}: "
                f"{dataset_config.name} ({dataset_cap} samples)"
            )

            try:
                # Download gasstation datasets only if flag is set; always download regular datasets if not cached
                is_gasstation = "gasstation" in dataset_config.name.lower()
                should_download = download_latest_gasstation_data if is_gasstation else True

                dataset_iterator = DatasetIterator(
                    dataset_config, 
                    max_samples=dataset_cap, 
                    cache_dir=cache_dir,
                    download=should_download,
                    cache_policy=cache_policy,
                    hf_token=hf_token,
                    seed=seed,
                )

                batch_images = []
                batch_metadata = []

                sample_index = 0
                batch_id = 0
                for sample in dataset_iterator:
                    sample_index += 1
                    try:
                        image_array, label = process_image_sample(sample)

                        if image_array is None or label is None:
                            continue

                        try:
                            sample_seed = None if seed is None else (seed + sample_index)
                            aug_hwc, _, _, _ = apply_random_augmentations(
                                image_array, plan.target_size, seed=sample_seed, level=augment_level, crop_prob=crop_prob
                            )
                            aug_chw = np.transpose(aug_hwc, (2, 0, 1))
                        except Exception as e:
                            logger.error(f"Augmentation failed: {e}\n{traceback.format_exc()}")
                            continue

                        batch_images.append(aug_chw)
                        batch_metadata.append((label, sample, sample_index, dataset_config.name, sample_seed))

                        if len(batch_images) >= batch_size:
                            batch_id += 1
                            process_batch(
                                session, input_specs, batch_images, batch_metadata, tracker, batch_id
                            )
                            
                            batch_images = []
                            batch_metadata = []

                            if tracker.count % 500 == 0:
                                logger.info(f"Progress: {tracker.count} samples")

                    except Exception as e:
                        logger.warning(f"Failed to process image sample from {dataset_config.name}: {e}\n{traceback.format_exc()}")
                        benchmark_results["errors"].append(f"Image processing error: {str(e)[:200]}")

                if batch_images:
                    batch_id += 1
                    process_batch(
                        session, input_specs, batch_images, batch_metadata, tracker, batch_id
                    )

                log_dataset_summary(logger, tracker, dataset_config.name, include_skipped=False)

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                benchmark_results["errors"].append(f"Dataset error for {dataset_config.name}: {str(e)[:100]}")

        df = finalize_run(
            config=run_config,
            plan=plan,
            tracker=tracker,
            benchmark_results=benchmark_results,
            results_key="image_results", 
            extra_fields=None
        )
        return df

    except Exception as e:
        logger.error(f"Benchmark image testing failed: {e}")
        benchmark_results["image_results"] = {"error": str(e)}
        raise e
