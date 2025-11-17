import os
import time
import asyncio
import numpy as np
from typing import Dict, Optional

from ..logger import get_logger
from ..processing.archive import video_archive_manager
from ..processing.media import process_video_bytes_sample
from ..processing.transforms import (
    apply_random_augmentations,
)
from ..config import DEFAULT_VIDEO_BATCH_SIZE
from ..dataset.iterator import DatasetIterator
 
from ..model.inference import process_model_output
from .recording import ResultTracker, log_dataset_summary
from .common import BenchmarkRunConfig, build_plan, create_tracker, finalize_run
import pandas as pd

logger = get_logger(__name__)


def process_video_batch(
    session,
    input_specs,
    batch_videos,
    batch_metadata,
    tracker: ResultTracker,
    batch_id: int,
):
    """push a batch of videos through the model and record rows in tracker."""
    if not batch_videos:
        return

    first = batch_videos[0]
    batch_array = np.empty((len(batch_videos),) + first.shape, dtype=first.dtype)
    for i, vid in enumerate(batch_videos):
        batch_array[i] = vid

    start = time.time()
    outputs = session.run(None, {input_specs[0].name: batch_array})
    batch_inference_time = (time.time() - start) * 1000
    per_sample_time = batch_inference_time / len(batch_videos)

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
            batch_size=len(batch_videos),
            sample_seed=sample_seed,
        )


async def video_prefetcher(
    dataset_iterator,
    dataset_config,
    queue: asyncio.Queue,
    max_queue_size: int = 2,
    seed: int = None,
    target_size=None,
    augment_level: Optional[int] = 0,
    crop_prob: float = 0.0,
):
    """
    Prefetch and preprocess videos from the dataset iterator.
    
    Args:
        dataset_iterator: Iterator over dataset samples
        dataset_config: Configuration for the dataset
        queue: Async queue to put processed videos into
        max_queue_size: Maximum size of prefetch queue
        seed: Optional random seed for reproducible augmentations
        target_size: Optional (H, W) tuple for fixed target size from model specs
    """
    try:
        sample_index = 0
        for sample in dataset_iterator:
            sample_index += 1
            try:
                video_array, label = process_video_bytes_sample(sample)

                if video_array is None or label is None:
                    # Put a skip marker in the queue
                    await queue.put(("skip", None, None, None, None))
                    continue

                try:
                    # video_array is THWC uint8; augment in THWC and transpose once at the end
                    sample_seed = None if seed is None else (seed + sample_index)
                    aug_thwc, _, _, _ = apply_random_augmentations(
                        video_array, target_size, seed=sample_seed, level=augment_level, crop_prob=crop_prob
                    )
                    aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))
                    video_array = np.expand_dims(aug_tchw, 0)  # 1,T,C,H,W
                except Exception as e:
                    logger.error(f"Video augmentation failed: {e}")
                    await queue.put(("skip", None, None, None, None))
                    continue

                await queue.put(("data", video_array, label, sample))
                
            except Exception as e:
                logger.warning(f"Failed to process video sample from {dataset_config.name}: {e}")
                await queue.put(("error", None, None, None, None))
                
    except Exception as e:
        logger.error(f"Prefetcher error for dataset {dataset_config.name}: {e}")
    finally:
        # Signal end of dataset
        await queue.put(("done", None, None, None, None))


async def run_video_benchmark(
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
    """Test model on benchmark video datasets for AI-generated content detection."""
    
    if batch_size is None:
        batch_size = DEFAULT_VIDEO_BATCH_SIZE

    try:
        hf_token = os.environ.get("HF_TOKEN")

        if gasstation_only:
            logger.info("Loading gasstation video datasets only")
        else:
            logger.info("Loading benchmark video datasets")
        
        run_config = BenchmarkRunConfig(
            modality="video",
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
            logger.error("No benchmark video datasets configured")
            benchmark_results["video_results"] = {"error": "No datasets available"}
            return 0.0

        tracker = create_tracker(run_config, plan, input_specs)

        with video_archive_manager(cache_dir=cache_dir) as video_cache:
            skipped_samples = 0
            benchmark_results.setdefault("errors", [])
            logger.info(f"Sampling plan targets {plan.sampling_summary.actual_total_samples} samples across {plan.sampling_summary.num_datasets} datasets")

            for dataset_idx, dataset_config in enumerate(plan.available_datasets):
                dataset_cap = plan.sampling_plan[dataset_config.name]
                logger.info(
                    f"Processing dataset {dataset_idx + 1}/{len(plan.available_datasets)}: "
                    f"{dataset_config.name} ({dataset_cap} samples)"
                )

                dataset_skipped = 0

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

                    prefetch_queue = asyncio.Queue(maxsize=16)
                    prefetch_task = asyncio.create_task(
                        video_prefetcher(
                            dataset_iterator,
                            dataset_config,
                            prefetch_queue,
                            seed=seed,
                            target_size=plan.target_size,
                            augment_level=augment_level,
                            crop_prob=crop_prob,
                        )
                    )

                    sample_index = 0
                    batch_videos = []
                    batch_metadata = []
                    batch_id = 0

                    while True:
                        item = await prefetch_queue.get()
                        item_type = item[0]

                        if item_type == "done":
                            break
                        elif item_type == "skip":
                            dataset_skipped += 1
                            skipped_samples += 1
                            tracker.add_skip(dataset_name=dataset_config.name, sample_index=sample_index + 1, sample={}, reason="prefetch-skip")
                            if dataset_skipped % 10 == 0:
                                logger.debug(
                                    f"Dataset {dataset_config.name}: Skipped {dataset_skipped} samples so far"
                                )
                            continue
                        elif item_type == "error":
                            benchmark_results["errors"].append(f"Video processing error during prefetch")
                            dataset_skipped += 1
                            skipped_samples += 1
                            tracker.add_error(dataset_name=dataset_config.name, sample_index=sample_index + 1, sample={}, error_message="prefetch-error")
                            continue

                        video_array, label, sample = item[1], item[2], item[3]
                        sample_index += 1
                        
                        batch_videos.append(video_array[0])
                        sample_seed = None if seed is None else (seed + sample_index)
                        batch_metadata.append((label, sample, sample_index, dataset_config.name, sample_seed))

                        if len(batch_videos) >= batch_size:
                            batch_id += 1
                            process_video_batch(
                                session, input_specs, batch_videos, batch_metadata, tracker, batch_id
                            )
                            
                            batch_videos = []
                            batch_metadata = []

                            if tracker.count % 500 == 0:
                                logger.info(
                                    f"Progress: {tracker.count} samples"
                                )
                    
                    # Wait for prefetcher to finish
                    await prefetch_task

                    if batch_videos:
                        batch_id += 1
                        process_video_batch(
                            session, input_specs, batch_videos, batch_metadata, tracker, batch_id
                        )

                    log_dataset_summary(logger, tracker, dataset_config.name, include_skipped=True)

                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                    benchmark_results["errors"].append(f"Dataset error for {dataset_config.name}: {str(e)[:100]}")

            cache_info = video_cache.get_cache_info()
            df = finalize_run(
                config=run_config,
                plan=plan, 
                tracker=tracker, 
                benchmark_results=benchmark_results, 
                results_key="video_results", 
                extra_fields={"cache_info": cache_info}
            )

            logger.info(f"Archive cache: {cache_info['unpacked_archives']} archives cached locally")
            if skipped_samples > 0:
                logger.warning(f"Skipped {skipped_samples} samples due to missing/inaccessible archives or processing errors")

            return df

    except Exception as e:
        logger.error(f"Benchmark video testing failed: {e}")
        benchmark_results["video_results"] = {"error": str(e)}
        raise e
