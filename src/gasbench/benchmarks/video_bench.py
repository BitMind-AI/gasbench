import os
import time
import asyncio
import numpy as np
import traceback
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import threading

from ..logger import get_logger
from ..processing.archive import video_archive_manager
from ..processing.media import process_video_bytes_sample, process_video_frames_sample
from ..processing.transforms import (
    apply_random_augmentations,
)
from ..config import DEFAULT_VIDEO_BATCH_SIZE
from ..dataset.iterator import DatasetIterator

from .utils.inference import process_model_output
from .recording import BenchmarkRunRecorder, log_dataset_summary
from .common import BenchmarkRunConfig, build_plan, create_tracker, finalize_run
import pandas as pd

logger = get_logger(__name__)


class VideoPrefetchPipeline:
    """Pipeline for parallel loading and preprocessing of video samples."""

    def __init__(
        self,
        dataset_iterator,
        target_size,
        batch_size,
        seed,
        augment_level,
        crop_prob,
        num_workers=3,
        max_queue_size=6,
    ):
        self.dataset_iterator = dataset_iterator
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed
        self.augment_level = augment_level
        self.crop_prob = crop_prob
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

        self.batch_queue = Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.error = None

        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()

    def _preprocess_sample(self, sample, sample_index, dataset_name):
        """Preprocess a single sample (runs in worker thread)."""
        try:
            if "video_frames" in sample:
                video_array, label = process_video_frames_sample(sample)
            else:
                video_array, label = process_video_bytes_sample(sample)

            if video_array is None or label is None:
                return None

            sample_seed = None if self.seed is None else (self.seed + sample_index)
            try:
                aug_thwc, _, _, _ = apply_random_augmentations(
                    video_array,
                    self.target_size,
                    seed=sample_seed,
                    level=self.augment_level,
                    crop_prob=self.crop_prob,
                )
            except Exception as e:
                logger.error(f"Video augmentation failed: {e}")
                return None

            aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))

            return {
                "video": aug_tchw,
                "label": label,
                "sample": sample,
                "sample_index": sample_index,
                "dataset_name": dataset_name,
                "sample_seed": sample_seed,
            }
        except Exception as e:
            logger.warning(f"Failed to preprocess video sample: {e}")
            return None

    def _producer_loop(self):
        """Main loop that feeds samples to workers and batches results."""
        try:
            batch = []
            sample_index = 0
            dataset_name = getattr(self.dataset_iterator.config, "name", "unknown")

            futures = []
            for sample in self.dataset_iterator:
                if self.stop_event.is_set():
                    break

                sample_index += 1

                future = self.executor.submit(
                    self._preprocess_sample, sample, sample_index, dataset_name
                )
                futures.append(future)

                if len(futures) >= self.num_workers * 2:
                    for future in futures:
                        if self.stop_event.is_set():
                            break
                        result = future.result()
                        if result is not None:
                            batch.append(result)
                            if len(batch) >= self.batch_size:
                                self.batch_queue.put(batch)
                                batch = []
                    futures = []

            for future in futures:
                if self.stop_event.is_set():
                    break
                result = future.result()
                if result is not None:
                    batch.append(result)
                    if len(batch) >= self.batch_size:
                        self.batch_queue.put(batch)
                        batch = []

            if batch and not self.stop_event.is_set():
                self.batch_queue.put(batch)

            self.batch_queue.put(None)

        except Exception as e:
            self.error = e
            logger.error(
                f"Error in video prefetch pipeline: {e}\n{traceback.format_exc()}"
            )
            self.batch_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.error:
            raise self.error

        try:
            batch = self.batch_queue.get(timeout=300)
            if batch is None:
                raise StopIteration
            return batch
        except Empty:
            logger.error("Timeout waiting for batch from video prefetch pipeline")
            raise StopIteration

    def close(self):
        """Clean up resources."""
        self.stop_event.set()
        self.executor.shutdown(wait=False)

        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except Empty:
                break


def process_video_batch(
    session,
    input_specs,
    batch_videos,
    batch_metadata,
    tracker: BenchmarkRunRecorder,
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
    outputs = None
    try:
        outputs = session.run(None, {input_specs[0].name: batch_array})
    except Exception as e:
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
    per_sample_time = batch_inference_time / len(batch_videos)

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
            batch_size=len(batch_videos),
            sample_seed=sample_seed,
        )


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
    run_id: Optional[str] = None,
    dataset_filters: Optional[list] = None,
    skip_missing: bool = False,
    holdout_weight: float = 1.0,
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
            run_id=run_id,
            dataset_filters=dataset_filters,
            holdout_weight=holdout_weight,
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
            logger.info(
                f"Sampling plan targets {plan.sampling_summary.actual_total_samples} samples across {plan.sampling_summary.num_datasets} datasets"
            )

            for dataset_idx, dataset_config in enumerate(plan.available_datasets):
                dataset_cap = plan.sampling_plan[dataset_config.name]
                logger.info(
                    f"Processing dataset {dataset_idx + 1}/{len(plan.available_datasets)}: "
                    f"{dataset_config.name} ({dataset_cap} samples)"
                )

                dataset_skipped = 0

                try:
                    is_gasstation = "gasstation" in dataset_config.name.lower()
                    if skip_missing:
                        should_download = False
                    else:
                        should_download = (
                            download_latest_gasstation_data if is_gasstation else True
                        )

                    dataset_iterator = DatasetIterator(
                        dataset_config,
                        max_samples=dataset_cap,
                        cache_dir=cache_dir,
                        download=should_download,
                        cache_policy=cache_policy,
                        hf_token=hf_token,
                        seed=seed,
                    )

                    if skip_missing and dataset_iterator.get_total_cached_count() == 0:
                        logger.warning(f"Skipping {dataset_config.name} (not cached, --skip-missing enabled)")
                        continue

                    pipeline = VideoPrefetchPipeline(
                        dataset_iterator=dataset_iterator,
                        target_size=plan.target_size,
                        batch_size=batch_size,
                        seed=seed,
                        augment_level=augment_level,
                        crop_prob=crop_prob,
                        num_workers=3,
                        max_queue_size=6,
                    )

                    batch_id = 0
                    try:
                        for batch_data in pipeline:
                            batch_id += 1

                            batch_videos = [item["video"] for item in batch_data]
                            batch_metadata = [
                                (
                                    item["label"],
                                    item["sample"],
                                    item["sample_index"],
                                    item["dataset_name"],
                                    item["sample_seed"],
                                )
                                for item in batch_data
                            ]

                            process_video_batch(
                                session,
                                input_specs,
                                batch_videos,
                                batch_metadata,
                                tracker,
                                batch_id,
                            )

                            if tracker.count % 500 == 0:
                                logger.info(f"Progress: {tracker.count} samples")

                    finally:
                        pipeline.close()

                        log_dataset_summary(
                            logger, tracker, dataset_config.name, include_skipped=True
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to process dataset {dataset_config.name}: {e}"
                    )
                    benchmark_results["errors"].append(
                        f"Dataset error for {dataset_config.name}: {str(e)[:100]}"
                    )

            cache_info = video_cache.get_cache_info()
            df = finalize_run(
                config=run_config,
                plan=plan,
                tracker=tracker,
                benchmark_results=benchmark_results,
                results_key="video_results",
                extra_fields={"cache_info": cache_info},
            )

            logger.info(
                f"Archive cache: {cache_info['unpacked_archives']} archives cached locally"
            )
            if skipped_samples > 0:
                logger.warning(
                    f"Skipped {skipped_samples} samples due to missing/inaccessible archives or processing errors"
                )

            return df

    except Exception as e:
        logger.error(f"Benchmark video testing failed: {e}")
        benchmark_results["video_results"] = {"error": str(e)}
        raise e
