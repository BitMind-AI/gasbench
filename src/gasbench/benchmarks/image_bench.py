import os
import time
import traceback
import numpy as np
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from queue import Queue, Empty
import threading

from ..logger import get_logger
from ..processing.media import process_image_sample
from ..processing.transforms import (
    apply_random_augmentations,
)
from ..config import (
    DEFAULT_IMAGE_BATCH_SIZE,
)

from ..dataset.iterator import DatasetIterator

from .utils.inference import process_model_output
from .recording import BenchmarkRunRecorder, log_dataset_summary
from .common import BenchmarkRunConfig, build_plan, create_tracker, finalize_run
import pandas as pd

logger = get_logger(__name__)

_HEAVY_SAMPLE_KEYS = frozenset(("image", "image_bytes", "image_path"))


class PrefetchPipeline:
    """Pipeline for parallel loading and preprocessing of image samples.

    When the iterator yields lazy samples (image_path instead of image bytes),
    file I/O is performed inside worker threads so that multiple disk reads from
    network volumes happen concurrently.
    """

    def __init__(
        self,
        dataset_iterator,
        target_size,
        batch_size,
        seed,
        augment_level,
        crop_prob,
        num_workers=8,
        max_queue_size=8,
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

    def _read_and_preprocess(self, sample, sample_index, dataset_name):
        """Read file from disk (if lazy), decode, and augment. Runs in worker thread."""
        try:
            image_path = sample.get("image_path")
            if image_path:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                sample = {**sample, "image": image_bytes}

            image_array, label = process_image_sample(sample)
            if image_array is None or label is None:
                return None

            sample_seed = None if self.seed is None else (self.seed + sample_index)
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
                "image": aug_chw,
                "label": label,
                "sample": sample_meta,
                "sample_index": sample_index,
                "dataset_name": dataset_name,
                "sample_seed": sample_seed,
            }
        except Exception as e:
            logger.warning(f"Failed to preprocess sample {sample_index}: {e}")
            return None

    def _producer_loop(self):
        """Read + preprocess samples in parallel with bounded concurrency."""
        try:
            dataset_name = getattr(self.dataset_iterator.config, "name", "unknown")
            max_in_flight = self.num_workers * 4

            sample_iter = enumerate(self.dataset_iterator, 1)
            pending = set()
            exhausted = False
            batch = []

            while not self.stop_event.is_set():
                while len(pending) < max_in_flight and not exhausted:
                    try:
                        idx, sample = next(sample_iter)
                        future = self.executor.submit(
                            self._read_and_preprocess, sample, idx, dataset_name
                        )
                        pending.add(future)
                    except StopIteration:
                        exhausted = True
                        break

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    if self.stop_event.is_set():
                        break
                    try:
                        result = future.result()
                    except Exception:
                        continue
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
            logger.error(f"Error in prefetch pipeline: {e}\n{traceback.format_exc()}")
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
            logger.error("Timeout waiting for batch from prefetch pipeline")
            raise StopIteration

    def close(self):
        """Clean up resources."""
        self.stop_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)

        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except Empty:
                break


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
    run_id: Optional[str] = None,
    dataset_filters: Optional[list] = None,
    skip_missing: bool = False,
    holdout_weight: float = 1.0,
    holdouts_only: bool = False,
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
            run_id=run_id,
            dataset_filters=dataset_filters,
            holdout_weight=holdout_weight,
            holdouts_only=holdouts_only,
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
                    lazy_read=True,
                )

                if skip_missing and dataset_iterator.get_total_cached_count() == 0:
                    logger.warning(f"Skipping {dataset_config.name} (not cached, --skip-missing enabled)")
                    continue

                pipeline = PrefetchPipeline(
                    dataset_iterator=dataset_iterator,
                    target_size=plan.target_size,
                    batch_size=batch_size,
                    seed=seed,
                    augment_level=augment_level,
                    crop_prob=crop_prob,
                )

                batch_id = 0
                try:
                    for batch_data in pipeline:
                        batch_id += 1

                        batch_images = [item["image"] for item in batch_data]
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

                        process_batch(
                            session,
                            input_specs,
                            batch_images,
                            batch_metadata,
                            tracker,
                            batch_id,
                        )

                        if tracker.count % 500 == 0:
                            logger.info(f"Progress: {tracker.count} samples")

                finally:
                    pipeline.close()

                log_dataset_summary(
                    logger, tracker, dataset_config.name, include_skipped=False
                )

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                benchmark_results["errors"].append(
                    f"Dataset error for {dataset_config.name}: {str(e)[:100]}"
                )

        df = finalize_run(
            config=run_config,
            plan=plan,
            tracker=tracker,
            benchmark_results=benchmark_results,
            results_key="image_results",
            extra_fields=None,
        )
        return df

    except Exception as e:
        logger.error(f"Benchmark image testing failed: {e}")
        benchmark_results["image_results"] = {"error": str(e)}
        raise e
