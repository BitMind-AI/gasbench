import os
import time
import asyncio
import numpy as np
import traceback
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from queue import Queue, Empty
import threading

from ..logger import get_logger
from ..processing.archive import video_archive_manager
from ..processing.media import process_video_bytes_sample, process_video_frames_sample
from ..processing.transforms import (
    apply_random_augmentations,
    apply_video_robustness_augmentations,
    extract_num_frames_from_input_specs,
)
from ..config import DEFAULT_VIDEO_BATCH_SIZE
from ..constants import MAX_VIDEO_NUM_FRAMES
from ..dataset.iterator import DatasetIterator

from .utils.inference import process_model_output
from .recording import BenchmarkRunRecorder, log_dataset_summary, build_sample_id
from .common import BenchmarkRunConfig, build_plan, create_tracker, finalize_run

_VID_AUG_VERSION = "vid_v1"


def _aug_cache_path(cache_dir: str, sample_id: str) -> str:
    return os.path.join(cache_dir, sample_id[:2], f"{sample_id}_{_VID_AUG_VERSION}.npy")


def _write_aug_cache(path: str, array) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        import numpy as _np
        _np.save(tmp, array)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
import pandas as pd

logger = get_logger(__name__)

_HEAVY_VIDEO_KEYS = frozenset(("video_bytes", "video_path"))


class VideoPrefetchPipeline:
    """Pipeline for parallel loading and preprocessing of video samples.

    When the iterator yields lazy samples (video_path instead of video_bytes),
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
        num_workers=4,
        max_queue_size=6,
        num_frames=16,
        frame_rate=None,
        robustness_pass=False,
        aug_cache_dir=None,
    ):
        self.dataset_iterator = dataset_iterator
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed
        self.augment_level = augment_level
        self.crop_prob = crop_prob
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.robustness_pass = robustness_pass
        self.aug_cache_dir = aug_cache_dir

        self.batch_queue = Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.error = None

        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()

    def _read_and_preprocess(self, sample, sample_index, dataset_name):
        """Read file from disk (if lazy), decode, and augment. Runs in worker thread."""
        try:
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

            sample_seed = None if self.seed is None else (self.seed + sample_index)
            try:
                if self.robustness_pass:
                    if self.aug_cache_dir:
                        sid = build_sample_id(sample)
                        cache_path = _aug_cache_path(self.aug_cache_dir, sid)
                        if os.path.exists(cache_path):
                            aug_thwc = np.load(cache_path)
                        else:
                            aug_thwc, _, _, _ = apply_video_robustness_augmentations(
                                video_array, self.target_size, seed=sample_seed
                            )
                            _write_aug_cache(cache_path, aug_thwc)
                    else:
                        aug_thwc, _, _, _ = apply_video_robustness_augmentations(
                            video_array,
                            self.target_size,
                            seed=sample_seed,
                        )
                else:
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

            sample_meta = {k: v for k, v in sample.items() if k not in _HEAVY_VIDEO_KEYS}

            return {
                "video": aug_tchw,
                "label": label,
                "sample": sample_meta,
                "sample_index": sample_index,
                "dataset_name": dataset_name,
                "sample_seed": sample_seed,
            }
        except Exception as e:
            logger.warning(f"Failed to preprocess video sample {sample_index}: {e}")
            return None

    def _producer_loop(self):
        """Read + preprocess samples in parallel with bounded concurrency."""
        try:
            dataset_name = getattr(self.dataset_iterator.config, "name", "unknown")
            max_in_flight = self.num_workers * 3

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
        self.executor.shutdown(wait=False, cancel_futures=True)

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
    aug_pass: bool = False,
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
            aug_pass=aug_pass,
        )


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
    n_aug_per_dataset: int = 0,
    aug_weight: float = 0.2,
    aug_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Test model on benchmark video datasets for AI-generated content detection."""

    if batch_size is None:
        batch_size = DEFAULT_VIDEO_BATCH_SIZE

    # Resolve num_frames and frame_rate from input shape / preprocessing config.
    # If frame_rate is not specified, frames are taken sequentially at native video fps.
    num_frames = extract_num_frames_from_input_specs(input_specs)
    frame_rate = None
    if hasattr(session, "get_preprocessing_config"):
        preproc = session.get_preprocessing_config()
        if num_frames is None:
            num_frames = preproc.get("num_frames", 16)
        frame_rate = preproc.get("frame_rate")
    if num_frames is None:
        num_frames = 16
    if num_frames > MAX_VIDEO_NUM_FRAMES:
        logger.warning(
            f"num_frames={num_frames} exceeds maximum allowed ({MAX_VIDEO_NUM_FRAMES}). "
            f"Clamping to {MAX_VIDEO_NUM_FRAMES}."
        )
        num_frames = MAX_VIDEO_NUM_FRAMES
    logger.info(f"Video preprocessing: num_frames={num_frames}, frame_rate={frame_rate}")

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
            hf_token=hf_token,
            batch_size=batch_size,
            augment_level=augment_level or 0,
            crop_prob=crop_prob or 0.0,
            records_parquet_path=records_parquet_path,
            run_id=run_id,
            dataset_filters=dataset_filters,
            holdout_weight=holdout_weight,
            holdouts_only=holdouts_only,
            content_category=content_category,
            score_composition=score_composition,
            n_aug_per_dataset=n_aug_per_dataset,
            aug_weight=aug_weight,
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
                        hf_token=hf_token,
                        seed=seed,
                        lazy_read=True,
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
                        num_frames=num_frames,
                        frame_rate=frame_rate,
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

            if n_aug_per_dataset > 0:
                aug_seed = seed if seed is not None else 42
                logger.info(
                    f"Starting video augmentation robustness pass: "
                    f"{n_aug_per_dataset} samples/dataset"
                    + (f" (aug_cache_dir={aug_cache_dir})" if aug_cache_dir else "")
                )
                for dataset_idx, dataset_config in enumerate(plan.available_datasets):
                    logger.info(
                        f"Robustness pass {dataset_idx + 1}/{len(plan.available_datasets)}: "
                        f"{dataset_config.name}"
                    )
                    try:
                        is_gasstation = "gasstation" in dataset_config.name.lower()
                        should_download = (
                            download_latest_gasstation_data if is_gasstation else True
                        ) if not skip_missing else False

                        aug_iterator = DatasetIterator(
                            dataset_config,
                            max_samples=n_aug_per_dataset,
                            cache_dir=cache_dir,
                            download=should_download,
                            hf_token=hf_token,
                            seed=seed,
                            lazy_read=True,
                        )

                        if skip_missing and aug_iterator.get_total_cached_count() == 0:
                            continue

                        aug_pipeline = VideoPrefetchPipeline(
                            dataset_iterator=aug_iterator,
                            target_size=plan.target_size,
                            batch_size=batch_size,
                            seed=aug_seed,
                            augment_level=augment_level,
                            crop_prob=crop_prob,
                            num_frames=num_frames,
                            frame_rate=frame_rate,
                            robustness_pass=True,
                            aug_cache_dir=aug_cache_dir,
                        )

                        aug_batch_id = 0
                        try:
                            for batch_data in aug_pipeline:
                                aug_batch_id += 1
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
                                    aug_batch_id,
                                    aug_pass=True,
                                )
                        finally:
                            aug_pipeline.close()

                    except Exception as e:
                        logger.error(
                            f"Video robustness pass failed for {dataset_config.name}: {e}"
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
