import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue
import threading
from typing import Dict, Optional

import numpy as np

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
_AUDIO_WORKERS = 4
_HEAVY_AUDIO_KEYS = ("audio", "audio_bytes", "audio_path", "preprocessed_waveform")


class AudioPrefetchPipeline:
    """Decode the next clips while the GPU scores the current batch.

    Workers match the sandbox CPU count. Sample seeds stay seed + sample_index
    from a serial walk, so a resumed run and a fresh run crop the same audio.
    Checkpointed samples are skipped before decode.
    """

    def __init__(
        self,
        dataset_iterator,
        *,
        dataset_name,
        tracker,
        batch_size,
        seed,
        target_sr,
        aug_pass,
        aug_cache_dir,
        aug_cache_readonly,
        num_workers=_AUDIO_WORKERS,
    ):
        self.dataset_iterator = dataset_iterator
        self.dataset_name = dataset_name
        self.tracker = tracker
        self.batch_size = batch_size
        self.seed = seed
        self.target_sr = target_sr
        self.aug_pass = aug_pass
        self.aug_cache_dir = aug_cache_dir
        self.aug_cache_readonly = aug_cache_readonly
        self.num_workers = num_workers
        self.batch_queue = Queue(maxsize=num_workers)
        self.stop_event = threading.Event()
        self.error = None
        self.errors = []
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()

    def _prepare(self, sample, sample_index, dataset_name):
        verify_sample(sample)
        try:
            audio_sample = load_audio_sample(sample)
            sample_seed = None if self.seed is None else (self.seed + sample_index)
            if audio_sample.get("is_preprocessed", False):
                audio_array = audio_sample.get("preprocessed_waveform")
                label = audio_sample.get("label")
                if audio_array is None or label is None:
                    return None
                if hasattr(audio_array, "numpy"):
                    audio_array = audio_array.numpy()
            else:
                audio_array, label = process_audio_sample(
                    audio_sample,
                    target_sr=self.target_sr,
                    seed=sample_seed,
                )
                if audio_array is None or label is None:
                    return None
                if hasattr(audio_array, "numpy"):
                    audio_array = audio_array.numpy()

            if self.aug_pass:
                cache_path = (
                    aud_aug_cache_path(
                        self.aug_cache_dir, build_sample_id(sample), sample_seed
                    )
                    if self.aug_cache_dir else None
                )
                if cache_path and use_augmentation_cache(sample, cache_path):
                    audio_array = np.load(cache_path, allow_pickle=False)
                else:
                    audio_array, _, _, _ = apply_audio_robustness_augmentations(
                        np.asarray(audio_array).squeeze(),
                        target_sr=self.target_sr,
                        seed=sample_seed,
                    )
                    if cache_path and not self.aug_cache_readonly:
                        write_aug_cache(cache_path, audio_array)

            sample_meta = {k: v for k, v in sample.items() if k not in _HEAVY_AUDIO_KEYS}
            return {
                "audio": audio_array,
                "label": label,
                "sample": sample_meta,
                "sample_index": sample_index,
                "dataset_name": dataset_name,
                "sample_seed": sample_seed,
            }
        except CheckpointError:
            raise
        except Exception as exc:
            message = f"Audio processing error: {exc}"
            logger.warning(f"Failed to preprocess audio sample {sample_index}: {exc}")
            self.errors.append(message[:120])
            return None

    def _producer_loop(self):
        try:
            dataset_name = self.dataset_name
            max_in_flight = self.num_workers * 4
            sample_iter = enumerate(self.dataset_iterator, 1)
            pending = set()
            exhausted = False
            batch = []

            while not self.stop_event.is_set():
                while len(pending) < max_in_flight and not exhausted:
                    try:
                        idx, sample = next(sample_iter)
                        if self.tracker is not None and self.tracker.is_checkpointed(
                            dataset_name=dataset_name,
                            sample_index=idx,
                            sample=sample,
                            aug_pass=self.aug_pass,
                        ):
                            continue
                        future = self.executor.submit(self._prepare, sample, idx, dataset_name)
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
                    except CheckpointError:
                        raise
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
        except Exception as exc:
            self.error = exc
            logger.error(f"Error in audio prefetch pipeline: {exc}\n{traceback.format_exc()}")
            self.batch_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.error:
            raise self.error
        try:
            batch = self.batch_queue.get(timeout=300)
        except Empty:
            raise RuntimeError("Audio prefetch queue stalled for 300s") from None
        if batch is None:
            if self.error:
                raise self.error
            raise StopIteration
        return batch

    def close(self):
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

                    pipeline = AudioPrefetchPipeline(
                        dataset_iterator,
                        dataset_name=dataset_cfg.name,
                        tracker=tracker,
                        batch_size=batch_size,
                        seed=seed,
                        target_sr=target_sr,
                        aug_pass=aug_pass,
                        aug_cache_dir=aug_cache_dir,
                        aug_cache_readonly=aug_cache_readonly,
                    )
                    batch_id = 0
                    try:
                        for batch_data in pipeline:
                            batch_id += 1
                            batch_audio = [item["audio"] for item in batch_data]
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
                                batch_audio,
                                batch_metadata,
                                tracker,
                                batch_id,
                                aug_pass=aug_pass,
                            )
                            if tracker.count % 500 == 0:
                                logger.info(f"Progress: {tracker.count} samples")
                    finally:
                        benchmark_results["errors"].extend(pipeline.errors[:20])
                        pipeline.close()

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
