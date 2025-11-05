import json
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
    compress_video_frames_jpeg_torchvision,
    extract_target_size_from_input_specs,
)
from ..config import DEFAULT_TARGET_SIZE, DEFAULT_VIDEO_BATCH_SIZE
from ..dataset.config import (
    get_benchmark_size,
    discover_benchmark_video_datasets,
    calculate_weighted_dataset_sampling,
    build_dataset_info,
)
from ..dataset.iterator import DatasetIterator
from .utils import (
    Metrics,
    multiclass_to_binary,
    update_generator_stats,
    calculate_per_source_accuracy,
    should_track_sample,
    create_misclassification_record,
    aggregate_misclassification_stats,
)
from ..model.inference import process_model_output

logger = get_logger(__name__)


def process_video_batch(
    session,
    input_specs,
    batch_videos,
    batch_metadata,
    metrics,
    generator_stats,
    incorrect_samples,
):
    """push a batch of videos through the model."""
    if not batch_videos:
        return 0, 0, []

    batch_array = np.stack(batch_videos, axis=0)

    start = time.time()
    outputs = session.run(None, {input_specs[0].name: batch_array})
    batch_inference_time = (time.time() - start) * 1000
    per_sample_time = batch_inference_time / len(batch_videos)
    inference_times = [per_sample_time] * len(batch_videos)

    correct = 0
    for i, (true_label_binary, true_label_multiclass, sample, sample_index, dataset_name) in enumerate(batch_metadata):
        predicted_binary, predicted_multiclass, pred_probs = process_model_output(outputs[0][i])

        metrics.update(
            true_label_binary, predicted_binary,
            true_label_multiclass, predicted_multiclass,
            pred_probs
        )

        is_correct = predicted_binary == true_label_binary
        if is_correct:
            correct += 1
        else:
            if should_track_sample(sample, dataset_name):
                misclassification = create_misclassification_record(
                    sample,
                    sample_index,
                    true_label_multiclass,
                    predicted_multiclass,
                )
                incorrect_samples.append(misclassification)

        update_generator_stats(generator_stats, sample, true_label_binary, predicted_binary)

    return correct, len(batch_videos), inference_times


async def video_prefetcher(
    dataset_iterator,
    dataset_config,
    queue: asyncio.Queue,
    max_queue_size: int = 2,
    seed: int = None,
    target_size=None,
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
                video_array, true_label_multiclass = process_video_bytes_sample(sample)

                if video_array is None or true_label_multiclass is None:
                    # Put a skip marker in the queue
                    await queue.put(("skip", None, None, None, None))
                    continue

                try:
                    tchw = video_array[0]
                    thwc = np.transpose(tchw, (0, 2, 3, 1))
                    sample_seed = None if seed is None else (seed + sample_index)
                    aug_thwc, _, _, _ = apply_random_augmentations(
                        thwc, target_size, seed=sample_seed
                    )
                    aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))

                    video_array = np.expand_dims(aug_tchw, 0)
                except Exception as e:
                    logger.error(f"Video augmentation failed: {e}")
                    await queue.put(("skip", None, None, None, None))
                    continue

                true_label_binary = multiclass_to_binary(true_label_multiclass)
                await queue.put(("data", video_array, true_label_binary, true_label_multiclass, sample))
                
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
) -> float:
    """Test model on benchmark video datasets for AI-generated content detection."""
    
    if batch_size is None:
        batch_size = DEFAULT_VIDEO_BATCH_SIZE

    try:
        hf_token = os.environ.get("HF_TOKEN")

        if gasstation_only:
            logger.info("Loading gasstation video datasets only")
        else:
            logger.info("Loading benchmark video datasets")
        
        available_datasets = discover_benchmark_video_datasets(mode, gasstation_only)

        if not available_datasets:
            logger.error("No benchmark video datasets configured")
            benchmark_results["video_results"] = {"error": "No datasets available"}
            return 0.0

        logger.info(f"Using {len(available_datasets)} video datasets for benchmarking")

        target_size = extract_target_size_from_input_specs(input_specs)
        if target_size is None:
            target_size = DEFAULT_TARGET_SIZE
            logger.info(f"Model has dynamic axes, using default target size: {target_size[0]}x{target_size[1]}")
        else:
            logger.info(f"Using fixed target size from model: {target_size[0]}x{target_size[1]}")

        with video_archive_manager(cache_dir=cache_dir) as video_cache:
            correct = 0
            total = 0
            inference_times = []
            per_dataset_results = {}
            metrics = Metrics()
            skipped_samples = 0
            incorrect_samples = []  # Track misclassified gasstation samples

            target_samples = get_benchmark_size("video", mode)
            dataset_sampling = calculate_weighted_dataset_sampling(available_datasets, target_samples)
            
            actual_total_samples = sum(dataset_sampling.values())

            # Calculate summary stats for logging
            gasstation_count = len([d for d in available_datasets if "gasstation" in d.name.lower()])
            regular_count = len(available_datasets) - gasstation_count
            gasstation_cap = dataset_sampling.get(
                next((d.name for d in available_datasets if "gasstation" in d.name.lower()), ""), 0
            )
            regular_cap = dataset_sampling.get(
                next((d.name for d in available_datasets if "gasstation" not in d.name.lower()), ""), 0
            )

            sampling_info = {
                "batch_size": batch_size,
                "target_samples": target_samples,
                "actual_total_samples": actual_total_samples,
                "num_datasets": len(available_datasets),
                "gasstation_datasets": gasstation_count,
                "regular_datasets": regular_count,
                "gasstation_samples_per_dataset": gasstation_cap,
                "regular_samples_per_dataset": regular_cap,
                "dataset_breakdown": {
                    "real": len([d for d in available_datasets if d.media_type == 'real']),
                    "synthetic": len([d for d in available_datasets if d.media_type == 'synthetic']),
                    "semisynthetic": len([d for d in available_datasets if d.media_type == 'semisynthetic'])
                }
            }
            logger.info(f"Sampling configuration: {json.dumps(sampling_info)}")
            
            dataset_info = build_dataset_info(available_datasets, dataset_sampling)

            generator_stats = benchmark_results.get("video_generator_stats", {})

            for dataset_idx, dataset_config in enumerate(available_datasets):
                dataset_cap = dataset_sampling[dataset_config.name]
                logger.info(
                    f"Processing dataset {dataset_idx + 1}/{len(available_datasets)}: "
                    f"{dataset_config.name} ({dataset_cap} samples)"
                )

                dataset_correct = 0
                dataset_total = 0
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

                    # Create prefetch queue with larger size to allow more parallelism
                    prefetch_queue = asyncio.Queue(maxsize=16)

                    prefetch_task = asyncio.create_task(
                        video_prefetcher(
                            dataset_iterator,
                            dataset_config,
                            prefetch_queue,
                            seed=seed,
                            target_size=target_size,
                        )
                    )

                    sample_index = 0
                    batch_videos = []
                    batch_metadata = []

                    while True:
                        item = await prefetch_queue.get()
                        item_type = item[0]

                        if item_type == "done":
                            break
                        elif item_type == "skip":
                            dataset_skipped += 1
                            skipped_samples += 1
                            if dataset_skipped % 10 == 0:
                                logger.debug(
                                    f"Dataset {dataset_config.name}: Skipped {dataset_skipped} samples so far"
                                )
                            continue
                        elif item_type == "error":
                            benchmark_results["errors"].append(f"Video processing error during prefetch")
                            dataset_skipped += 1
                            skipped_samples += 1
                            continue

                        video_array, true_label_binary, true_label_multiclass, sample = item[1], item[2], item[3], item[4]
                        sample_index += 1
                        
                        batch_videos.append(video_array[0])
                        batch_metadata.append((true_label_binary, true_label_multiclass, sample, sample_index, dataset_config.name))

                        if len(batch_videos) >= batch_size:
                            batch_correct, batch_total, batch_times = process_video_batch(
                                session, input_specs, batch_videos, batch_metadata,
                                metrics, generator_stats, incorrect_samples
                            )
                            correct += batch_correct
                            dataset_correct += batch_correct
                            total += batch_total
                            dataset_total += batch_total
                            inference_times.extend(batch_times)
                            
                            batch_videos = []
                            batch_metadata = []

                            if total % 500 == 0:
                                logger.info(
                                    f"Progress: {total}/{actual_total_samples} samples, "
                                    f"Accuracy: {correct / total:.2%}"
                                )
                    
                    # Wait for prefetcher to finish
                    await prefetch_task

                    if batch_videos:
                        batch_correct, batch_total, batch_times = process_video_batch(
                            session, input_specs, batch_videos, batch_metadata,
                            metrics, generator_stats, incorrect_samples
                        )
                        correct += batch_correct
                        dataset_correct += batch_correct
                        total += batch_total
                        dataset_total += batch_total
                        inference_times.extend(batch_times)

                    dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0.0
                    per_dataset_results[dataset_config.name] = {
                        "correct": dataset_correct,
                        "total": dataset_total,
                        "accuracy": dataset_accuracy,
                        "skipped": dataset_skipped,
                    }

                    if generator_stats:
                        benchmark_results["video_generator_stats"] = generator_stats

                    logger.info(
                        f"Dataset {dataset_config.name}: {dataset_accuracy:.2%} accuracy "
                        f"({dataset_correct}/{dataset_total}), skipped: {dataset_skipped}"
                    )

                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                    benchmark_results["errors"].append(f"Dataset error for {dataset_config.name}: {str(e)[:100]}")

            accuracy = correct / total if total > 0 else 0.0
            if skipped_samples > 0:
                logger.info(f"Video benchmark: {total} samples processed, {skipped_samples} skipped")

            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
            p95_inference_time = float(np.percentile(inference_times, 95)) if inference_times else 0.0

            binary_mcc = metrics.calculate_binary_mcc() if total > 0 else 0.0
            multiclass_mcc = metrics.calculate_multiclass_mcc()
            binary_ce = metrics.calculate_binary_cross_entropy()
            multiclass_ce = metrics.calculate_multiclass_cross_entropy()
            sn34_score = metrics.compute_sn34_score()

            per_source_accuracy = calculate_per_source_accuracy(available_datasets, per_dataset_results)

            cache_info = video_cache.get_cache_info()

            misclassification_stats = aggregate_misclassification_stats(incorrect_samples)

            benchmark_results["video_results"] = {
                "benchmark_score": accuracy,
                "sn34_score": sn34_score,
                "total_samples": total,
                "correct_predictions": correct,
                "avg_inference_time_ms": avg_inference_time,
                "p95_inference_time_ms": p95_inference_time,
                "binary_mcc": binary_mcc,
                "multiclass_mcc": multiclass_mcc,
                "binary_cross_entropy": binary_ce,
                "multiclass_cross_entropy": multiclass_ce,
                "per_source_accuracy": per_source_accuracy,
                "per_dataset_results": per_dataset_results,
                "dataset_info": dataset_info,
                "cache_info": cache_info,
                "misclassified_samples": incorrect_samples,
                "misclassification_stats": misclassification_stats,
            }

            if benchmark_results.get("video_generator_stats"):
                benchmark_results["video_results"]["generator_stats"] = benchmark_results["video_generator_stats"]

            logger.info(f"✅ Benchmark complete: {accuracy:.2%} ({correct}/{total} correct)")
            logger.info(f"Archive cache: {cache_info['unpacked_archives']} archives cached locally")
            if skipped_samples > 0:
                logger.warning(f"Skipped {skipped_samples} samples due to missing/inaccessible archives or processing errors")
            
            return accuracy

    except Exception as e:
        logger.error(f"Benchmark video testing failed: {e}")
        benchmark_results["video_results"] = {"error": str(e)}
        raise e
