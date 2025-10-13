import json
import time
import numpy as np
from typing import Dict

from ..logger import get_logger
from ..processing.archive import video_archive_manager
from ..processing.media import process_video_bytes_sample
from ..processing.transforms import apply_random_augmentations, compress_video_frames_jpeg_torchvision
from ..dataset.config import (
    get_benchmark_size,
    discover_benchmark_video_datasets,
    calculate_weighted_dataset_sampling,
    build_dataset_info,
)
from ..dataset.iterator import DatasetIterator
from .metrics import (
    ConfusionMatrix,
    multiclass_to_binary,
    update_generator_stats,
    calculate_per_source_accuracy,
)
from ..model.inference import process_model_output

logger = get_logger(__name__)


async def run_video_benchmark(
    session,
    input_specs,
    benchmark_results: Dict,
    mode: str = "full",
    gasstation_only: bool = False,
    cache_dir: str = "/.cache/gasbench",
) -> float:
    """Test model on benchmark video datasets for AI-generated content detection."""

    try:
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

        with video_archive_manager(cache_dir=cache_dir) as video_cache:
            correct = 0
            total = 0
            inference_times = []
            per_dataset_results = {}
            confusion_matrix = ConfusionMatrix()
            skipped_samples = 0

            target_size = get_benchmark_size("video", mode)
            dataset_sampling = calculate_weighted_dataset_sampling(available_datasets, target_size)
            
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
                "target_samples": target_size,
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
                    dataset_iterator = DatasetIterator(dataset_config, max_samples=dataset_cap, cache_dir=cache_dir)
                    for sample in dataset_iterator:
                        try:
                            # Decode video bytes and extract frames
                            video_array, true_label_multiclass = process_video_bytes_sample(sample)

                            if video_array is None or true_label_multiclass is None:
                                dataset_skipped += 1
                                skipped_samples += 1
                                if dataset_skipped % 10 == 0:
                                    logger.debug(
                                        f"Dataset {dataset_config.name}: Skipped {dataset_skipped} samples so far"
                                    )
                                continue

                            # Apply augmentations
                            try:
                                tchw = video_array[0]
                                thwc = np.transpose(tchw, (0, 2, 3, 1))
                                aug_thwc, _, _, _ = apply_random_augmentations(thwc)
                                aug_thwc = compress_video_frames_jpeg_torchvision(aug_thwc, quality=75)
                                aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))
                                video_array = np.expand_dims(aug_tchw, 0)
                            except Exception:
                                # If augmentation fails, continue with unaugmented video
                                pass

                            true_label_binary = multiclass_to_binary(true_label_multiclass)

                            start = time.time()
                            outputs = session.run(None, {input_specs[0].name: video_array})
                            inference_times.append((time.time() - start) * 1000)

                            predicted_binary, predicted_multiclass = process_model_output(outputs[0])

                            confusion_matrix.update(
                                true_label_binary, predicted_binary,
                                true_label_multiclass, predicted_multiclass
                            )

                            is_correct = predicted_binary == true_label_binary
                            if is_correct:
                                correct += 1
                                dataset_correct += 1

                            total += 1
                            dataset_total += 1

                            update_generator_stats(generator_stats, sample, true_label_binary, predicted_binary)

                            if total % 500 == 0:
                                logger.info(
                                    f"Progress: {total}/{target_size} samples, "
                                    f"Accuracy: {correct / total:.2%}"
                                )

                            if total % 100 == 0:
                                frames_count = video_array.shape[1] if video_array is not None else 0
                                logger.debug(
                                    f"Sample {total}: "
                                    f"True={true_label_multiclass}→{true_label_binary}, "
                                    f"Pred={predicted_multiclass}→{predicted_binary}, "
                                    f"Correct={is_correct}, "
                                    f"Generator={sample.get('model_name', 'unknown')}, "
                                    f"Dataset={dataset_config.name}, "
                                    f"Frames={frames_count}"
                                )

                        except Exception as e:
                            logger.warning(f"Failed to process video sample from {dataset_config.name}: {e}")
                            benchmark_results["errors"].append(f"Video processing error: {str(e)[:100]}")
                            dataset_skipped += 1
                            skipped_samples += 1

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

            binary_mcc = confusion_matrix.calculate_binary_mcc() if total > 0 else 0.0
            multiclass_mcc = confusion_matrix.calculate_multiclass_mcc()

            per_source_accuracy = calculate_per_source_accuracy(available_datasets, per_dataset_results)

            cache_info = video_cache.get_cache_info()

            benchmark_results["video_results"] = {
                "benchmark_score": accuracy,
                "total_samples": total,
                "correct_predictions": correct,
                "avg_inference_time_ms": avg_inference_time,
                "p95_inference_time_ms": p95_inference_time,
                "binary_mcc": binary_mcc,
                "multiclass_mcc": multiclass_mcc,
                "per_source_accuracy": per_source_accuracy,
                "per_dataset_results": per_dataset_results,
                "dataset_info": dataset_info,
                "cache_info": cache_info,
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
