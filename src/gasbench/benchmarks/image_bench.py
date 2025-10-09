import json
import time
import numpy as np
from typing import Dict

from ..logger import get_logger
from ..processing.media import process_image_sample
from ..processing.transforms import apply_random_augmentations, compress_image_jpeg_pil
from ..dataset.config import (
    get_benchmark_size,
    discover_benchmark_image_datasets,
    calculate_dataset_sampling,
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


async def run_image_benchmark(
    session,
    input_specs,
    benchmark_results: Dict,
    mode: str = "full",
    gasstation_only: bool = False,
    cache_dir: str = "/.cache/gasbench",
) -> float:
    """Test model on benchmark image datasets for AI-generated content detection."""

    try:
        if gasstation_only:
            logger.info("Loading gasstation image datasets only")
        else:
            logger.info("Loading benchmark image datasets")
        
        available_datasets = discover_benchmark_image_datasets(mode, gasstation_only)

        if not available_datasets:
            logger.error("No benchmark image datasets configured")
            benchmark_results["image_results"] = {"error": "No datasets available"}
            return 0.0

        logger.info(f"Using {len(available_datasets)} image datasets for benchmarking")

        correct = 0
        total = 0
        inference_times = []
        per_dataset_results = {}
        confusion_matrix = ConfusionMatrix()

        target_size = get_benchmark_size("image", mode)

        per_dataset_cap, min_samples_per_dataset = calculate_dataset_sampling(
            len(available_datasets), target_size
        )
        
        sampling_info = {
            "target_samples": target_size,
            "num_datasets": len(available_datasets),
            "per_dataset_cap": per_dataset_cap,
            "min_per_dataset": min_samples_per_dataset,
            "dataset_breakdown": {
                "real": len([d for d in available_datasets if d.media_type == 'real']),
                "synthetic": len([d for d in available_datasets if d.media_type == 'synthetic']),
                "semisynthetic": len([d for d in available_datasets if d.media_type == 'semisynthetic'])
            }
        }
        logger.info(f"Sampling configuration: {json.dumps(sampling_info)}")
        
        dataset_info = build_dataset_info(available_datasets, per_dataset_cap)

        generator_stats = benchmark_results.get("image_generator_stats", {})

        for dataset_idx, dataset_config in enumerate(available_datasets):
            logger.info(
                f"Processing dataset {dataset_idx + 1}/{len(available_datasets)}: "
                f"{dataset_config.name} ({per_dataset_cap} samples)"
            )

            dataset_correct = 0
            dataset_total = 0

            try:
                dataset_iterator = DatasetIterator(dataset_config, max_samples=per_dataset_cap, cache_dir=cache_dir)

                expected_label = 0 if dataset_config.media_type == 'real' else 1
                logger.info(
                    f"Processing {dataset_config.name} "
                    f"(media_type={dataset_config.media_type}, expected_label={expected_label})"
                )

                for sample in dataset_iterator:
                    try:
                        if dataset_total == 0:
                            first_sample_info = {
                                "dataset": dataset_config.name,
                                "keys": list(sample.keys()),
                                "image_type": str(type(sample.get('image')))
                            }

                        image_array, true_label_multiclass = process_image_sample(sample)

                        if image_array is not None:
                            try:
                                chw = image_array[0]
                                hwc = np.transpose(chw, (1, 2, 0))
                                aug_hwc, _, _, _ = apply_random_augmentations(hwc)
                                aug_hwc = compress_image_jpeg_pil(aug_hwc, quality=75)
                                aug_chw = np.transpose(aug_hwc, (2, 0, 1))
                                image_array = np.expand_dims(aug_chw, 0)
                            except Exception as aug_e:
                                logger.debug(f"Image augmentation skipped: {aug_e}")

                        if image_array is None or true_label_multiclass is None:
                            continue

                        true_label_binary = multiclass_to_binary(true_label_multiclass)

                        start = time.time()
                        outputs = session.run(None, {input_specs[0].name: image_array})
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
                            logger.debug(
                                f"Sample {total}: "
                                f"True={true_label_multiclass}→{true_label_binary}, "
                                f"Pred={predicted_multiclass}→{predicted_binary}, "
                                f"Correct={is_correct}, "
                                f"Generator={sample.get('model_name', 'unknown')}, "
                                f"Dataset={dataset_config.name}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to process image sample from {dataset_config.name}: {e}")
                        benchmark_results["errors"].append(f"Image inference error: {str(e)[:100]}")

                dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0.0
                per_dataset_results[dataset_config.name] = {
                    "correct": dataset_correct,
                    "total": dataset_total,
                    "accuracy": dataset_accuracy,
                }

                if generator_stats:
                    benchmark_results["image_generator_stats"] = generator_stats

                logger.info(
                    f"Dataset {dataset_config.name}: {dataset_accuracy:.2%} accuracy "
                    f"({dataset_correct}/{dataset_total})"
                )

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                benchmark_results["errors"].append(f"Dataset error for {dataset_config.name}: {str(e)[:100]}")

        accuracy = correct / total if total > 0 else 0.0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        p95_inference_time = float(np.percentile(inference_times, 95)) if inference_times else 0.0

        binary_mcc = confusion_matrix.calculate_binary_mcc() if total > 0 else 0.0
        multiclass_mcc = confusion_matrix.calculate_multiclass_mcc()

        per_source_accuracy = calculate_per_source_accuracy(available_datasets, per_dataset_results)

        benchmark_results["image_results"] = {
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
        }

        if benchmark_results.get("image_generator_stats"):
            benchmark_results["image_results"]["generator_stats"] = benchmark_results["image_generator_stats"]

        logger.info(f"✅ Benchmark complete: {accuracy:.2%} ({correct}/{total} correct)")
        return accuracy

    except Exception as e:
        logger.error(f"Benchmark image testing failed: {e}")
        benchmark_results["image_results"] = {"error": str(e)}
        raise e
