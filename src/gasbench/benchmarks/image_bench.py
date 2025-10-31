import json
import os
import time
import numpy as np
from typing import Dict, Optional

from ..logger import get_logger
from ..processing.media import process_image_sample
from ..processing.transforms import apply_random_augmentations, compress_image_jpeg_pil
from ..dataset.config import (
    get_benchmark_size,
    discover_benchmark_image_datasets,
    calculate_weighted_dataset_sampling,
    build_dataset_info,
)
from ..dataset.iterator import DatasetIterator
from .utils import (
    ConfusionMatrix,
    multiclass_to_binary,
    update_generator_stats,
    calculate_per_source_accuracy,
    should_track_sample,
    create_misclassification_record,
    aggregate_misclassification_stats,
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
    download_latest_gasstation_data: bool = False,
    cache_policy: Optional[str] = None,
    seed: Optional[int] = None,
) -> float:
    """Test model on benchmark image datasets for AI-generated content detection."""

    try:
        hf_token = os.environ.get("HF_TOKEN")

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
        incorrect_samples = []  # Track misclassified gasstation samples

        target_size = get_benchmark_size("image", mode)
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

        generator_stats = benchmark_results.get("image_generator_stats", {})

        dataset_info = build_dataset_info(available_datasets, dataset_sampling)
        for dataset_idx, dataset_config in enumerate(available_datasets):
            dataset_cap = dataset_sampling[dataset_config.name]
            logger.info(
                f"Processing dataset {dataset_idx + 1}/{len(available_datasets)}: "
                f"{dataset_config.name} ({dataset_cap} samples)"
            )

            dataset_correct = 0
            dataset_total = 0

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

                sample_index = 0  # Track sample index for unique IDs
                for sample in dataset_iterator:
                    sample_index += 1
                    try:
                        # Process image sample (load, convert to numpy)
                        image_array, true_label_multiclass = process_image_sample(sample)

                        if image_array is None or true_label_multiclass is None:
                            continue

                        # Apply augmentations
                        try:
                            chw = image_array[0]
                            hwc = np.transpose(chw, (1, 2, 0))
                            aug_hwc, _, _, _ = apply_random_augmentations(hwc)
                            aug_hwc = compress_image_jpeg_pil(aug_hwc, quality=75)
                            aug_chw = np.transpose(aug_hwc, (2, 0, 1))
                            image_array = np.expand_dims(aug_chw, 0)
                        except Exception:
                            # If augmentation fails, continue with unaugmented image
                            pass

                        # Run inference
                        start = time.time()
                        outputs = session.run(None, {input_specs[0].name: image_array})
                        inference_times.append((time.time() - start) * 1000)

                        predicted_binary, predicted_multiclass = process_model_output(outputs[0])

                        true_label_binary = multiclass_to_binary(true_label_multiclass)
                        confusion_matrix.update(
                            true_label_binary, predicted_binary,
                            true_label_multiclass, predicted_multiclass
                        )

                        is_correct = predicted_binary == true_label_binary
                        if is_correct:
                            correct += 1
                            dataset_correct += 1
                        else:
                            # Track misclassified gasstation samples with generator info
                            if should_track_sample(sample, dataset_config.name):
                                misclassification = create_misclassification_record(
                                    sample,
                                    sample_index,
                                    true_label_multiclass,
                                    predicted_multiclass,
                                )
                                incorrect_samples.append(misclassification)

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
                        benchmark_results["errors"].append(f"Image processing error: {str(e)[:100]}")

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

        misclassification_stats = aggregate_misclassification_stats(incorrect_samples)

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
            "misclassified_samples": incorrect_samples,
            "misclassification_stats": misclassification_stats,
        }

        if benchmark_results.get("image_generator_stats"):
            benchmark_results["image_results"]["generator_stats"] = benchmark_results["image_generator_stats"]

        logger.info(f"✅ Benchmark complete: {accuracy:.2%} ({correct}/{total} correct)")
        return accuracy

    except Exception as e:
        logger.error(f"Benchmark image testing failed: {e}")
        benchmark_results["image_results"] = {"error": str(e)}
        raise e
