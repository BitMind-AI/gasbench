import json
import os
import time
import traceback
import numpy as np
from typing import Dict, Optional

from ..logger import get_logger
from ..processing.media import process_audio_sample
from ..config import DEFAULT_TARGET_SIZE  # Not really used for audio but keeping structure
from ..dataset.config import (
    get_benchmark_size,
    discover_benchmark_audio_datasets,
    calculate_weighted_dataset_sampling,
    build_dataset_info,
    load_holdout_datasets_from_yaml,
    apply_mode_to_datasets,
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

DEFAULT_AUDIO_BATCH_SIZE = 16  # Smaller batch size for audio tensors? Or larger? 16 seems safe.


def process_batch(
    session,
    input_specs,
    batch_audio,
    batch_metadata,
    metrics,
    generator_stats,
    incorrect_samples,
    per_dataset_pred_counts,
):
    """push a batch of audio samples through the model."""
    if not batch_audio:
        return 0, 0, []

    # batch_audio is a list of tensors. We need to stack them.
    # Assuming all audio tensors have same length after processing.
    # If variable length, model needs to handle it or we need padding.
    # process_audio_sample does resampling but not fixed length padding/trimming by default in this plan.
    # However, ONNX models usually expect fixed input size or dynamic axes.
    # For simplicity, we assume the model or preprocessing handles size, or we stack if shapes match.
    
    try:
        # Convert list of tensors/arrays to single batch array
        # Check if they are tensors or numpy arrays
        first = batch_audio[0]
        if hasattr(first, "numpy"):
            first = first.numpy()
            batch_audio_np = [b.numpy() for b in batch_audio]
        else:
            batch_audio_np = batch_audio

        batch_array = np.stack(batch_audio_np)
    except Exception as e:
        logger.error(f"Failed to stack audio batch: {e}")
        return 0, 0, []

    start = time.time()
    outputs = session.run(None, {input_specs[0].name: batch_array})
    batch_inference_time = (time.time() - start) * 1000
    per_sample_time = batch_inference_time / len(batch_audio)
    inference_times = [per_sample_time] * len(batch_audio)

    correct = 0
    for i, (true_label_multiclass, sample, sample_index, dataset_name) in enumerate(batch_metadata):
        predicted_binary, predicted_multiclass, pred_probs = process_model_output(outputs[0][i])

        true_label_binary = multiclass_to_binary(true_label_multiclass)
        metrics.update(
            true_label_binary, predicted_binary,
            true_label_multiclass, predicted_multiclass,
            pred_probs
        )

        # Track per-dataset predicted label counts
        ds_counts = per_dataset_pred_counts.get(dataset_name)
        if ds_counts is None:
            ds_counts = {0: 0, 1: 0, 2: 0}
            per_dataset_pred_counts[dataset_name] = ds_counts
        if predicted_multiclass in ds_counts:
            ds_counts[predicted_multiclass] += 1
        else:
            ds_counts[predicted_multiclass] = ds_counts.get(predicted_multiclass, 0) + 1

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
    
    return correct, len(batch_audio), inference_times


async def run_audio_benchmark(
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
) -> float:
    """Test model on benchmark audio datasets for AI-generated content detection."""
    
    if batch_size is None:
        batch_size = DEFAULT_AUDIO_BATCH_SIZE

    try:
        hf_token = os.environ.get("HF_TOKEN")

        if gasstation_only:
            logger.info("Loading gasstation audio datasets only")
        else:
            logger.info("Loading benchmark audio datasets")

        available_datasets = discover_benchmark_audio_datasets(mode, gasstation_only, yaml_path=dataset_config)

        if holdout_config and not gasstation_only:
            try:
                holdouts = load_holdout_datasets_from_yaml(holdout_config).get("audio", [])
                holdouts = apply_mode_to_datasets(holdouts, mode)
                available_datasets.extend(holdouts)
            except Exception as e:
                logger.error(f"Failed to load holdout audio datasets: {e}")

        if not available_datasets:
            logger.error("No benchmark audio datasets configured")
            benchmark_results["audio_results"] = {"error": "No datasets available"}
            return 0.0

        logger.info(f"Using {len(available_datasets)} audio datasets for benchmarking")

        # Determine target sample rate or size from input specs if possible
        # For now we default to 16000Hz in process_audio_sample, but we might need to infer from model
        # input_specs[0].shape might give hints if it's fixed size.
        target_sr = 16000 # Default
        
        correct = 0
        total = 0
        inference_times = []
        per_dataset_results = {}
        per_dataset_pred_counts = {}
        metrics = Metrics()
        incorrect_samples = []

        target_samples = get_benchmark_size("audio", mode)
        dataset_sampling = calculate_weighted_dataset_sampling(available_datasets, target_samples)

        actual_total_samples = sum(dataset_sampling.values())

        # Summary stats
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

        generator_stats = benchmark_results.get("audio_generator_stats", {})

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

                batch_audio = []
                batch_metadata = []

                sample_index = 0
                for sample in dataset_iterator:
                    sample_index += 1
                    try:
                        audio_array, true_label_multiclass = process_audio_sample(sample, target_sr=target_sr)

                        if audio_array is None or true_label_multiclass is None:
                            continue

                        # Ensure audio is in correct format (e.g. 1D or 2D numpy array)
                        # torchaudio returns tensor (Channels, Time)
                        # For inference we usually want numpy array
                        if hasattr(audio_array, "numpy"):
                            audio_array = audio_array.numpy()

                        batch_audio.append(audio_array)
                        batch_metadata.append((true_label_multiclass, sample, sample_index, dataset_config.name))

                        if len(batch_audio) >= batch_size:
                            batch_correct, batch_total, batch_times = process_batch(
                                session, input_specs, batch_audio, batch_metadata,
                                metrics, generator_stats, incorrect_samples, per_dataset_pred_counts
                            )
                            correct += batch_correct
                            dataset_correct += batch_correct
                            total += batch_total
                            dataset_total += batch_total
                            inference_times.extend(batch_times)
                            
                            batch_audio = []
                            batch_metadata = []

                            if total % 500 == 0:
                                logger.info(
                                    f"Progress: {total}/{actual_total_samples} samples, "
                                    f"Accuracy: {correct / total:.2%}"
                                )

                    except Exception as e:
                        logger.warning(f"Failed to process audio sample from {dataset_config.name}: {e}\n{traceback.format_exc()}")
                        benchmark_results["errors"].append(f"Audio processing error: {str(e)[:100]}")

                if batch_audio:
                    batch_correct, batch_total, batch_times = process_batch(
                        session, input_specs, batch_audio, batch_metadata,
                        metrics, generator_stats, incorrect_samples, per_dataset_pred_counts
                    )
                    correct += batch_correct
                    dataset_correct += batch_correct
                    total += batch_total
                    dataset_total += batch_total
                    inference_times.extend(batch_times)

                dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0.0
                pred_counts_raw = per_dataset_pred_counts.get(dataset_config.name, {})
                predictions = {
                    "real": int(pred_counts_raw.get(0, 0)),
                    "synthetic": int(pred_counts_raw.get(1, 0)),
                    "semisynthetic": int(pred_counts_raw.get(2, 0)),
                }
                per_dataset_results[dataset_config.name] = {
                    "correct": dataset_correct,
                    "total": dataset_total,
                    "accuracy": dataset_accuracy,
                    "predictions": predictions,
                }

                if generator_stats:
                    benchmark_results["audio_generator_stats"] = generator_stats

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

        binary_mcc = metrics.calculate_binary_mcc() if total > 0 else 0.0
        multiclass_mcc = metrics.calculate_multiclass_mcc()
        binary_ce = metrics.calculate_binary_cross_entropy()
        multiclass_ce = metrics.calculate_multiclass_cross_entropy()
        sn34_score = metrics.compute_sn34_score()

        per_source_accuracy = calculate_per_source_accuracy(available_datasets, per_dataset_results)

        misclassification_stats = aggregate_misclassification_stats(incorrect_samples)

        benchmark_results["audio_results"] = {
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
            "misclassified_samples": incorrect_samples,
            "misclassification_stats": misclassification_stats,
        }

        if benchmark_results.get("audio_generator_stats"):
            benchmark_results["audio_results"]["generator_stats"] = benchmark_results["audio_generator_stats"]

        logger.info(f"✅ Benchmark complete: {accuracy:.2%} ({correct}/{total} correct)")
        return accuracy

    except Exception as e:
        logger.error(f"Benchmark audio testing failed: {e}")
        benchmark_results["audio_results"] = {"error": str(e)}
        raise e

