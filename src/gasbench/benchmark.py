#!/usr/bin/env python3
import os
import json
import time
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .logger import get_logger
from .processing.media import configure_huggingface_cache
from .benchmarks.image_bench import run_image_benchmark
from .benchmarks.video_bench import run_video_benchmark
from .benchmarks.utils import create_inference_session

logger = get_logger(__name__)


async def run_benchmark(
    model_path: str,
    modality: str,
    mode: str = "full",
    gasstation_only: bool = False,
    cache_dir: Optional[str] = None,
    download_latest_gasstation_data: bool = False,
    cache_policy: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataset_config: Optional[str] = None,
    holdout_config: Optional[str] = None,
    records_parquet_path: Optional[str] = None,
) -> Dict:
    """
    Args:
        model_path: Path to ONNX model file
        modality: Type of modality to test ("image" or "video")
        mode: Benchmark mode - "debug", "small", or "full" (default: "full")
        gasstation_only: If True, only use gasstation datasets
        cache_dir: Directory for caching (defaults to /.cache/gasbench)
        download_latest_gasstation_data: If True, download latest gasstation data before benchmarking (default: False)
        cache_policy: Optional path to cache policy JSON file with generator priorities
        seed: Optional random seed for non-gasstation dataset sampling (for reproducible random sampling)
        batch_size: Batch size for model inference (default: 8)
        dataset_config: Optional path to custom dataset YAML config file (default: uses bundled config)

    Returns:
        Dict with benchmark results including scores and metrics
    """
    if not cache_dir:
        cache_dir = "/.cache/gasbench"

    configure_huggingface_cache(cache_dir)
    run_id = str(uuid.uuid4())

    benchmark_results = {
        "model_path": model_path,
        "modality": modality,
        "timestamp": time.time(),
        "benchmark_completed": False,
        "validation": {},
        "benchmark_score": 0.0,
        "errors": [],
        "metrics": {},
        "run_id": run_id,
        "mode": mode,
        "gasstation_only": gasstation_only,
    }

    start_time = time.time()

    try:
        logger.info(
            f"BENCHMARK START: {modality.upper()} model - {model_path}"
        )

        session, input_specs = await load_model_for_benchmark(
            model_path, modality, benchmark_results
        )

        if not session:
            return benchmark_results

        # Run benchmark for the specified modality
        benchmark_score = await execute_benchmark(
            session,
            input_specs,
            modality,
            benchmark_results,
            mode,
            gasstation_only,
            cache_dir,
            download_latest_gasstation_data,
            cache_policy,
            seed,
            batch_size,
            dataset_config,
            holdout_config,
            records_parquet_path,
        )

        benchmark_results["benchmark_score"] = benchmark_score

        benchmark_results["metrics"]["modality"] = modality
        benchmark_results["metrics"]["model_path"] = model_path
        benchmark_results["metrics"]["download_latest_gasstation_data"] = download_latest_gasstation_data
        benchmark_results["benchmark_completed"] = True

    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        benchmark_results["errors"].append(f"Benchmark error: {str(e)}")
        benchmark_results["benchmark_completed"] = False
        if modality == "image" and "image_results" not in benchmark_results:
            benchmark_results["image_results"] = {"error": str(e)}
        elif modality == "video" and "video_results" not in benchmark_results:
            benchmark_results["video_results"] = {"error": str(e)}

        raise e

    finally:
        benchmark_results["metrics"]["benchmark_duration_seconds"] = (
            time.time() - start_time
        )

    return benchmark_results


async def load_model_for_benchmark(
    model_path: str, modality: str, benchmark_results: Dict
):
    """Load and validate ONNX model for benchmarking."""

    if not os.path.exists(model_path):
        benchmark_results["errors"].append(f"Model file not found: {model_path}")
        return None, None

    try:
        # Load the model for this benchmark run
        session = create_inference_session(model_path, modality)

        input_specs = session.get_inputs()
        output_specs = session.get_outputs()

        benchmark_results["validation"]["model_path"] = model_path
        benchmark_results["validation"]["input_shape"] = str(input_specs[0].shape)
        benchmark_results["validation"]["input_type"] = str(input_specs[0].type)
        benchmark_results["validation"]["output_shape"] = str(output_specs[0].shape)

        logger.info(f"Model loaded successfully")
        
        # Get execution providers
        providers = session.get_providers()
        using_gpu = "CUDAExecutionProvider" in providers
        
        model_info = {
            "path": model_path,
            "input_shape": str(input_specs[0].shape),
            "output_shape": str(output_specs[0].shape),
            "execution_provider": providers[0] if providers else "Unknown",
            "gpu_enabled": using_gpu
        }
        logger.info(f"Model info:\n{json.dumps(model_info, indent=2)}")

        return session, input_specs

    except Exception as e:
        logger.error(f"Failed to load model for inference: {e}")
        benchmark_results["errors"].append(f"ONNX runtime error: {str(e)}")
        benchmark_results["benchmark_completed"] = False
        return None, None


async def execute_benchmark(
    session,
    input_specs,
    modality: str,
    benchmark_results: Dict,
    mode: str,
    gasstation_only: bool = False,
    cache_dir: str = "/.cache/gasbench",
    download_latest_gasstation_data: bool = False,
    cache_policy: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataset_config: Optional[str] = None,
    holdout_config: Optional[str] = None,
    records_parquet_path: Optional[str] = None,
) -> float:
    """Execute the actual benchmark evaluation."""

    logger.info(f"Running {modality} benchmark (mode={mode}, gasstation_only={gasstation_only}, download_latest_gasstation_data={download_latest_gasstation_data})")
    if modality == "image":
        df = await run_image_benchmark(
            session,
            input_specs,
            benchmark_results,
            mode,
            gasstation_only,
            cache_dir,
            download_latest_gasstation_data,
            cache_policy,
            seed,
            batch_size,
            dataset_config,
            holdout_config,
            records_parquet_path=records_parquet_path,
        )
        benchmark_score = benchmark_results.get("image_results", {}).get("benchmark_score", 0.0)
    elif modality == "video":
        df = await run_video_benchmark(
            session,
            input_specs,
            benchmark_results,
            mode,
            gasstation_only,
            cache_dir,
            download_latest_gasstation_data,
            cache_policy,
            seed,
            batch_size,
            dataset_config,
            holdout_config,
            records_parquet_path=records_parquet_path,
        )
        benchmark_score = benchmark_results.get("video_results", {}).get("benchmark_score", 0.0)
    else:
        raise ValueError(f"Invalid modality: {modality}. Must be 'image' or 'video'")

    return benchmark_score


def print_benchmark_summary(benchmark_results: Dict):
    """Print a comprehensive summary of benchmark results."""

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print(f"Model Path: {benchmark_results.get('model_path', 'N/A')}")
    print(f"Modality: {benchmark_results.get('modality', 'N/A').upper()}")
    print(f"Run ID: {benchmark_results.get('run_id', 'N/A')}")
    print(f"Mode: {benchmark_results.get('mode', 'full').upper()}")
    print(f"Gasstation Only: {benchmark_results.get('gasstation_only', False)}")
    print(f"Completed: {benchmark_results.get('benchmark_completed', False)}")
    duration = benchmark_results.get("metrics", {}).get("benchmark_duration_seconds", 0)
    print(f"Duration: {duration:.2f} seconds")

    score = benchmark_results.get("benchmark_score", 0.0)
    print(f"\n🎯 OVERALL SCORE: {score:.2%}")

    # Modality-specific results
    modality = benchmark_results.get("modality")
    if modality:
        results_key = f"{modality}_results"
        if results_key in benchmark_results:
            results = benchmark_results[results_key]

            print(f"\n📊 {modality.upper()} RESULTS:")
            print(f"  Total Samples: {results.get('total_samples', 0)}")
            print(f"  Correct Predictions: {results.get('correct_predictions', 0)}")
            print(f"  Accuracy: {results.get('benchmark_score', 0.0):.2%}")

            avg_time = results.get("avg_inference_time_ms", 0)
            p95_time = results.get("p95_inference_time_ms", 0)
            if avg_time > 0:
                print(f"  Avg Inference Time: {avg_time:.1f}ms")
            if p95_time > 0:
                print(f"  P95 Inference Time: {p95_time:.1f}ms")

            sn34_score = results.get("sn34_score", 0.0)
            if sn34_score != 0.0:
                print(f"  SN34 Score: {sn34_score:.4f}")

            binary_mcc = results.get("binary_mcc", 0.0)
            multiclass_mcc = results.get("multiclass_mcc", 0.0)
            if binary_mcc != 0.0:
                print(f"  Binary MCC: {binary_mcc:.4f}")
            if multiclass_mcc != 0.0:
                print(f"  Multiclass MCC: {multiclass_mcc:.4f}")

            binary_ce = results.get("binary_cross_entropy", 0.0)
            multiclass_ce = results.get("multiclass_cross_entropy", 0.0)
            if binary_ce != 0.0:
                print(f"  Binary Cross-Entropy: {binary_ce:.4f}")
            if multiclass_ce != 0.0:
                print(f"  Multiclass Cross-Entropy: {multiclass_ce:.4f}")

            per_dataset = results.get("per_dataset_results", {})
            if per_dataset:
                print(f"\n📋 PER-DATASET RESULTS:")
                for dataset_name, dataset_results in per_dataset.items():
                    accuracy = dataset_results.get("accuracy", 0.0)
                    total = dataset_results.get("total", 0)
                    correct = dataset_results.get("correct", 0)
                    print(f"  {dataset_name}: {accuracy:.2%} ({correct}/{total})")

            misclass_stats = results.get("misclassification_stats", {})
            if misclass_stats and misclass_stats.get("total_misclassified", 0) > 0:
                print(f"\n❌ MISCLASSIFIED GASSTATION SAMPLES:")
                print(f"  Total Misclassified: {misclass_stats['total_misclassified']}")

                by_generator = misclass_stats.get("by_generator", {})
                if by_generator:
                    print(f"  By Generator (Top 5):")
                    sorted_gens = sorted(by_generator.items(), key=lambda x: x[1], reverse=True)[:5]
                    for hotkey, count in sorted_gens:
                        hotkey_short = hotkey[:12] + "..." if len(hotkey) > 12 else hotkey
                        print(f"    {hotkey_short}: {count}")

                by_week = misclass_stats.get("by_week", {})
                if by_week:
                    print(f"  By Week:")
                    for week, count in sorted(by_week.items()):
                        print(f"    {week}: {count}")

            # Accuracy by media type (real/synthetic/semisynthetic)
            dataset_info = results.get("dataset_info", {})
            dataset_types = dataset_info.get("dataset_media_types", {})
            if per_dataset and dataset_types:
                by_type = {}
                for ds_name, ds_stats in per_dataset.items():
                    mtype = dataset_types.get(ds_name)
                    if not mtype:
                        continue
                    entry = by_type.setdefault(mtype, {"samples": 0, "correct": 0})
                    entry["samples"] += int(ds_stats.get("total", 0))
                    entry["correct"] += int(ds_stats.get("correct", 0))
                if by_type:
                    print(f"\n🎭 ACCURACY BY MEDIA TYPE:")
                    for mtype, v in by_type.items():
                        samples = v["samples"]
                        correct = v["correct"]
                        acc = (correct / samples) if samples > 0 else 0.0
                        print(f"  {mtype}: {acc:.2%} ({correct}/{samples})")

    validation = benchmark_results.get("validation", {})
    if validation:
        print(f"\n🔍 MODEL VALIDATION:")
        print(f"  Input Shape: {validation.get('input_shape', 'N/A')}")
        print(f"  Input Type: {validation.get('input_type', 'N/A')}")
        print(f"  Output Shape: {validation.get('output_shape', 'N/A')}")

    errors = benchmark_results.get("errors", [])
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for i, error in enumerate(errors[:5], 1):
            print(f"  {i}. {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    print("=" * 80)


def save_results_to_json(
    benchmark_results: Dict, output_dir: Optional[str] = None
) -> str:
    """
    Save benchmark results to a timestamped JSON file.

    Args:
        benchmark_results: Dictionary containing all benchmark results
        output_dir: Directory to save the JSON file (defaults to current directory)

    Returns:
        Path to the saved JSON file
    """
    if output_dir is None:
        output_dir = "."

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modality = benchmark_results.get("modality", "unknown")
    filename = f"gasbench_results_{modality}_{timestamp}.json"
    filepath = output_path / filename

    # Prepare results for JSON serialization
    output_data = {
        "metadata": {
            "run_id": benchmark_results.get("run_id"),
            "timestamp": benchmark_results.get("timestamp"),
            "datetime": datetime.fromtimestamp(
                benchmark_results.get("timestamp", time.time())
            ).isoformat(),
            "model_path": benchmark_results.get("model_path"),
            "modality": benchmark_results.get("modality"),
            "mode": benchmark_results.get("mode", "full"),
            "gasstation_only": benchmark_results.get("gasstation_only", False),
            "benchmark_completed": benchmark_results.get("benchmark_completed", False),
            "duration_seconds": benchmark_results.get("metrics", {}).get(
                "benchmark_duration_seconds", 0
            ),
        },
        "overall_score": benchmark_results.get("benchmark_score", 0.0),
        "validation": benchmark_results.get("validation", {}),
        "errors": benchmark_results.get("errors", []),
    }

    # Add modality-specific results
    modality = benchmark_results.get("modality")
    if modality:
        results_key = f"{modality}_results"
        if results_key in benchmark_results:
            results = benchmark_results[results_key]

            # Extract detailed metrics
            output_data["results"] = {
                "total_samples": results.get("total_samples", 0),
                "correct_predictions": results.get("correct_predictions", 0),
                "accuracy": results.get("benchmark_score", 0.0),
                "avg_inference_time_ms": results.get("avg_inference_time_ms", 0),
                "p95_inference_time_ms": results.get("p95_inference_time_ms", 0),
                "binary_mcc": results.get("binary_mcc", 0.0),
                "multiclass_mcc": results.get("multiclass_mcc", 0.0),
            }

            # Per-source accuracy
            # Accuracy by media type (real/synthetic/semisynthetic)
            per_dataset = results.get("per_dataset_results", {})
            dataset_info = results.get("dataset_info", {})
            dataset_types = dataset_info.get("dataset_media_types", {})
            if per_dataset and dataset_types:
                by_type = {}
                for ds_name, ds_stats in per_dataset.items():
                    mtype = dataset_types.get(ds_name)
                    if not mtype:
                        continue
                    entry = by_type.setdefault(mtype, {"samples": 0, "correct": 0})
                    entry["samples"] += int(ds_stats.get("total", 0))
                    entry["correct"] += int(ds_stats.get("correct", 0))
                output_data["accuracy_by_media_type"] = {
                    t: {
                        "samples": v["samples"],
                        "correct": v["correct"],
                        "accuracy": (v["correct"] / v["samples"]) if v["samples"] > 0 else 0.0,
                    }
                    for t, v in by_type.items()
                }

            # Dataset info
            dataset_info = results.get("dataset_info", {})
            if dataset_info:
                output_data["dataset_info"] = dataset_info
            
            # Misclassification data
            misclassified_samples = results.get("misclassified_samples", [])
            misclassification_stats = results.get("misclassification_stats", {})
            if misclassified_samples:
                output_data["misclassified_samples"] = misclassified_samples
                output_data["misclassification_stats"] = misclassification_stats

    # Write to JSON file
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results summary saved to: {filepath}")

    return str(filepath)
