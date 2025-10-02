#!/usr/bin/env python3
import os
import time
import uuid
import asyncio
from typing import Dict, Optional

from .logger import get_logger
from .processing.media import configure_huggingface_cache
from .benchmarks.image_bench import run_image_benchmark
from .benchmarks.video_bench import run_video_benchmark
from .model.inference import create_inference_session

logger = get_logger(__name__)


async def run_benchmark(
    model_path: str,
    modality: str,
    debug_mode: bool = False,
    gasstation_only: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict:
    """
    Args:
        model_path: Path to ONNX model file
        modality: Type of modality to test ("image" or "video")
        debug_mode: Use DEBUG_BENCHMARK_DATASETS for faster testing
        gasstation_only: If True, only use gasstation datasets
        cache_dir: Directory for caching (defaults to /.cache/gasbench)

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
        "debug_mode": debug_mode,
        "gasstation_only": gasstation_only,
    }

    start_time = time.time()

    try:
        logger.info(
            f"🎯 BENCHMARK START: {modality.upper()} model | Path: {model_path}"
        )

        session, input_specs = await load_model_for_benchmark(
            model_path, modality, benchmark_results
        )
        
        if not session:
            return benchmark_results

        # Run benchmark for the specified modality
        benchmark_score = await execute_benchmark(
            session, input_specs, modality, benchmark_results, debug_mode, gasstation_only, cache_dir
        )

        benchmark_results["benchmark_score"] = benchmark_score

        benchmark_results["metrics"]["modality"] = modality
        benchmark_results["metrics"]["model_path"] = model_path
        benchmark_results["benchmark_completed"] = True

        logger.info(f"📊 Benchmark score: {benchmark_score:.2%}")
        logger.info(f"✅ Benchmark COMPLETED for {modality} modality")

    except Exception as e:
        logger.error(f"❌ Benchmark failed with error: {e}")
        benchmark_results["errors"].append(f"Benchmark error: {str(e)}")
        benchmark_results["benchmark_completed"] = False

        # Initialize missing result structures
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


async def load_model_for_benchmark(model_path: str, modality: str, benchmark_results: Dict):
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

        logger.info(f"✅ Model loaded successfully: {model_path}")
        logger.info(f"📋 Input shape: {input_specs[0].shape}")
        logger.info(f"📋 Output shape: {output_specs[0].shape}")

        return session, input_specs

    except Exception as e:
        logger.error(f"❌ Failed to load model for inference: {e}")
        benchmark_results["errors"].append(f"ONNX runtime error: {str(e)}")
        benchmark_results["benchmark_completed"] = False
        return None, None


async def execute_benchmark(
    session, 
    input_specs, 
    modality: str, 
    benchmark_results: Dict, 
    debug_mode: bool, 
    gasstation_only: bool = False,
    cache_dir: str = "/.cache/gasbench"
) -> float:
    """Execute the actual benchmark evaluation."""
    
    logger.info(f"Running {modality} benchmark (gasstation_only={gasstation_only})")
    if modality == "image":
        benchmark_score = await run_image_benchmark(
            session,
            input_specs,
            benchmark_results,
            debug_mode,
            gasstation_only,
            cache_dir,
        )
    elif modality == "video":
        benchmark_score = await run_video_benchmark(
            session, 
            input_specs, 
            benchmark_results, 
            debug_mode, 
            gasstation_only, 
            cache_dir
        )
    else:
        raise ValueError(f"Invalid modality: {modality}. Must be 'image' or 'video'")
    
    return benchmark_score


def print_benchmark_summary(benchmark_results: Dict):
    """Print a comprehensive summary of benchmark results."""
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    print(f"Model Path: {benchmark_results.get('model_path', 'N/A')}")
    print(f"Modality: {benchmark_results.get('modality', 'N/A').upper()}")
    print(f"Run ID: {benchmark_results.get('run_id', 'N/A')}")
    print(f"Debug Mode: {benchmark_results.get('debug_mode', False)}")
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
            
            avg_time = results.get('avg_inference_time_ms', 0)
            p95_time = results.get('p95_inference_time_ms', 0)
            if avg_time > 0:
                print(f"  Avg Inference Time: {avg_time:.1f}ms")
            if p95_time > 0:
                print(f"  P95 Inference Time: {p95_time:.1f}ms")
            
            binary_mcc = results.get('binary_mcc', 0.0)
            multiclass_mcc = results.get('multiclass_mcc', 0.0)
            if binary_mcc != 0.0:
                print(f"  Binary MCC: {binary_mcc:.4f}")
            if multiclass_mcc != 0.0:
                print(f"  Multiclass MCC: {multiclass_mcc:.4f}")
            
            per_dataset = results.get('per_dataset_results', {})
            if per_dataset:
                print(f"\n📋 PER-DATASET RESULTS:")
                for dataset_name, dataset_results in per_dataset.items():
                    accuracy = dataset_results.get('accuracy', 0.0)
                    total = dataset_results.get('total', 0)
                    correct = dataset_results.get('correct', 0)
                    print(f"  {dataset_name}: {accuracy:.2%} ({correct}/{total})")
            
            per_source = results.get('per_source_accuracy', {})
            if per_source:
                print(f"\n🎭 PER-SOURCE ACCURACY:")
                for source_type, datasets in per_source.items():
                    print(f"  {source_type.upper()}:")
                    for dataset_name, stats in datasets.items():
                        correct = stats.get('correct', 0)
                        incorrect = stats.get('incorrect', 0)
                        total = correct + incorrect
                        accuracy = correct / total if total > 0 else 0.0
                        print(f"    {dataset_name}: {accuracy:.2%} ({correct}/{total})")
    
    validation = benchmark_results.get("validation", {})
    if validation:
        print(f"\n🔍 MODEL VALIDATION:")
        print(f"  Input Shape: {validation.get('input_shape', 'N/A')}")
        print(f"  Input Type: {validation.get('input_type', 'N/A')}")
        print(f"  Output Shape: {validation.get('output_shape', 'N/A')}")
    
    errors = benchmark_results.get("errors", [])
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for i, error in enumerate(errors[:5], 1):
            print(f"  {i}. {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    print("="*80)
