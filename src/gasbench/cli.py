#!/usr/bin/env python3
"""Command-line interface for GASBench."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .benchmark import run_benchmark, print_benchmark_summary, save_results_to_json


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GASBench - ML Model Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run image benchmark in debug mode
  gasbench --image-model model.onnx --debug
  
  # Run video benchmark with custom cache directory
  gasbench --video-model model.onnx --cache-dir /tmp/my_cache
  
  # Run only gasstation datasets
  gasbench --image-model model.onnx --gasstation-only
        """
    )
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--image-model",
        type=str,
        metavar="PATH",
        help="Path to ONNX image detection model"
    )
    model_group.add_argument(
        "--video-model",
        type=str,
        metavar="PATH",
        help="Path to ONNX video detection model"
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--debug", 
        action="store_const",
        const="debug",
        dest="mode",
        help="Debug mode: only use first image and video datasets for quick testing"
    )
    mode_group.add_argument(
        "--small", 
        action="store_const",
        const="small",
        dest="mode",
        help="Small mode: download only 1 archive per dataset, extract 100 items from each"
    )
    mode_group.add_argument(
        "--full", 
        action="store_const",
        const="full",
        dest="mode",
        help="Full mode: use complete configurations from yaml file (default)"
    )
    
    parser.add_argument(
        "--gasstation-only", 
        action="store_true", 
        help="Only use gasstation datasets for evaluation"
    )
    parser.add_argument(
        "--download-latest-gasstation-data",
        action="store_true",
        help="Download latest gasstation data before benchmarking (default: False, uses cached data)"
    )
    parser.add_argument(
        "--cache-dir", 
        help="Directory for caching datasets (default: /.cache/gasbench)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save JSON results file (default: current directory)"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Default mode is 'full' if no mode flag is provided
    mode = args.mode if args.mode else "full"
    
    # Determine model path and modality
    if args.image_model:
        model_path = Path(args.image_model)
        modality = "image"
    else:  # args.video_model
        model_path = Path(args.video_model)
        modality = "video"
    
    # Validate model path
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    if not model_path.suffix.lower() == '.onnx':
        print(f"Error: Model file must be an ONNX file (.onnx), got: {model_path.suffix}")
        return 1
    
    # Print configuration as JSON for clarity
    config = {
        "model": str(model_path),
        "modality": modality.upper(),
        "mode": mode.upper(),
        "gasstation_only": args.gasstation_only,
        "download_latest_gasstation_data": args.download_latest_gasstation_data,
    }
    if args.cache_dir:
        config["cache_directory"] = args.cache_dir
    
    print("\n🎯 Starting gasbench evaluation")
    print(json.dumps(config, indent=2))
    print("-" * 60)
    
    try:
        results = asyncio.run(
            run_benchmark(
                model_path=str(model_path),
                modality=modality,
                mode=mode,
                gasstation_only=args.gasstation_only,
                cache_dir=args.cache_dir,
                download_latest_gasstation_data=args.download_latest_gasstation_data
            )
        )
        
        print_benchmark_summary(results)
        
        # Save results to JSON file
        output_path = save_results_to_json(results, output_dir=args.output_dir)
        print(f"\nResults saved to: {output_path}")
        
        if results.get("benchmark_completed"):
            print("\n✅ Benchmark completed successfully")
            return 0
        else:
            print("\nBenchmark failed")
            return 1
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
