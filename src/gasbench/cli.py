#!/usr/bin/env python3

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from . import __version__
from .benchmark import run_benchmark, print_benchmark_summary, save_results_to_json


def add_common_args(parser):
    """Add common arguments shared by multiple commands."""
    parser.add_argument(
        "--cache-dir", 
        help="Directory for caching datasets (default: /.cache/gasbench)"
    )


def add_mode_args(parser):
    """Add mode selection arguments."""
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


def command_run(args):
    """Execute the benchmark run command."""
    mode = args.mode if args.mode else "full"

    # Determine which model and modality to use
    if args.image_model:
        model_path = Path(args.image_model)
        modality = "image"
    elif args.video_model:
        model_path = Path(args.video_model)
        modality = "video"
    else:
        print("Error: Must specify either --image-model or --video-model")
        return 1

    # Validate model path
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not model_path.suffix.lower() == ".onnx":
        print(
            f"Error: Model file must be an ONNX file (.onnx), got: {model_path.suffix}"
        )
        return 1

    # Print configuration as JSON for clarity
    results_dir = getattr(args, 'results_dir', 'results')
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parquet_path = str(results_path / f"records_{modality}_{timestamp}.parquet")
    
    config = {
        "model": str(model_path),
        "modality": modality.upper(),
        "mode": mode.upper(),
        "gasstation_only": args.gasstation_only,
        "download_latest_gasstation_data": args.download_latest_gasstation_data,
        "results_directory": results_dir,
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
                download_latest_gasstation_data=args.download_latest_gasstation_data,
                seed=args.seed,
                batch_size=args.batch_size,
                dataset_config=args.dataset_config,
                holdout_config=getattr(args, 'holdout_config', None),
                records_parquet_path=parquet_path,
            )
        )

        print_benchmark_summary(results)

        output_path = save_results_to_json(results, output_dir=results_dir)
        print(f"\n📊 Results saved to: {results_dir}")
        print(f"  - JSON summary: {output_path}")
        res_key = f"{modality}_results"
        ppath = results.get(res_key, {}).get("parquet_path")
        if ppath:
            print(f"  - Parquet records: {ppath}")

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


def command_download(args):
    """Execute the dataset download command."""
    print("\n📥 Starting gasbench dataset download")

    config = {
        "modality": args.modality.upper() if args.modality else "ALL",
        "mode": (args.mode if args.mode else "full").upper(),
        "gasstation_only": args.gasstation_only,
        "concurrent_downloads": args.concurrent,
    }
    if args.cache_dir:
        config["cache_directory"] = args.cache_dir
    if args.num_weeks:
        config["num_weeks"] = args.num_weeks
    if hasattr(args, 'cache_policy') and args.cache_policy:
        config["cache_policy"] = args.cache_policy
    if hasattr(args, 'unlimited_samples') and args.unlimited_samples:
        config["unlimited_samples"] = True
    if hasattr(args, 'no_eviction') and args.no_eviction:
        config["allow_eviction"] = False

    print(json.dumps(config, indent=2))
    print("-" * 60)

    try:
        from .download_manager import download_datasets

        asyncio.run(
            download_datasets(
                modality=args.modality,
                mode=args.mode if args.mode else "full",
                gasstation_only=args.gasstation_only,
                cache_dir=args.cache_dir,
                concurrent_downloads=args.concurrent,
                num_weeks=args.num_weeks,
                seed=args.seed,
                cache_policy=getattr(args, 'cache_policy', None),
                allow_eviction=not getattr(args, 'no_eviction', False),
                unlimited_samples=getattr(args, 'unlimited_samples', False),
                dataset_config=getattr(args, 'dataset_config', None),
                holdout_config=getattr(args, 'holdout_config', None),
            )
        )

        print("\n✅ Download completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        return 130
    except Exception as e:
        print(f"\nDownload failed with error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GASBench - ML Model Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available gasbench commands",
        dest="command",
        required=True,
        help="Command to execute",
    )

    # ========== RUN COMMAND ==========
    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmark evaluation on a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run image benchmark in debug mode
  gasbench run --image-model model.onnx --debug
  
  # Run video benchmark with custom cache directory and save results
  gasbench run --video-model model.onnx --cache-dir /tmp/my_cache --results-dir ./results
  
  # Run only gasstation datasets
  gasbench run --image-model model.onnx --gasstation-only
        """,
    )

    # Model selection (mutually exclusive)
    model_group = run_parser.add_mutually_exclusive_group(required=True)
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

    add_mode_args(run_parser)
    add_common_args(run_parser)

    run_parser.add_argument(
        "--gasstation-only",
        action="store_true",
        help="Only use gasstation datasets for evaluation",
    )
    run_parser.add_argument(
        "--download-latest-gasstation-data",
        action="store_true",
        help="Download latest gasstation data before benchmarking (default: False, uses cached data)",
    )
    run_parser.add_argument(
        "--results-dir",
        type=str,
        metavar="PATH",
        help="Directory to save results (default: results/)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for non-gasstation dataset sampling (for reproducible random sampling)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for model inference (default: 32 for images, 4 for videos)",
    )
    run_parser.add_argument(
        "--dataset-config",
        type=str,
        metavar="PATH",
        help="Path to custom dataset YAML config file (default: uses bundled config)",
    )
    run_parser.add_argument(
        "--holdout-config",
        type=str,
        metavar="PATH",
        help="Path to private holdout YAML with additional datasets (names will be obfuscated)",
    )

    run_parser.set_defaults(func=command_run)

    # ========== DOWNLOAD COMMAND ==========
    download_parser = subparsers.add_parser(
        "download",
        help="Download benchmark datasets efficiently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all image datasets in full mode
  gasbench download --modality image --full
  
  # Download only gasstation datasets with 4 concurrent downloads
  gasbench download --gasstation-only --concurrent 4
  
  # Download debug datasets (fast, for testing)
  gasbench download --debug
  
  # Download last 4 weeks of gasstation data
  gasbench download --gasstation-only --num-weeks 4
        """,
    )

    download_parser.add_argument(
        "--modality",
        choices=["image", "video", "all"],
        help="Download datasets for specific modality (default: all)",
    )

    add_mode_args(download_parser)
    add_common_args(download_parser)

    download_parser.add_argument(
        "--gasstation-only",
        action="store_true",
        help="Only download gasstation datasets",
    )
    download_parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Number of concurrent downloads (default: 4)",
    )
    download_parser.add_argument(
        "--num-weeks",
        type=int,
        help="For gasstation datasets: number of recent weeks to download (default: current week only)",
    )
    download_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for non-gasstation dataset sampling (for reproducible random sampling)",
    )
    download_parser.add_argument(
        "--cache-policy",
        help="Path to cache policy JSON file for intelligent sample eviction",
    )
    download_parser.add_argument(
        "--no-eviction",
        action="store_true",
        help="Disable sample eviction and accumulate all samples (even beyond cache limits)",
    )
    download_parser.add_argument(
        "--unlimited-samples",
        action="store_true",
        help="Download ALL available samples (no 10k/5k cap)",
    )
    download_parser.add_argument(
        "--dataset-config",
        type=str,
        metavar="PATH",
        help="Path to custom dataset YAML config file (default: uses bundled config)",
    )
    download_parser.add_argument(
        "--holdout-config",
        type=str,
        metavar="PATH",
        help="Path to private holdout YAML with additional datasets (names will be obfuscated)",
    )

    download_parser.set_defaults(func=command_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
