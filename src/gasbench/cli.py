#!/usr/bin/env python3
"""Command-line interface for GASBench."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .benchmark import run_benchmark, print_benchmark_summary, save_results_to_json


def add_common_args(parser):
    """Add common arguments shared by multiple commands."""
    parser.add_argument(
        "--cache-dir", help="Directory for caching datasets (default: /.cache/gasbench)"
    )


def add_mode_args(parser):
    """Add mode selection arguments."""
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--debug",
        action="store_const",
        const="debug",
        dest="mode",
        help="Debug mode: only use first image and video datasets for quick testing",
    )
    mode_group.add_argument(
        "--small",
        action="store_const",
        const="small",
        dest="mode",
        help="Small mode: download only 1 archive per dataset, extract 100 items from each",
    )
    mode_group.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="mode",
        help="Full mode: use complete configurations from yaml file (default)",
    )


def command_run(args):
    """Execute the benchmark run command."""
    # Default mode is 'full' if no mode flag is provided
    mode = args.mode if args.mode else "full"

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not model_path.suffix.lower() == ".onnx":
        print(
            f"Error: Model file must be an ONNX file (.onnx), got: {model_path.suffix}"
        )
        return 1

    # Print configuration as JSON for clarity
    config = {
        "model": str(model_path),
        "modality": args.modality.upper(),
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
                modality=args.modality,
                mode=mode,
                gasstation_only=args.gasstation_only,
                cache_dir=args.cache_dir,
                download_latest_gasstation_data=args.download_latest_gasstation_data,
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

    print(json.dumps(config, indent=2))
    print("-" * 60)

    try:
        # TODO: Implement efficient concurrent download
        from .download_manager import download_datasets

        asyncio.run(
            download_datasets(
                modality=args.modality,
                mode=args.mode if args.mode else "full",
                gasstation_only=args.gasstation_only,
                cache_dir=args.cache_dir,
                concurrent_downloads=args.concurrent,
                num_weeks=args.num_weeks,
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

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

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
  gasbench run model.onnx image --debug
  
  # Run video benchmark with custom cache directory
  gasbench run model.onnx video --cache-dir /tmp/my_cache
  
  # Run only gasstation datasets
  gasbench run model.onnx image --gasstation-only
        """,
    )

    run_parser.add_argument("model_path", help="Path to ONNX model file")
    run_parser.add_argument(
        "modality", choices=["image", "video"], help="Model modality to benchmark"
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
        "--output-dir",
        help="Directory to save JSON results file (default: current directory)",
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

    download_parser.set_defaults(func=command_download)

    # Parse arguments and execute command
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
