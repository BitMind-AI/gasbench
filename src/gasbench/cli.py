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
    mode = args.mode if args.mode else "full"

    # Determine which model and modality to use
    if args.image_model:
        model_path = Path(args.image_model)
        modality = "image"
    elif args.video_model:
        model_path = Path(args.video_model)
        modality = "video"
    elif args.audio_model:
        model_path = Path(args.audio_model)
        modality = "audio"
    else:
        print("Error: Must specify either --image-model, --video-model, or --audio-model")
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
    results_dir = args.results_dir if args.results_dir else "results"
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
                holdout_config=getattr(args, "holdout_config", None),
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


def command_verify_cache(args):
    """Execute the cache verification command."""
    from collections import defaultdict
    from .dataset.cache import (
        scan_cache_directory,
        compute_cache_statistics,
        verify_cache_against_configs,
        format_size_bytes,
    )

    cache_dir = args.cache_dir if args.cache_dir else "/.cache/gasbench"

    print(f"\n🔍 Verifying cache at: {cache_dir}")
    print("-" * 60)

    datasets = scan_cache_directory(cache_dir)

    if not datasets:
        print("📭 No cached datasets found")
        if args.dataset_config or args.holdout_config:
            return _verify_against_config(set(), args, cache_dir)
        return 0

    cached_names = {ds["name"] for ds in datasets}
    total_samples = sum(ds["sample_count"] for ds in datasets)
    total_size_bytes = sum(ds["size_bytes"] for ds in datasets)

    by_modality, by_media_type = compute_cache_statistics(datasets)

    print(f"\n📊 CACHE SUMMARY")
    print(f"Total Datasets: {len(datasets)}")
    print(f"Total Samples:  {total_samples:,}")
    print(f"Total Size:     {format_size_bytes(total_size_bytes)}")

    print(f"\n📈 BY MODALITY")
    for modality in sorted(by_modality.keys()):
        stats = by_modality[modality]
        print(
            f"  {modality.upper():12} {stats['count']:3} datasets, {stats['samples']:7,} samples, {format_size_bytes(stats['size']):>10}"
        )

    print(f"\n🎭 BY MEDIA TYPE")
    for mtype in sorted(by_media_type.keys()):
        stats = by_media_type[mtype]
        print(
            f"  {mtype:15} {stats['count']:3} datasets, {stats['samples']:7,} samples, {format_size_bytes(stats['size']):>10}"
        )

    if args.verbose:
        print(f"\n📋 DETAILED LISTING")

        datasets_by_mod = defaultdict(list)
        for ds in datasets:
            datasets_by_mod[ds["modality"]].append(ds)

        for modality in sorted(datasets_by_mod.keys()):
            print(f"\n  {modality.upper()} DATASETS:")
            for ds in sorted(datasets_by_mod[modality], key=lambda x: x["name"]):
                size_str = format_size_bytes(ds["size_bytes"])
                print(
                    f"    • {ds['name']:50} {ds['sample_count']:5,} samples  {size_str:>10}  [{ds['media_type']}]"
                )

    if args.dataset_config or args.holdout_config:
        return _verify_against_config(cached_names, args, cache_dir)

    return 0


def _verify_against_config(cached_names, args, cache_dir):
    """Verify cache completeness against config files."""
    from collections import defaultdict
    from .dataset.cache import verify_cache_against_configs

    try:
        present, missing, expected_datasets = verify_cache_against_configs(
            cached_names=cached_names,
            dataset_config=args.dataset_config,
            holdout_config=args.holdout_config,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"\n⚠️  Error loading configs: {e}")
        return 1

    if not expected_datasets:
        return 0

    expected_names = {ds.name for ds in expected_datasets}

    print(f"\n🔎 CONFIG VERIFICATION")
    print(f"Expected Datasets: {len(expected_names)}")
    print(f"Present in Cache:  {len(present)} ✅")
    print(f"Missing from Cache: {len(missing)} {'❌' if missing else '✅'}")

    if missing:
        print(f"\n❌ MISSING DATASETS:")
        by_modality = defaultdict(list)
        for ds in expected_datasets:
            if ds.name in missing:
                by_modality[ds.modality].append(ds)

        for modality in sorted(by_modality.keys()):
            datasets_list = by_modality[modality]
            print(f"\n  {modality.upper()}:")
            for ds in sorted(datasets_list, key=lambda x: x.name):
                print(f"    • {ds.name:50} [{ds.media_type}]")

        return 1

    print("\n✅ All expected datasets are present in cache")
    return 0


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
    if hasattr(args, "datasets") and args.datasets:
        config["datasets"] = args.datasets
    if hasattr(args, "cache_policy") and args.cache_policy:
        config["cache_policy"] = args.cache_policy
    if hasattr(args, "unlimited_samples") and args.unlimited_samples:
        config["unlimited_samples"] = True
    if hasattr(args, "no_eviction") and args.no_eviction:
        config["allow_eviction"] = False

    print(json.dumps(config, indent=2))
    print("-" * 60)

    try:
        from .download_manager import download_datasets

        result = asyncio.run(
            download_datasets(
                modality=args.modality,
                mode=args.mode if args.mode else "full",
                gasstation_only=args.gasstation_only,
                no_gasstation=getattr(args, "skip_gasstation", False),
                cache_dir=args.cache_dir,
                concurrent_downloads=args.concurrent,
                num_weeks=args.num_weeks,
                seed=args.seed,
                cache_policy=getattr(args, "cache_policy", None),
                allow_eviction=not getattr(args, "no_eviction", False),
                unlimited_samples=getattr(args, "unlimited_samples", False),
                dataset_config=getattr(args, "dataset_config", None),
                holdout_config=getattr(args, "holdout_config", None),
                dataset_filters=getattr(args, "datasets", None),
            )
        )

        if result and result["completed"] > 0:
            print(f"\n✅ Successfully downloaded {result['completed']}/{result['total']} datasets")
            return 0 if result["failed"] == 0 else 1
        else:
            print("\n❌ No datasets were successfully downloaded")
            return 1

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

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

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
        help="Path to ONNX image detection model",
    )
    model_group.add_argument(
        "--video-model",
        type=str,
        metavar="PATH",
        help="Path to ONNX video detection model",
    )
    model_group.add_argument(
        "--audio-model",
        type=str,
        metavar="PATH",
        help="Path to ONNX audio detection model"
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
        default="results",
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
  
  # Download specific datasets by name
  gasbench download --datasets pica-100k ffhq --debug
  
  # Download only gasstation datasets with 4 concurrent downloads
  gasbench download --gasstation-only --concurrent 4
  
  # Skip gasstation datasets (download everything else)
  gasbench download --skip-gasstation
  
  # Download debug datasets (fast, for testing)
  gasbench download --debug
  
  # Download last 4 weeks of gasstation data
  gasbench download --gasstation-only --num-weeks 4
        """,
    )

    download_parser.add_argument(
        "--modality",
        choices=["image", "video", "audio", "all"],
        help="Download datasets for specific modality (default: all)",
    )
    download_parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        metavar="NAME",
        help="Filter datasets by name (space-separated list, supports partial matches)",
    )

    add_mode_args(download_parser)
    add_common_args(download_parser)

    download_parser.add_argument(
        "--gasstation-only",
        action="store_true",
        help="Only download gasstation datasets",
    )
    download_parser.add_argument(
        "--skip-gasstation",
        action="store_true",
        help="Skip gasstation datasets (download everything else)",
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

    # ========== PREPROCESS COMMAND ==========
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess audio datasets and cache as tensors for faster benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess a specific audio dataset with GPU acceleration
  gasbench preprocess --dataset deepfake-audio-dataset --gpu
  
  # Preprocess all audio datasets
  gasbench preprocess --all --gpu
  
  # Preprocess with custom cache directory
  gasbench preprocess --all --cache-dir /custom/cache --gpu
        """,
    )

    preprocess_parser.add_argument(
        "--dataset",
        type=str,
        help="Name of audio dataset to preprocess (e.g., deepfake-audio-dataset)",
    )
    preprocess_parser.add_argument(
        "--all",
        action="store_true",
        help="Preprocess all audio datasets",
    )
    preprocess_parser.add_argument(
        "--cache-dir",
        type=str,
        help="Base cache directory (default: /.cache/gasbench)",
    )
    preprocess_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for preprocessing (faster if CUDA available)",
    )
    preprocess_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU processing (default: 32)",
    )

    preprocess_parser.set_defaults(func=command_preprocess)

    # ========== VERIFY-CACHE COMMAND ==========
    verify_parser = subparsers.add_parser(
        "verify-cache",
        help="Verify and display cache contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show cache summary
  gasbench verify-cache
  
  # Show detailed listing of all datasets
  gasbench verify-cache --verbose
  
  # Check cache against holdout config
  gasbench verify-cache --holdout-config /path/to/holdout-v2.yaml --cache-dir /workspace/benchmark/data/
  
  # Verify both main and holdout configs
  gasbench verify-cache --dataset-config dataset.yaml --holdout-config holdout.yaml
        """,
    )

    add_common_args(verify_parser)

    verify_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed listing of all cached datasets",
    )
    verify_parser.add_argument(
        "--dataset-config",
        type=str,
        metavar="PATH",
        help="Verify cache against dataset YAML config file",
    )
    verify_parser.add_argument(
        "--holdout-config",
        type=str,
        metavar="PATH",
        help="Verify cache against holdout YAML config file",
    )

    verify_parser.set_defaults(func=command_verify_cache)

    args = parser.parse_args()
    return args.func(args)


def command_preprocess(args):
    """Execute the audio preprocessing command."""
    print("\n🔄 Starting audio dataset preprocessing")
    
    if not args.dataset and not args.all:
        print("Error: Must specify either --dataset or --all")
        return 1
    
    cache_dir = args.cache_dir or "/.cache/gasbench"
    
    config = {
        "cache_dir": cache_dir,
        "use_gpu": args.gpu,
        "batch_size": args.batch_size,
    }
    
    if args.dataset:
        config["dataset"] = args.dataset
    
    print(json.dumps(config, indent=2))
    print("-" * 60)
    
    try:
        from .processing.preprocess_audio import preprocess_dataset, preprocess_all_datasets
        
        if args.all:
            preprocess_all_datasets(cache_dir=cache_dir, use_gpu=args.gpu)
        else:
            success = preprocess_dataset(
                dataset_name=args.dataset,
                cache_dir=cache_dir,
                use_gpu=args.gpu,
                batch_size=args.batch_size
            )
            if not success:
                return 1
        
        print("\n✅ Preprocessing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        return 130
    except Exception as e:
        print(f"\nPreprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
