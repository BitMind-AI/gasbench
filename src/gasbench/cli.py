#!/usr/bin/env python3
"""Command-line interface for GASBench."""

import argparse
import asyncio
import sys
from pathlib import Path

from .benchmark import run_benchmark, print_benchmark_summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GASBench - ML Model Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run image benchmark in debug mode
  gasbench model.onnx image --debug
  
  # Run video benchmark with custom cache directory
  gasbench model.onnx video --cache-dir /tmp/my_cache
  
  # Run only gasstation datasets
  gasbench model.onnx image --gasstation-only
        """
    )
    
    parser.add_argument(
        "model_path", 
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "modality", 
        choices=["image", "video"], 
        help="Model modality to benchmark"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Use debug mode with smaller datasets for faster testing"
    )
    parser.add_argument(
        "--gasstation-only", 
        action="store_true", 
        help="Only use gasstation datasets for evaluation"
    )
    parser.add_argument(
        "--cache-dir", 
        help="Directory for caching datasets (default: /.cache/gasbench)"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        return 1
    
    if not model_path.suffix.lower() == '.onnx':
        print(f"❌ Error: Model file must be an ONNX file (.onnx), got: {model_path.suffix}")
        return 1
    
    print("⛽ Starting gasbench evaluation ⛽")
    print(f"  Model: {model_path}")
    print(f"  Modality: {args.modality.upper()}")
    print(f"  Debug Mode: {args.debug}")
    print(f"  Gasstation Only: {args.gasstation_only}")
    if args.cache_dir:
        print(f"  Cache Directory: {args.cache_dir}")
    print("-" * 60)
    
    try:
        results = asyncio.run(
            run_benchmark(
                model_path=str(model_path),
                modality=args.modality,
                debug_mode=args.debug,
                gasstation_only=args.gasstation_only,
                cache_dir=args.cache_dir
            )
        )
        
        print_benchmark_summary(results)
        
        if results.get("benchmark_completed"):
            print("\n✅ Benchmark completed successfully!")
            return 0
        else:
            print("\n❌ Benchmark failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
