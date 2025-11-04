# GASBench

A benchmark evaluation package for discriminative models on [Bittensor Subnet 34 (GAS - Generative Adversarial Subnet)](https://github.com/BitMind-AI/bitmind-subnet). This tool evaluates AI-generated content detection models on diverse image and video datasets.

## Overview

This package provides a self-contained benchmark evaluation system for testing models on diverse datasets:

- **Image & Video Benchmarks** - Test models on various datasets for AI-generated content detection
- **Data Processing** - Dataset downloads, caching, preprocessing, and augmentation with aspect ratio preservation
- **Comprehensive Metrics** - Accuracy, MCC, cross-entropy, SN34 score, inference times, and per-dataset breakdowns

## Installation

GPU support is installed by default:

```bash
cd gasbench
pip install -e .
```

For CPU-only (no CUDA):

```bash
pip install -e .[cpu]
```

## Usage

### Command Line Interface

```bash
# Run image benchmark
gasbench model.onnx image

# Run video benchmark
gasbench model.onnx video

# Debug mode (faster, smaller datasets)
gasbench model.onnx image --debug

# Use only gasstation datasets
gasbench model.onnx image --gasstation-only

# Custom cache directory
gasbench model.onnx video --cache-dir /tmp/my_cache

# Save JSON results to a specific directory
gasbench model.onnx image --output-dir ./results
```
Results are automatically saved to a timestamped JSON file (e.g., `gasbench_results_image_20241002_143022.json`) in the current directory or the directory specified by `--output-dir`.

### Python API

```python
import asyncio
from gasbench import run_benchmark, print_benchmark_summary, save_results_to_json

async def evaluate_model():
    results = await run_benchmark(
        model_path="path/to/model.onnx",
        modality="image",  # or "video"
        debug_mode=False,
        gasstation_only=False
    )
    
    print_benchmark_summary(results)
    
    output_path = save_results_to_json(results, output_dir="./results")
    print(f"Results saved to: {output_path}")
    
    return results

results = asyncio.run(evaluate_model())
```

### Low-Level API

For more control over the benchmark process:

```python
import asyncio
from gasbench import run_image_benchmark, run_video_benchmark, create_inference_session

async def evaluate_model():
    # Create ONNX inference session
    session = create_inference_session("path/to/model.onnx", "image")
    input_specs = session.get_inputs()
    
    benchmark_results = {"errors": []}
    
    # Run benchmarks
    image_score = await run_image_benchmark(
        session=session,
        input_specs=input_specs,
        benchmark_results=benchmark_results,
        debug_mode=False,
        gasstation_only=False
    )
    
    video_score = await run_video_benchmark(
        session=session,
        input_specs=input_specs,
        benchmark_results=benchmark_results,
        debug_mode=False,
        gasstation_only=False
    )
    
    print(f"Image: {image_score:.2%}, Video: {video_score:.2%}")
    return benchmark_results

asyncio.run(evaluate_model())
```

## Model Requirements

### Input Shape

**Set explicit spatial dimensions in your ONNX model** (e.g., `[batch, 3, 224, 224]` or `[batch, 3, 384, 384]`).

Models with dynamic dimensions (e.g., `[batch, 3, height, width]`) will fallback to 224×224, but this is not recommended as it may not match your model's training resolution.

### Preprocessing

All inputs are automatically preprocessed with:
- **Shortest-edge resize** to target dimensions (preserves aspect ratio)
- **Center crop** to exact target size (square output)
- **Random augmentations** (rotation, flip, distortions, JPEG compression)

This ensures consistent input sizes for efficient batching while preserving image quality.

### Output Format

For detailed ONNX model requirements, see the [ONNX Model Specification](https://github.com/BitMind-AI/bitmind-subnet/blob/main/docs/ONNX.md) from the BitMind Subnet documentation.

## Configuration

Please see the BitMind Subnet's [Incentive Mechanism docs](https://github.com/BitMind-AI/bitmind-subnet/blob/main/docs/Incentive.md) for a full list of data sources (expand the dropdowns at the top)

### Debug Mode

Use `debug_mode=True` for faster testing with smaller datasets during development.

### Gasstation Only Mode

Use `gasstation_only=True` to evaluate only on gasstation datasets for faster, focused evaluation.

### Cache Directory

By default, datasets are cached at `/tmp/benchmark_data/`. Specify a custom location with `cache_dir` parameter or `--cache-dir` flag.

### Metrics

- **benchmark_score**: Raw accuracy (correct predictions / total samples)
- **sn34_score**: Combined metric [0, 1] averaging normalized MCC and cross-entropy (higher is better)
- **binary_mcc**: Matthews Correlation Coefficient for binary classification (real vs AI)
- **multiclass_mcc**: Matthews Correlation Coefficient for 3-class (real, synthetic, semisynthetic)
- **binary_cross_entropy**: Cross-entropy loss for binary classification
- **multiclass_cross_entropy**: Cross-entropy loss for multiclass classification

### JSON Output

Benchmark results are automatically saved to a timestamped JSON file with the following structure:

```json
{
  "metadata": {
    "run_id": "unique-run-identifier",
    "timestamp": 1696258822.0,
    "datetime": "2024-10-02T14:30:22",
    "model_path": "path/to/model.onnx",
    "modality": "image",
    "debug_mode": false,
    "gasstation_only": false,
    "benchmark_completed": true,
    "duration_seconds": 123.45
  },
  "overall_score": 0.1111,
  "results": {
    "benchmark_score": 0.8523,
    "sn34_score": 0.7891,
    "total_samples": 5000,
    "correct_predictions": 4261,
    "avg_inference_time_ms": 12.3,
    "p95_inference_time_ms": 45.6,
    "binary_mcc": 0.4321,
    "multiclass_mcc": 0.1234,
    "binary_cross_entropy": 0.2345,
    "multiclass_cross_entropy": 0.3456
  },
  "per_source_accuracy": {
    "real": {
      "dataset_name": {
        "samples": 1000,
        "correct": 890,
        "incorrect": 110,
        "accuracy": 0.89
      }
    },
    "synthetic": {
      "...": "..."
    },
    "semisynthetic": {
      "...": "..."
    }
  },
  "validation": {
    "input_shape": "['batch', 3, 'height', 'width']",
    "input_type": "tensor(uint8)",
    "output_shape": "['batch', 2]"
  },
  "errors": []
}
```
