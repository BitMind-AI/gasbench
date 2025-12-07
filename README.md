# GASBench

A benchmark evaluation package for discriminative models on [Bittensor Subnet 34 (GAS - Generative Adversarial Subnet)](https://github.com/BitMind-AI/bitmind-subnet).  

GASBench evaluates AI-generated content detection models across **image**, **video**, and **audio** modalities.

## Overview

This package provides a self-contained benchmark evaluation system for testing models on diverse datasets:

- **Image, Video & Audio Benchmarks**  
  Test discriminative models on curated datasets for AI-generated content detection

- **Data Processing**  
  Dataset download, caching, preprocessing, and augmentation with aspect ratio preservation for image/video, and standardized resampling/windowing for audio

- **Comprehensive Metrics**  
  Accuracy, MCC, cross-entropy, inference times, and per-dataset breakdowns

For detailed ONNX model requirements for all three modalities, see the  
👉 **[ONNX Model Specification](./docs/ONNX.md)**

---

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

---

## Usage

### Command Line Interface

```bash
# Run image benchmark
gasbench model.onnx image

# Run video benchmark
gasbench model.onnx video

# Run audio benchmark
gasbench model.onnx audio

# Debug mode (faster, smaller datasets)
gasbench model.onnx image --debug
gasbench model.onnx video --debug
gasbench model.onnx audio --debug

# Custom cache directory
gasbench model.onnx video --cache-dir /tmp/my_cache

# Save JSON results to a specific directory
gasbench model.onnx image --output-dir ./results
gasbench model.onnx video --output-dir ./results
gasbench model.onnx audio --output-dir ./results

# For SN34 miners: Use only gasstation datasets
gasbench model.onnx image --gasstation-only
gasbench model.onnx video --gasstation-only
gasbench model.onnx audio --gasstation-only

```

Results are automatically saved to a timestamped JSON file.

---

## Python API

```python
import asyncio
from gasbench import run_benchmark, print_benchmark_summary, save_results_to_json

async def evaluate_model():
    results = await run_benchmark(
        model_path="path/to/model.onnx",
        modality="image",  # "image" | "video" | "audio"
        debug_mode=False,
        gasstation_only=False,
    )

    print_benchmark_summary(results)

    output_path = save_results_to_json(results, output_dir="./results")
    return results

results = asyncio.run(evaluate_model())
```

---

## Model Requirements (High-Level)

- Models must be exported as **ONNX classifiers** (image, video, or audio).
- GASBench expects **batched inputs** and **logits outputs**.
- Image/video preprocessing (resize/crop/augment) and audio preprocessing (mono/resample/crop) are handled by GASBench.
- Both **binary (real vs synthetic)** and **3-class (real, semisynthetic, synthetic)** models are supported.

For exact ONNX requirements, see:  
👉 **[ONNX Model Specification](./docs/ONNX.md)**

---

## Metrics

- **benchmark_score**
- **sn34_score**
- **binary_mcc**
- **binary_cross_entropy**
- **avg_inference_time_ms**
- **p95_inference_time_ms**

---

### Cache Directory
By default, datasets are cached at `/tmp/benchmark_data/`. Specify a custom location with `cache_dir` parameter or `--cache-dir` flag.

---

## JSON Output Structure

A structured JSON summary is automatically generated after each run:

```json
{
  "metadata": {
    "run_id": "64a8c5eb-560a-4822-9ae5-b51d27737831",
    "timestamp": 1765074390.323,
    "datetime": "2025-12-07T02:26:30.323068",
    "model_path": "model.onnx",
    "modality": "audio",
    "mode": "full",
    "gasstation_only": false,
    "benchmark_completed": true,
    "duration_seconds": 1054.31
  },
  "overall_score": 0.8523,
  "validation": {
    "model_path": "model.onnx",
    "input_shape": "['batch_size', 96000]",
    "input_type": "tensor(float)",
    "output_shape": "['batch_size', 2]"
  },
  "errors": [],
  "results": {
    "total_samples": 5000,
    "correct_predictions": 4261,
    "accuracy": 0.8522,
    "avg_inference_time_ms": 12.3,
    "p95_inference_time_ms": 45.6,
    "binary_mcc": 0.7045,
    "binary_cross_entropy": 0.2345,
    "sn34_score": 0.7891
  },
  "accuracy_by_media_type": {
    "real": {
      "samples": 2000,
      "correct": 1780,
      "accuracy": 0.89
    },
    "synthetic": {
      "samples": 2500,
      "correct": 2200,
      "accuracy": 0.88
    }
  },
  "dataset_info": {
    "datasets_used": ["dataset-a", "dataset-b", "..."],
    "evaluation_type": "synthetic_detection",
    "dataset_media_types": {
      "dataset-a": "real",
      "dataset-b": "synthetic"
    },
    "samples_per_dataset": {
      "dataset-a": 100,
      "dataset-b": 100
    }
  }
}
```
