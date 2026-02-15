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

For model submission requirements, see the  
👉 **[Safetensors Model Specification](./docs/Safetensors.md)** (required for competition)

> **Note**: ONNX format is no longer accepted for competition. See [ONNX.md](./docs/ONNX.md) for legacy reference.

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
# Run image benchmark with safetensors model
gasbench run --image-model ./my_image_model/ --debug

# Run video benchmark
gasbench run --video-model ./my_video_model/ --debug

# Run audio benchmark
gasbench run --audio-model ./my_audio_model/ --debug

# Full benchmark (all datasets)
gasbench run --image-model ./my_model/ --full

# Custom cache directory
gasbench run --video-model ./my_model/ --cache-dir /tmp/my_cache

# Save results to a specific directory
gasbench run --image-model ./my_model/ --results-dir ./results

# For SN34 miners: Use only gasstation datasets
gasbench run --image-model ./my_model/ --gasstation-only
```

Model directory must contain: `model_config.yaml`, `model.py`, `*.safetensors`

Results are automatically saved to a timestamped JSON file.

---

## Python API

```python
import asyncio
from gasbench import run_benchmark, print_benchmark_summary, save_results_to_json

async def evaluate_model():
    results = await run_benchmark(
        model_path="path/to/my_model/",  # directory with model_config.yaml, model.py, *.safetensors
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

- Models must be submitted in **safetensors format** as a directory containing `model_config.yaml`, `model.py`, and `*.safetensors` weights.
- Your `model.py` must define a `load_model(weights_path, num_classes)` function that returns a PyTorch `nn.Module`.
- GASBench expects **batched inputs** (raw 0-255 pixel values for image/video, waveform tensors for audio) and **logits outputs**.
- Image/video preprocessing (resize/crop/augment) and audio preprocessing (mono/resample/crop) are handled by GASBench. Input normalization should be done inside your model's `forward()` method.
- **Binary (real vs synthetic)** classification with `num_classes: 2` is the standard format.

For full submission requirements, see:  
👉 **[Safetensors Model Specification](./docs/Safetensors.md)** (required for competition)

> **Note**: ONNX format is no longer accepted. See [ONNX.md](./docs/ONNX.md) for legacy reference.

---

## Metrics

- **sn34_score** -- Primary competition metric. Geometric mean of normalized MCC and Brier score: $\sqrt{\text{MCC\_norm}^{1.2} \cdot \text{Brier\_norm}^{1.8}}$. Rewards both discrimination accuracy and probability calibration.
- **binary_mcc** -- Matthews Correlation Coefficient for binary real/synthetic classification (-1 to +1)
- **binary_brier** -- Brier score measuring calibration quality (0 = perfect, 0.25 = random)
- **binary_cross_entropy** -- Log-loss for predicted probabilities
- **benchmark_score** -- Overall benchmark score
- **avg_inference_time_ms** -- Mean inference time per sample
- **p95_inference_time_ms** -- 95th percentile inference time

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
    "model_path": "./my_audio_model/",
    "modality": "audio",
    "mode": "full",
    "gasstation_only": false,
    "benchmark_completed": true,
    "duration_seconds": 1054.31
  },
  "overall_score": 0.8523,
  "validation": {
    "model_path": "./my_audio_model/",
    "num_classes": 2,
    "weights_file": "model.safetensors"
  },
  "errors": [],
  "results": {
    "total_samples": 5000,
    "correct_predictions": 4261,
    "accuracy": 0.8522,
    "avg_inference_time_ms": 12.3,
    "p95_inference_time_ms": 45.6,
    "binary_mcc": 0.7045,
    "binary_brier": 0.1523,
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
