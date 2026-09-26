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
  Binary and multiclass MCC, Brier and cross-entropy calibration metrics,
  inference times, robustness, and per-dataset breakdowns

For model submission requirements, see the  
👉 **[Safetensors Model Specification](./docs/Safetensors.md)** (required for competition)

To learn how to submit your model to **Bittensor Subnet 34** — including the entrance exam, full benchmark, and scoring pipeline — see the  
👉 **[Discriminative Mining Guide](https://github.com/BitMind-AI/bitmind-subnet/blob/main/docs/Discriminative-Mining.md)**

---

## Installation

To run benchmarks, install the `gpu` extra:

```bash
cd gasbench
pip install -e '.[gpu]'
```

The base install is intentionally minimal — just the dataset registry
(`gasbench.dataset.config` and the bundled YAML configs) with no
PyTorch. Use it when you only need dataset definitions
(e.g. bitmind-subnet's validator cache):

```bash
pip install -e .
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

# Add an audio robustness pass on 100 samples per dataset
gasbench run --audio-model ./my_audio_model/ --n-aug-per-dataset 100 --aug-weight 0.2

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

The optional audio robustness pass applies an 8 kHz downsample/upsample round
trip, -6 dB gain, and white noise at 30 dB signal-to-noise ratio after normal
mono/16 kHz/six-second preprocessing. It supports both raw audio and cached
preprocessed tensors. The base pass stays unchanged; augmented predictions use
the same scoring, sample pairing, and checkpoint path as image/video robustness
passes. Use `--aug-cache-dir` to reuse versioned augmented arrays and
`--aug-cache-readonly` to prevent writes (cache misses are computed in memory).
Audio cache entries include the sample seed so different seeds use different noise.

Every benchmark run checkpoints through `BenchmarkRunRecorder` after each inference
batch. Checkpoints default to `<cache-dir>/runs/<run-id>/checkpoint`; use
`--checkpoint-dir` to choose another location. To resume, rerun the same command
with the same `--run-id` and storage directory. Completed predictions are restored
before decoding, and scores and parquet output are rebuilt through the normal
recorder path. A new run ID starts a new evaluation.

The checkpoint freezes sample selection (including the robustness pass), source
paths and file-generation metadata (size, nanosecond modification/change times),
model/evaluator content hashes, seed (42 when omitted), and scoring settings.
Startup does not read media or augmentation payloads. Pending inputs are checked
against their frozen metadata before processing; changed inputs fail closed.
Input storage must be trusted and preserve these timestamps across mounts, as
Modal Volumes do. Metadata checks do not protect against a storage owner who can
forge file timestamps. Model setup and uncommitted work may repeat after interruption.

Checkpoint storage must survive the process or container being replaced. The
Python API accepts `checkpoint_persist(directory)` for filesystems requiring an
explicit remote commit. One coordinator must own each run directory; automatic
container replacement and distributed ownership are the caller's responsibility.

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
- Output heads are modality-specific: image uses 3 classes, video uses 4, and audio uses 2. Class order is part of the model contract.

For full submission requirements, see:  
👉 **[Safetensors Model Specification](./docs/Safetensors.md)** (required for competition)

For the experimental visual taxonomy and exact scoring definitions, see:
👉 **[Classification Taxonomy and Scoring](./docs/Classification-and-Scoring.md)**

---

## Metrics

- **sn34_score** -- Primary score selected by the benchmark configuration. It combines normalized MCC and Brier calibration performance.
- **gorodkin_mcc / multiclass_brier / multiclass_sn34_score** -- Multiclass metrics that reward correct provenance classification.
- **binary_mcc / binary_brier / binary_cross_entropy / binary_sn34_score** -- Compatibility metrics after collapsing all non-real classes into synthetic.
- **base_sn34_score / aug_sn34_score** -- Normal and augmented-pass scores when robustness evaluation is enabled.
- **benchmark_score** -- Weighted classification accuracy.
- **avg_inference_time_ms / p95_inference_time_ms** -- Mean and 95th-percentile inference latency.

See [Classification Taxonomy and Scoring](./docs/Classification-and-Scoring.md) for class definitions, formulas, dataset weighting, and robustness blending.

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
