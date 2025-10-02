# GASBench

A benchmark evaluation package for ML models, providing tools for evaluating AI-generated content detection models on image and video datasets.

## Overview

This package provides a self-contained benchmark evaluation system for testing models on diverse datasets:

- **Image & Video Benchmarks** - Test models on various datasets for AI-generated content detection
- **Data Processing** - Dataset downloads, caching,  preprocessing, amd augmenting
- **Comprehensive Metrics** - Accuracy, MCC, inference times, and per-dataset breakdowns

## Installation

### From Source

```bash
cd gasbench
pip install -e .
```

### GPU Support (Optional)

For GPU acceleration:

```bash
pip install -e .[gpu]
```

### Development

For development with additional tools:

```bash
pip install -e .[dev]
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
```

### Python API

```python
import asyncio
from gasbench import run_benchmark, print_benchmark_summary

async def evaluate_model():
    results = await run_benchmark(
        model_path="path/to/model.onnx",
        modality="image",  # or "video"
        debug_mode=False,
        gasstation_only=False
    )
    
    print_benchmark_summary(results)
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

## Configuration

### Debug Mode

Use `debug_mode=True` for faster testing with smaller datasets during development.

### Gasstation Only Mode

Use `gasstation_only=True` to evaluate only on gasstation datasets for faster, focused evaluation.

### Cache Directory

By default, datasets are cached at `/tmp/benchmark_data/`. Specify a custom location with `cache_dir` parameter or `--cache-dir` flag.

## Model Requirements

Your ONNX model should:
- Accept inputs of shape `(1, 3, 224, 224)` for images or `(1, 16, 3, 224, 224)` for videos
- Use `uint8` data type (0-255 range)
- Output one of:
  - **Binary**: 2 outputs `[human_prob, ai_prob]`
  - **3-class**: 3 outputs `[real_prob, synthetic_prob, semisynthetic_prob]`
  - **Single sigmoid**: 1 output where >0.5 indicates AI-generated

## Datasets

### Image Datasets
- **Real**: CelebA-HQ, FFHQ, MS-COCO, OpenImages
- **Synthetic**: JourneyDB, GenImage, Midjourney outputs
- **Semisynthetic**: Face-swap and modified content
- **Gasstation**: Primary benchmark dataset

### Video Datasets
- **Real**: PE-Video, ImageNet-VidVRD
- **Synthetic**: Veo2/3 preferences, AI-generated videos
- **Semisynthetic**: Modified/edited video content
- **Gasstation**: Primary benchmark dataset

## Metrics

### Classification Metrics
- **Accuracy** - Overall classification accuracy
- **Binary MCC** - Matthews Correlation Coefficient for binary classification
- **Multiclass MCC** - MCC for 3-class classification

### Performance Metrics
- **Avg Inference Time** - Mean inference time per sample (ms)
- **P95 Inference Time** - 95th percentile inference time (ms)

### Detailed Breakdowns
- **Per-dataset Results** - Accuracy by individual dataset
- **Per-source Accuracy** - Results by media type (real/synthetic/semisynthetic)
- **Generator Stats** - Performance against specific AI generators (gasstation only)

## Troubleshooting

### Common Issues

**Import Errors**: Install dependencies via `pip install -e .`

**ONNX Runtime**:
- CPU: `onnxruntime>=1.22.0`
- GPU: `onnxruntime-gpu>=1.22.0` with CUDA available

**Memory Issues**: Use debug mode or ensure sufficient disk space for caching

**Network Issues**: Datasets download from HuggingFace Hub - ensure connectivity

**Model Format**: Verify input shapes and uint8 data type (0-255)

### Performance Tips

1. **Use GPU** - Install `onnxruntime-gpu` for faster inference
2. **Cache Datasets** - Let datasets cache locally to avoid re-downloads
3. **Debug Mode** - Use during development for faster iteration
4. **Gasstation Only** - Focus on primary datasets for quicker evaluation

## License

This code is part of the larger benchmark evaluation system. Please refer to the main project license for usage terms.
