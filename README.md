# GASBench

GASBench evaluates AI-generated content detectors on **image, video, and audio**
datasets for [Bittensor Subnet 34](https://github.com/BitMind-AI/bitmind-subnet).
It handles dataset downloads, media preprocessing, inference, and scoring.

## Install

Requires Python 3.10 or newer. From a checkout of this repository:

```bash
pip install -e '.[gpu]'
```

For dataset definitions alone, use `pip install -e .`; the base package does not
include the benchmark dependencies.

## Run a benchmark

Prepare a model directory containing `model_config.yaml`, `model.py`, and
safetensors weights. See the [model specification](docs/Safetensors.md) for
configuration examples and the input/output contract.

Start with a debug run:

```bash
gasbench run --image-model ./my_model --debug --cache-dir ./cache
```

Use `--video-model` or `--audio-model` for other modalities. Replace `--debug`
with `--full` for a full evaluation.

The CLI saves each run under `results/` with:

- `results.json`: scores, timing, and dataset breakdowns.
- `records.parquet`: per-sample predictions.
- `summary.txt`: a readable report.

See [scoring](docs/Classification-and-Scoring.md) to interpret the results, or
`gasbench run --help` for all options. The [running guide](docs/Running-Benchmarks.md)
covers dataset selection, caching, resuming runs, and robustness evaluation.

## Python API

```python
import asyncio
from gasbench import run_benchmark, print_benchmark_summary, save_results_to_json

results = asyncio.run(run_benchmark(
    model_path="./my_model",
    modality="image",  # "image", "video", or "audio"
    mode="debug",
    cache_dir="./cache",
))

print_benchmark_summary(results)
save_results_to_json(results, output_dir="./results")
```

## Documentation

- [Model specification](docs/Safetensors.md): package a model for evaluation.
- [Classification and scoring](docs/Classification-and-Scoring.md): class labels,
  metrics, and score weighting.
- [Running benchmarks](docs/Running-Benchmarks.md): configure and manage evaluations.
- [Discriminative mining guide](https://github.com/BitMind-AI/bitmind-subnet/blob/main/docs/Discriminative-Mining.md):
  submit a model to Subnet 34.
- [Releases](https://github.com/BitMind-AI/gasbench/releases): changes between versions.
