# Running benchmarks

For installation and a first run, see the [README](../README.md). Use
`gasbench run --help` for the complete option list.

## Select datasets

Runs use the bundled dataset configuration by default. These options narrow or
replace that selection:

| Option | Purpose |
| --- | --- |
| `--debug`, `--small`, `--full` | Choose an evaluation size; `full` is the default. |
| `--datasets PATTERN ...` | Match dataset names by case-insensitive substring. |
| `--dataset-config PATH` | Use a custom dataset YAML configuration. |
| `--gasstation-only` | Evaluate only GAS-Station datasets. |
| `--holdout-config PATH` | Include private holdout datasets. |
| `--holdouts-only` | Evaluate only datasets from `--holdout-config`. |
| `--skip-missing` | Use cached datasets without downloading missing data. |

For example:

```bash
gasbench run --image-model ./my_model --full \
  --datasets pica-100k --cache-dir ./cache
```

Filtering keeps each selected dataset's sample limit from the unfiltered run.
For class definitions and dataset weighting, see [classification and
scoring](Classification-and-Scoring.md).

## Store inputs and results

Set `--cache-dir` to a writable directory with space for the datasets; the default
is `/.cache/gasbench`. Reuse it across runs to reuse downloaded data.

The CLI writes a JSON report, prediction parquet, and text summary to a run
subdirectory under `results/`. Set `--results-dir` to change that location and
`--run-name` to name the subdirectory.

The Python API returns a results dictionary. Call `save_results_to_json()` to
save it, and pass `records_parquet_path` to `run_benchmark()` to export predictions.

## Resume an interrupted run

Give the run an ID and keep its storage available:

```bash
gasbench run --image-model ./my_model --full \
  --cache-dir ./cache --run-id detector-eval
```

Rerun the same command to resume. Completed inference batches are restored;
unfinished work runs again. Use a new run ID for a fresh evaluation.

Checkpoints default to `<cache-dir>/runs/<run-id>/checkpoint`. Use
`--checkpoint-dir` to place them elsewhere. Resume with the same model, benchmark
settings, input paths, GASBench build, and runtime; incompatible checkpoints
are rejected.

For managed workers:

- Keep datasets and checkpoints on storage that survives worker replacement.
- Use a trusted filesystem that preserves input file metadata across mounts;
  pending inputs are checked for changes using file size and timestamps.
- Pass `checkpoint_persist(directory)` to the Python API when the filesystem
  requires an explicit commit to persist writes remotely.
- Let one worker own a run directory at a time. The caller handles worker restarts.

## Evaluate robustness

Add a second pass over a subset of each dataset with `--n-aug-per-dataset`:

```bash
gasbench run --audio-model ./my_model --cache-dir ./cache \
  --n-aug-per-dataset 100 --aug-weight 0.2
```

The suite depends on the modality:

| Modality | Transformations |
| --- | --- |
| Image | Downscale/upsample and repeated JPEG/WebP compression. |
| Video | Downscale and H.264 recompression, with a JPEG fallback if video encoding is unavailable. |
| Audio | 8 kHz downsample/upsample, −6 dB gain, and white noise at 30 dB SNR. |

Robustness evaluation is disabled by default. `--aug-weight` controls the
augmented score's contribution to `sn34_score`; see [robustness
scoring](Classification-and-Scoring.md#dataset-composition-and-robustness).

Use `--aug-cache-dir` to reuse augmented inputs across evaluations. Add
`--aug-cache-readonly` to load existing entries without writing new ones; cache
misses are computed in memory. Audio augmentation supports raw audio and
preprocessed tensors, with seeded noise for reproducible runs.
