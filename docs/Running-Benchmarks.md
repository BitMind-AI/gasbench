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

## Video sampling

PyAV reads frames sequentially from the beginning, stopping after the requested
prefix. Frame-rate sampling uses an index stride based on average FPS; short
clips repeat the last selected frame. No whole-file copy or frame-count scan is
required. Files with missing frame counts can be evaluated; damage beyond the
requested prefix is not checked.

## Store inputs and results

Set `--cache-dir` to a writable directory with space for the datasets; the default
is `/.cache/gasbench`. Reuse it across runs to reuse downloaded data.

For repeated audio runs, `gasbench preprocess --dataset NAME --cache-dir ./cache`
converts cached audio to mono 16 kHz, six-second tensors. Conversion can be rerun
and retains source files in each dataset cache's `originals/` directory.

The CLI writes a JSON report, prediction parquet, and text summary to a run
subdirectory under `results/`. Set `--results-dir` to change that location and
`--run-name` to name the subdirectory.

The Python API returns a results dictionary. Call `save_results_to_json()` to
save it, and pass `records_parquet_path` to `run_benchmark()` to export predictions.

## Diagnose runtime

The JSON report's `performance` field (Python: `results["metrics"]["performance"]`)
contains stage totals, call counts, and maximum durations for the current attempt.
Startup, base inference, augmentation, model loading, and finalization are separate
groups. Per-dataset timing summaries also appear in the logs, including on errors.
Resumed predictions are counted as restored work; their original timings are not
added to the new attempt.

Use `input_wait` to identify waits for prepared batches, and compare `checkpoint`
with its `checkpoint_encode`, `checkpoint_local_write` (write, rename, fsync), and
`checkpoint_persist` (filesystem commit callback) components. Source reads, decode,
transforms, and augmentation-cache hits/misses are measured in preparation workers.
Filesystem writes also report creation, writing, file sync, rename, and directory sync separately.
Decode includes decoder-internal I/O, including video-file and frame-image reads;
preprocessed audio loading includes tensor deserialization.

Durations are inclusive and can overlap: **do not sum worker stages into wall
time**, or add checkpoint components to the checkpoint total. Pass `wall` is the
sum of dataset execution and final pass checkpoint times. `inference` includes transfers, model execution,
and output conversion; it is not a GPU-kernel-only measurement.

## Resume an interrupted run

Give the run an ID and keep its storage available:

```bash
gasbench run --image-model ./my_model --full \
  --cache-dir ./cache --run-id detector-eval
```

Rerun the same command to resume from the last durable checkpoint. Checkpoints group
inference batches, flushing after 60 seconds or 1,024 pending records, checked between
batches, and at pass boundaries, completion, or handled failures. Abrupt termination
replays the uncommitted tail; a long batch can exceed the interval. Use a new run ID
for a fresh evaluation.

Checkpoints default to `<cache-dir>/runs/<run-id>/checkpoint`. Use
`--checkpoint-dir` to place them elsewhere. Resume with the same model, benchmark
settings, input paths, GASBench build, and runtime; incompatible checkpoints
are rejected.

For managed workers:

- Keep datasets and checkpoints on storage that survives worker replacement.
- Use a trusted filesystem that preserves input file metadata across mounts;
  pending inputs are checked for changes using file size and timestamps.
- Pass `checkpoint_persist(directory)` to the Python API when the filesystem
  requires an explicit commit to persist writes remotely. It must persist all
  checkpoint files before returning; a failed commit aborts the writer.
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
scoring](Classification-and-Scoring.md#robustness-scoring).

Use `--aug-cache-dir` to reuse augmented inputs across evaluations. Add
`--aug-cache-readonly` to load existing entries without writing new ones; cache
misses are computed in memory. Audio augmentation supports raw audio and
preprocessed tensors, with seeded noise for reproducible runs.
