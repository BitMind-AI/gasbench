import json
import time
import hashlib
import uuid
import platform
import stat
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import DEFAULT_TARGET_SIZE
from ..dataset.config import (
    get_benchmark_size,
    discover_benchmark_datasets,
    calculate_weighted_dataset_sampling,
    build_dataset_info,
    load_holdout_datasets_from_yaml,
    apply_mode_to_datasets,
)
from ..processing.transforms import extract_target_size_from_input_specs
from ..dataset.iterator import DatasetIterator
from ._checkpoint import CheckpointError, RecorderCheckpoint
from .recording import (
    BenchmarkRunRecorder,
    build_sample_id,
    compute_metrics_from_df,
    compute_per_dataset_from_df,
    compute_generator_stats_from_df,
)
from .utils import calculate_per_source_accuracy
from .utils.inference import process_model_output
from ..logger import get_logger


def stack_uniform_batch(items: List[np.ndarray]) -> np.ndarray:
    """Stack a list of same-shaped arrays into a single batched array."""
    first = items[0]
    batch_array = np.empty((len(items),) + first.shape, dtype=first.dtype)
    for i, item in enumerate(items):
        batch_array[i] = item
    return batch_array


def run_batch_and_record(
    session,
    input_specs,
    batch_array: np.ndarray,
    batch_metadata,
    tracker: BenchmarkRunRecorder,
    batch_id: int,
    aug_pass: bool = False,
):
    """Run one batch through the model and record rows in the tracker.

    On inference failure every sample in the batch is recorded as an error
    (rather than propagating and aborting the whole dataset), so a model that
    crashes on hard samples cannot silently drop them from its score.
    """
    logger = get_logger(__name__)
    if not batch_metadata:
        return

    start = time.time()
    try:
        outputs = session.run(None, {input_specs[0].name: batch_array})
        if len(outputs[0]) != len(batch_metadata):
            raise ValueError("Model output batch size differs from input")
        predictions = [process_model_output(output) for output in outputs[0]]
    except Exception as e:
        logger.error(f"Inference failed: {e} (batch shape: {batch_array.shape})")
        for label, sample, sample_index, dataset_name, sample_seed in batch_metadata:
            tracker.add_error(
                dataset_name=dataset_name,
                sample_index=sample_index,
                sample=sample,
                error_message=f"inference-failed: {str(e)[:160]}",
                aug_pass=aug_pass,
            )
        tracker.checkpoint()
        return

    batch_inference_time = (time.time() - start) * 1000
    per_sample_time = batch_inference_time / len(batch_metadata)

    for i, (label, sample, sample_index, dataset_name, sample_seed) in enumerate(
        batch_metadata
    ):
        predicted, pred_probs = predictions[i]
        tracker.add_ok(
            dataset_name=dataset_name,
            sample_index=sample_index,
            sample=sample,
            label=label,
            predicted=predicted,
            probs=pred_probs,
            inference_time_ms=per_sample_time,
            batch_inference_time_ms=batch_inference_time,
            batch_id=batch_id,
            batch_size=len(batch_metadata),
            sample_seed=sample_seed,
            aug_pass=aug_pass,
        )

    tracker.checkpoint()


@dataclass
class BenchmarkRunConfig:
    modality: str
    mode: str
    gasstation_only: bool
    dataset_config_path: Optional[str]
    holdout_config_path: Optional[str]
    cache_dir: Optional[str]
    hf_token: Optional[str]
    batch_size: int
    augment_level: int
    crop_prob: float
    records_parquet_path: Optional[str]
    run_id: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_persist: Optional[Callable[[Path], None]] = None
    dataset_filters: Optional[List[str]] = None
    holdout_weight: float = 1.0  # (Legacy) weight multiplier for holdout datasets in benchmark_score only
    holdouts_only: bool = False  # If True, only run holdout datasets (requires holdout_config_path)
    content_category: Optional[str] = None  # Filter datasets by content_category (e.g. "faces")
    score_composition: Optional[Dict[str, float]] = None  # Target score weight share per provenance class, e.g. {"public": 0.5, "holdout": 0.3, "gasstation": 0.2}; weights all metrics incl. sn34_score
    multiclass_scoring: bool = False  # Derive sn34_score from Gorodkin multiclass MCC + multiclass Brier instead of the binary real-vs-not-real collapse. No-op for audio (2 classes).
    n_aug_per_dataset: int = 0  # Number of samples per dataset to re-evaluate with robustness augmentations (0 = disabled)
    aug_cache_dir: Optional[str] = None
    aug_cache_readonly: bool = False
    aug_weight: float = 0.2  # Weight of aug_sn34_score in blended final score (when n_aug_per_dataset > 0)


@dataclass
class SamplingSummary:
    target_samples: int
    actual_total_samples: int
    num_datasets: int
    gasstation_datasets: int
    regular_datasets: int
    gasstation_samples_per_dataset: int
    regular_samples_per_dataset: int
    dataset_breakdown: Dict[str, int]


@dataclass
class BenchmarkPlan:
    available_datasets: List
    sampling_plan: Dict[str, int]
    target_size: Tuple[int, int]
    dataset_info: Dict
    sampling_summary: SamplingSummary
    samples: Dict = field(default_factory=dict)


def build_plan(
    logger, config: BenchmarkRunConfig, input_specs
) -> Optional[BenchmarkPlan]:
    if config.holdouts_only:
        available_datasets = []
    else:
        available_datasets = discover_benchmark_datasets(
            modality=config.modality,
            mode=config.mode,
            gasstation_only=config.gasstation_only,
            yaml_path=config.dataset_config_path,
            content_category=config.content_category,
        )

    if config.holdout_config_path and not config.gasstation_only:
        try:
            holdouts = load_holdout_datasets_from_yaml(
                config.holdout_config_path,
                cache_dir=config.cache_dir
            ).get(config.modality, [])
            holdouts = apply_mode_to_datasets(holdouts, config.mode)
            available_datasets.extend(holdouts)
        except Exception as e:
            logger.error(f"Failed to load holdout {config.modality} datasets: {e}")

    if config.dataset_filters:
        original_count = len(available_datasets)
        filters_lower = [f.lower() for f in config.dataset_filters]
        available_datasets = [
            d for d in available_datasets
            if any(f in d.name.lower() for f in filters_lower)
        ]
        logger.info(f"Filtered {original_count} datasets to {len(available_datasets)} matching: {config.dataset_filters}")

    if not available_datasets:
        return None

    logger.info(
        f"Using {len(available_datasets)} {config.modality} datasets for benchmarking"
    )

    target_size = extract_target_size_from_input_specs(input_specs)
    if target_size is None:
        target_size = DEFAULT_TARGET_SIZE
        logger.info(f"Model has dynamic axes, using default target size: {target_size}")
    else:
        logger.info(f"Using fixed target size from model: {target_size}")

    target_samples = get_benchmark_size(config.modality, config.mode)
    sampling_plan = calculate_weighted_dataset_sampling(
        available_datasets, target_samples
    )
    actual_total_samples = sum(sampling_plan.values())

    gasstation_count = len(
        [d for d in available_datasets if "gasstation" in d.name.lower()]
    )
    regular_count = len(available_datasets) - gasstation_count
    gasstation_cap = sampling_plan.get(
        next(
            (d.name for d in available_datasets if "gasstation" in d.name.lower()), ""
        ),
        0,
    )
    regular_cap = sampling_plan.get(
        next(
            (d.name for d in available_datasets if "gasstation" not in d.name.lower()),
            "",
        ),
        0,
    )

    sampling_summary = SamplingSummary(
        target_samples=target_samples,
        actual_total_samples=actual_total_samples,
        num_datasets=len(available_datasets),
        gasstation_datasets=gasstation_count,
        regular_datasets=regular_count,
        gasstation_samples_per_dataset=gasstation_cap,
        regular_samples_per_dataset=regular_cap,
        dataset_breakdown={
            "real": len([d for d in available_datasets if d.media_type == "real"]),
            "synthetic": len(
                [d for d in available_datasets if d.media_type == "synthetic"]
            ),
            "semisynthetic": len(
                [d for d in available_datasets if d.media_type == "semisynthetic"]
            ),
        },
    )
    logger.info(f"Sampling configuration: {json.dumps(sampling_summary.__dict__)}")

    dataset_info = build_dataset_info(available_datasets, sampling_plan)
    return BenchmarkPlan(
        available_datasets=available_datasets,
        sampling_plan=sampling_plan,
        target_size=target_size,
        dataset_info=dataset_info,
        sampling_summary=sampling_summary,
    )


def fingerprint_files(paths, root):
    """Bind model/evaluator inputs by content, excluding transient Python bytecode."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_versions():
    versions = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    for package in (
        "gasbench",
        "numpy",
        "torch",
        "torchvision",
        "scipy",
        "pillow",
        "opencv-python-headless",
        "decord",
        "torchcodec",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def sample_files(sample):
    if "video_frames" in sample:
        return [Path(p) for p in sample["video_frames"]]
    return [
        Path(sample[f"{modality}_path"])
        for modality in ("image", "video", "audio")
        if f"{modality}_path" in sample
    ]


def sample_digest(sample):
    paths = sample_files(sample)
    if not paths or any(not p.is_file() for p in paths):
        raise CheckpointError("A selected sample is missing from the cache")
    return fingerprint_files(paths, paths[0].parent)


def file_metadata_digest(paths):
    """Identify cached file generations without reading media payloads.

    Inputs must be on a trusted filesystem that preserves modification/change
    timestamps across mounts. This detects replacement and in-place edits,
    including same-size edits with restored mtime (ctime still changes).
    It is not a content hash or protection against a malicious storage owner.
    Sandbox inputs are read-only; checkpoint batch checksums remain unchanged.
    """
    paths = sorted(paths)
    if not paths:
        raise CheckpointError("A selected sample is missing from the cache")
    entries = []
    for path in paths:
        try:
            info = path.stat()
        except OSError as exc:
            raise CheckpointError("Cannot stat a selected input") from exc
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointError("A selected input is not a regular file")
        entries.append((str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def verify_sample(sample):
    """Reject changed pending input before preprocessing; completed rows need no I/O."""
    if "file_metadata_sha256" in sample:
        if file_metadata_digest(sample_files(sample)) != sample["file_metadata_sha256"]:
            raise CheckpointError("A selected sample changed since this run started")
    elif "content_sha256" in sample:
        try:
            if sample_digest(sample) != sample["content_sha256"]:
                raise CheckpointError(
                    "A selected sample changed since this run started"
                )
        except OSError as exc:
            raise CheckpointError("Cannot read a selected sample") from exc


def use_augmentation_cache(sample, path):
    """Use only the derived artifact selected for this run, or regenerate it."""
    path = Path(path)
    if "augmentation_cache_metadata_sha256" in sample:
        expected = sample["augmentation_cache_metadata_sha256"]
        if expected is None:
            return False
        try:
            if file_metadata_digest([path]) != expected:
                raise CheckpointError("Selected augmentation cache changed or disappeared")
        except CheckpointError as exc:
            raise CheckpointError("Selected augmentation cache changed or disappeared") from exc
        return True
    if "augmentation_cache_sha256" not in sample:
        return path.is_file()
    expected = sample["augmentation_cache_sha256"]
    if expected is None:
        return False
    try:
        if not path.is_file() or fingerprint_files([path], path.parent) != expected:
            raise CheckpointError("Selected augmentation cache changed or disappeared")
    except OSError as exc:
        raise CheckpointError("Cannot read selected augmentation cache") from exc
    return True


def create_dataset_iterator(config, plan, dataset_config, *, aug_pass=False):
    """Use the iterator's frozen selection for both normal and resumed execution."""
    samples = plan.samples[dataset_config.name]["aug" if aug_pass else "base"]
    return DatasetIterator(
        dataset_config,
        max_samples=max(len(samples), 1),
        cache_dir=config.cache_dir,
        download=False,
        lazy_read=True,
        frozen_samples=samples,
    )


def create_tracker(
    config: BenchmarkRunConfig,
    plan: BenchmarkPlan,
    input_specs,
    *,
    session,
    seed,
    skip_missing=False,
    download_latest_gasstation_data=False,
) -> BenchmarkRunRecorder:
    """Freeze inputs once, then open the single recorder for this run."""
    config.run_id = config.run_id or str(uuid.uuid4())
    # run_id is an external identifier, not a path supplied by the caller.
    if Path(config.run_id).name != config.run_id or config.run_id in (".", ".."):
        raise ValueError("run_id must be a single path component")
    directory = (
        Path(config.checkpoint_dir)
        if config.checkpoint_dir
        else Path(config.cache_dir) / "runs" / config.run_id / "checkpoint"
    )
    config.checkpoint_dir = str(directory)
    saved = RecorderCheckpoint.read_manifest(directory)
    model_dir = getattr(session, "model_dir", None)
    if model_dir is None:
        raise ValueError(
            "Benchmark inference sessions must expose model_dir for checkpoint identity"
        )
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise CheckpointError("Model directory is unavailable")
    evaluator_dir = Path(__file__).resolve().parents[1]
    settings = {
        item.name: getattr(config, item.name)
        for item in fields(config)
        if item.name
        not in (
            "hf_token",
            "records_parquet_path",
            "checkpoint_dir",
            "checkpoint_persist",
        )
    }
    context = {
        "input_identity": "file-metadata-v1",
        "settings": settings,
        "seed": seed,
        "runtime": runtime_versions(),
        "skip_missing": skip_missing,
        "model_sha256": fingerprint_files(model_dir.rglob("*"), model_dir),
        "evaluator_sha256": fingerprint_files(
            evaluator_dir.rglob("*.py"), evaluator_dir
        ),
        "input_specs": [
            {"name": spec.name, "shape": list(spec.shape), "type": spec.type}
            for spec in input_specs
        ],
        "preprocessing": session.get_preprocessing_config()
        if hasattr(session, "get_preprocessing_config")
        else {},
        "datasets": [asdict(dataset) for dataset in plan.available_datasets],
        "sampling_plan": plan.sampling_plan,
        "target_size": list(plan.target_size),
    }
    # Normalize tuples/paths consistently with the recorder's JSON manifest.
    context = json.loads(json.dumps(context))
    if saved is not None:
        previous = saved.get("benchmark", {})
        if {k: v for k, v in previous.items() if k != "samples"} != context:
            raise CheckpointError(
                "Checkpoint configuration or model differs from this run"
            )
        if "samples" not in previous:
            raise CheckpointError("Checkpoint has no frozen sample selection")
        plan.samples = previous["samples"]
    else:
        logger = get_logger(__name__)
        started = time.monotonic()
        logger.info("Freezing sample selection using file metadata (no media payload scan)")
        for index, dataset in enumerate(plan.available_datasets, 1):
            selections = {}
            for pass_name, cap in (
                ("base", plan.sampling_plan[dataset.name]),
                ("aug", config.n_aug_per_dataset),
            ):
                if cap <= 0:
                    selections[pass_name] = []
                    continue
                iterator = DatasetIterator(
                    dataset,
                    max_samples=cap,
                    cache_dir=config.cache_dir,
                    download=not skip_missing
                    and (
                        download_latest_gasstation_data
                        if "gasstation" in dataset.name.lower()
                        else True
                    ),
                    hf_token=config.hf_token,
                    seed=seed,
                    metadata_only=True,
                )
                selections[pass_name] = list(iterator)
                for sample in selections[pass_name]:
                    sample["file_metadata_sha256"] = file_metadata_digest(sample_files(sample))
                    if pass_name == "aug" and config.aug_cache_dir:
                        from .aug_cache import img_aug_cache_path, vid_aug_cache_path

                        cache_path = (
                            img_aug_cache_path
                            if config.modality == "image"
                            else vid_aug_cache_path
                        )
                        path = Path(
                            cache_path(
                                config.aug_cache_dir,
                                build_sample_id(sample),
                                plan.target_size,
                            )
                        )
                        # Newly generated cache files are outputs of this attempt;
                        # only artifacts present in the frozen plan may be inputs.
                        sample["augmentation_cache_metadata_sha256"] = (
                            file_metadata_digest([path])
                            if path.is_file()
                            else None
                        )
            plan.samples[dataset.name] = selections
            if index == 1 or index % 10 == 0 or index == len(plan.available_datasets):
                logger.info(
                    "Frozen dataset %s/%s (%s) in %.1fs",
                    index, len(plan.available_datasets), dataset.name, time.monotonic() - started,
                )
    context["samples"] = plan.samples
    tracker = BenchmarkRunRecorder(
        run_id=config.run_id,
        mode=config.mode,
        modality=config.modality,
        target_size=plan.target_size,
        model_input_name=input_specs[0].name if input_specs else None,
        augment_level=config.augment_level or 0,
        crop_prob=config.crop_prob or 0.0,
        checkpoint_dir=directory,
        checkpoint_context=context,
        checkpoint_persist=config.checkpoint_persist,
    )
    get_logger(__name__).info(
        f"Run {config.run_id}: restored {tracker.count} rows from {directory}"
    )
    return tracker


def finalize_run(
    *,
    config: BenchmarkRunConfig,
    plan: BenchmarkPlan,
    tracker: BenchmarkRunRecorder,
    benchmark_results: Dict,
    results_key: str,
    extra_fields: Optional[Dict] = None,
):
    logger = get_logger(__name__)
    tracker.checkpoint()
    df = tracker.to_dataframe()
    metric_pack = compute_metrics_from_df(
        df,
        holdout_weight=config.holdout_weight,
        score_composition=config.score_composition,
        multiclass_scoring=config.multiclass_scoring,
    )

    # Blend aug_sn34_score into sn34_score when an aug pass was run.
    # base_sn34_score is preserved for transparency.
    if (
        config.n_aug_per_dataset > 0
        and config.aug_weight > 0
        and "aug_sn34_score" in metric_pack
    ):
        base_sn34 = metric_pack["sn34_score"]
        aug_sn34 = metric_pack["aug_sn34_score"]
        blended = (1.0 - config.aug_weight) * base_sn34 + config.aug_weight * aug_sn34
        metric_pack["base_sn34_score"] = base_sn34
        metric_pack["sn34_score"] = blended
        logger.info(
            f"sn34_score blended: base={base_sn34:.4f} aug={aug_sn34:.4f} "
            f"weight={config.aug_weight} → {blended:.4f}"
        )

    per_dataset_results = compute_per_dataset_from_df(df)
    per_source_accuracy = calculate_per_source_accuracy(
        plan.available_datasets, per_dataset_results
    )
    generator_stats = compute_generator_stats_from_df(df)

    parquet_path = None
    if config.records_parquet_path:
        try:
            parquet_path = tracker.write_parquet(config.records_parquet_path)
        except Exception as e:
            logger.warning(
                f"Failed to write parquet to {config.records_parquet_path}: {e}"
            )

    results = {
        **metric_pack,
        "total_samples": int((df["status"] == "ok").sum() if not df.empty else 0),
        "correct_predictions": (
            int(df["correct"].sum()) if not df.empty and "correct" in df else 0
        ),
        "per_source_accuracy": per_source_accuracy,
        "per_dataset_results": per_dataset_results,
        "dataset_info": plan.dataset_info,
        "records_count": int(len(df)),
        "sampling_summary": plan.sampling_summary.__dict__,
    }
    if generator_stats:
        results["generator_stats"] = generator_stats
    if parquet_path:
        results["parquet_path"] = parquet_path
        logger.info(f"Benchmark run recorded at: {parquet_path}")
    if extra_fields:
        results.update(extra_fields)

    benchmark_results[results_key] = results

    return df
