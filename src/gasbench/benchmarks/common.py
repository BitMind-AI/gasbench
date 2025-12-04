import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from .recording import (
    BenchmarkRunRecorder,
    compute_metrics_from_df,
    compute_per_dataset_from_df,
    compute_generator_stats_from_df,
    write_cache_policy_from_df,
)
from .utils import calculate_per_source_accuracy
from ..logger import get_logger


@dataclass
class BenchmarkRunConfig:
    modality: str
    mode: str
    gasstation_only: bool
    dataset_config_path: Optional[str]
    holdout_config_path: Optional[str]
    cache_dir: Optional[str]
    cache_policy_path: Optional[str]
    hf_token: Optional[str]
    batch_size: int
    augment_level: int
    crop_prob: float
    records_parquet_path: Optional[str]
    run_id: Optional[str] = None


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


def build_plan(
    logger, config: BenchmarkRunConfig, input_specs
) -> Optional[BenchmarkPlan]:
    available_datasets = discover_benchmark_datasets(
        modality=config.modality,
        mode=config.mode,
        gasstation_only=config.gasstation_only,
        yaml_path=config.dataset_config_path,
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


def create_tracker(
    config: BenchmarkRunConfig, plan: BenchmarkPlan, input_specs
) -> BenchmarkRunRecorder:
    return BenchmarkRunRecorder(
        run_id=config.run_id,
        mode=config.mode,
        modality=config.modality,
        target_size=plan.target_size,
        model_input_name=input_specs[0].name if input_specs else None,
        augment_level=config.augment_level or 0,
        crop_prob=config.crop_prob or 0.0,
    )


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
    df = tracker.to_dataframe()
    metric_pack = compute_metrics_from_df(df)
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

    if config.cache_policy_path:
        try:
            priorities = write_cache_policy_from_df(df, config.cache_policy_path)
            logger.info(
                f"Updated cache policy with {len(priorities)} generator priorities (fool_rate)"
            )
        except Exception as e:
            logger.warning(
                f"Failed to update cache policy at {config.cache_policy_path}: {e}"
            )
    return df
