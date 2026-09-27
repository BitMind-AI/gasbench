"""Shared base/augmentation execution and completion policy for every modality."""

from importlib import import_module

import numpy as np

from ..logger import get_logger
from .common import (
    build_plan,
    create_dataset_iterator,
    create_tracker,
    finalize_run,
    run_batch_and_record,
    stack_uniform_batch,
)
from .errors import BenchmarkError
from .recording import build_sample_id, log_dataset_summary

logger = get_logger(__name__)


def _prediction_identity(dataset, index, sample, augmented):
    return (dataset, index, build_sample_id(sample), augmented)


def validate_completed_plan(plan, tracker):
    expected = {
        _prediction_identity(dataset, index, sample, name == "aug")
        for dataset, passes in plan.samples.items()
        for name, samples in passes.items()
        for index, sample in enumerate(samples, 1)
    }
    actual = {
        (row["dataset_name"], row["iteration_index"], row["sample_id"], row["aug_pass"])
        for row in tracker.rows
    }
    if expected != actual or len(actual) != len(tracker.rows):
        raise BenchmarkError("Recorded outcomes do not match the frozen sample plan")
    if any(row["status"] == "error" for row in tracker.rows):
        raise BenchmarkError("Inference failed; this run cannot be scored")
    for pass_name in ("base", "aug"):
        selected = pass_name == "base" or any(
            passes[pass_name] for passes in plan.samples.values()
        )
        if selected and not any(
            row["status"] == "ok" and row["aug_pass"] == (pass_name == "aug")
            for row in tracker.rows
        ):
            raise BenchmarkError(f"No successful predictions in the {pass_name} pass")
    if not expected:
        raise BenchmarkError("No cached samples selected for this run")


async def run_modality_benchmark(
    session,
    input_specs,
    benchmark_results,
    *,
    config,
    seed=None,
    skip_missing=False,
    download_latest_gasstation_data=False,
):
    seed = 42 if seed is None else seed
    module = import_module(f".{config.modality}_bench", __package__)
    pipeline_type = module.PIPELINE
    preparation = module.preprocessing_options(session, input_specs)
    plan = build_plan(logger, config, input_specs)
    if plan is None:
        raise BenchmarkError("No benchmark datasets configured or selected")
    tracker = create_tracker(
        config,
        plan,
        input_specs,
        session=session,
        seed=seed,
        skip_missing=skip_missing,
        download_latest_gasstation_data=download_latest_gasstation_data,
    )
    benchmark_results["run_id"] = config.run_id
    benchmark_results["checkpoint_dir"] = config.checkpoint_dir
    if not any(passes["base"] for passes in plan.samples.values()):
        raise BenchmarkError("No cached base samples selected for this run")
    if any(row["status"] == "error" for row in tracker.rows):
        raise BenchmarkError(
            "Checkpoint contains failed inference; start a new run after fixing the model"
        )
    for augmented in [False, True] if config.n_aug_per_dataset else [False]:
        for dataset in plan.available_datasets:
            iterator = create_dataset_iterator(
                config, plan, dataset, aug_pass=augmented
            )
            logger.info(
                "%s: %s",
                "Robustness pass" if augmented else "Processing dataset",
                dataset.name,
            )
            with pipeline_type(
                iterator,
                target_size=plan.target_size,
                batch_size=config.batch_size,
                seed=seed,
                augment_level=config.augment_level,
                crop_prob=config.crop_prob,
                robustness_pass=augmented,
                aug_cache_dir=config.aug_cache_dir,
                aug_cache_readonly=config.aug_cache_readonly,
                tracker=tracker,
                **preparation,
            ) as pipeline:
                for batch_id, batch in enumerate(pipeline, 1):
                    ready = []
                    for item in batch:
                        if "skip_reason" in item:
                            tracker.add_skip(
                                dataset_name=item["dataset_name"],
                                sample_index=item["sample_index"],
                                sample=item["sample"],
                                reason=item["skip_reason"],
                                aug_pass=augmented,
                            )
                        else:
                            ready.append(item)
                    if ready:
                        arrays = [np.asarray(item["data"]) for item in ready]
                        metadata = [
                            (
                                item["label"],
                                item["sample"],
                                item["sample_index"],
                                item["dataset_name"],
                                item["sample_seed"],
                            )
                            for item in ready
                        ]
                        run_batch_and_record(
                            session,
                            input_specs,
                            stack_uniform_batch(arrays),
                            metadata,
                            tracker,
                            batch_id,
                            augmented,
                        )
                    else:
                        tracker.checkpoint()
            log_dataset_summary(logger, tracker, dataset.name, include_skipped=True)
    tracker.checkpoint()
    validate_completed_plan(plan, tracker)
    return finalize_run(
        config=config,
        plan=plan,
        tracker=tracker,
        benchmark_results=benchmark_results,
        results_key=f"{config.modality}_results",
    )
