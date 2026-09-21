"""Recovery uses real recorder rows, summaries, scoring, and parquet output."""

import pandas as pd
import pytest

from gasbench.benchmarks.recording import (
    BenchmarkRunRecorder,
    compute_metrics_from_df,
    compute_per_dataset_from_df,
)
from gasbench.benchmarks._checkpoint import CheckpointError


@pytest.fixture
def context():
    return {
        "model_hash": "model",
        "benchmark_version": "test",
        "seed": 42,
        "sample_plan": ["dataset@rev/a", "dataset@rev/b"],
        "score_composition": {"public": 1.0},
    }


def sample(name):
    return {
        "source_kind": "hf",
        "dataset_path": "repo",
        "source_file": name,
        "hf_resolved_revision": "revision",
        "media_type": "real",
    }


def add_prediction(
    recorder, name, *, index=1, dataset="dataset", aug_pass=False, label=0
):
    recorder.add_ok(
        dataset_name=dataset,
        sample_index=index,
        sample=sample(name),
        label=label,
        predicted=label,
        probs=[0.8, 0.1, 0.1] if label == 0 else [0.1, 0.8, 0.1],
        inference_time_ms=2.0,
        batch_inference_time_ms=2.0,
        batch_id=1,
        batch_size=1,
        sample_seed=42 + index,
        aug_pass=aug_pass,
    )


def test_resumed_recorder_uses_existing_metrics_summaries_and_parquet(
    tmp_path, context, monkeypatch
):
    monkeypatch.setattr("gasbench.benchmarks.recording.time.time", lambda: 100)
    original = BenchmarkRunRecorder(run_id="run", modality="image")
    interrupted = BenchmarkRunRecorder(
        run_id="run",
        modality="image",
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_context=context,
    )
    for recorder in (original, interrupted):
        add_prediction(recorder, "a")
        add_prediction(recorder, "a", aug_pass=True)
        recorder.add_skip(
            dataset_name="dataset", sample_index=3, sample=sample("c"), reason="decode"
        )
        recorder.add_error(
            dataset_name="dataset",
            sample_index=4,
            sample=sample("d"),
            error_message="inference",
        )
    assert interrupted.checkpoint() == 4
    # Later wall-clock time must not create a second run/start timestamp.
    monkeypatch.setattr("gasbench.benchmarks.recording.time.time", lambda: 200)
    resumed = BenchmarkRunRecorder(
        run_id="run",
        modality="image",
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_context=context,
    )
    assert resumed.run_started_at == original.run_started_at
    for recorder in (original, resumed):
        add_prediction(recorder, "b", index=2, label=1)
    assert resumed.checkpoint() == 1
    assert resumed.checkpoint() == 0
    pd.testing.assert_frame_equal(original.to_dataframe(), resumed.to_dataframe())
    assert resumed.get_dataset_summary("dataset", True) == original.get_dataset_summary(
        "dataset", True
    )
    assert compute_metrics_from_df(resumed.to_dataframe()) == compute_metrics_from_df(
        original.to_dataframe()
    )
    assert compute_per_dataset_from_df(
        resumed.to_dataframe()
    ) == compute_per_dataset_from_df(original.to_dataframe())
    resumed.write_parquet(str(tmp_path / "resumed.parquet"))
    original.write_parquet(str(tmp_path / "original.parquet"))
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "resumed.parquet"),
        pd.read_parquet(tmp_path / "original.parquet"),
    )


def test_existing_prediction_fields_distinguish_datasets_indices_and_passes(
    tmp_path, context
):
    recorder = BenchmarkRunRecorder(
        run_id="run", checkpoint_dir=tmp_path, checkpoint_context=context
    )
    variants = [dict(), {"dataset": "other"}, {"index": 2}, {"aug_pass": True}]
    for variant in variants:
        add_prediction(recorder, "same", **variant)
    assert not recorder.is_checkpointed(
        dataset_name="dataset", sample_index=1, sample=sample("same")
    )
    assert recorder.checkpoint() == 4
    resumed = BenchmarkRunRecorder(
        run_id="run", checkpoint_dir=tmp_path, checkpoint_context=context
    )
    for variant in variants:
        assert resumed.is_checkpointed(
            dataset_name=variant.get("dataset", "dataset"),
            sample_index=variant.get("index", 1),
            sample=sample("same"),
            aug_pass=variant.get("aug_pass", False),
        )
    assert not resumed.is_checkpointed(
        dataset_name="dataset", sample_index=3, sample=sample("same")
    )


def test_restore_uses_same_counters_for_augmented_skips_and_errors(tmp_path, context):
    recorder = BenchmarkRunRecorder(
        run_id="run", checkpoint_dir=tmp_path, checkpoint_context=context
    )
    add_prediction(recorder, "a")
    recorder.add_error(
        dataset_name="dataset",
        sample_index=1,
        sample=sample("a"),
        error_message="aug-failed",
        aug_pass=True,
    )
    recorder.add_skip(
        dataset_name="dataset",
        sample_index=2,
        sample=sample("b"),
        reason="aug-decode",
        aug_pass=True,
    )
    recorder.checkpoint()
    resumed = BenchmarkRunRecorder(
        run_id="run", checkpoint_dir=tmp_path, checkpoint_context=context
    )
    assert resumed.get_dataset_summary("dataset", True) == {
        "accuracy": 1.0,
        "correct": 1,
        "total": 1,
        "skipped": 0,
    }
    assert resumed.rows == recorder.rows
    assert resumed.is_checkpointed(
        dataset_name="dataset", sample_index=1, sample=sample("a"), aug_pass=True
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "other"},
        {"modality": "video"},
        {"target_size": (48, 48)},
        {"crop_prob": 0.5},
    ],
)
def test_recorder_identity_cannot_drift_on_resume(tmp_path, context, changes):
    kwargs = dict(
        run_id="run",
        modality="image",
        checkpoint_dir=tmp_path,
        checkpoint_context=context,
    )
    recorder = BenchmarkRunRecorder(**kwargs)
    add_prediction(recorder, "a")
    recorder.checkpoint()
    with pytest.raises(CheckpointError, match="manifest differs"):
        BenchmarkRunRecorder(**{**kwargs, **changes})


def test_checkpoint_requires_explicit_run_and_benchmark_context(tmp_path, context):
    with pytest.raises(ValueError, match="requires run_id"):
        BenchmarkRunRecorder(checkpoint_dir=tmp_path, checkpoint_context=context)
    with pytest.raises(ValueError, match="requires run_id"):
        BenchmarkRunRecorder(run_id="run", checkpoint_dir=tmp_path)


def test_plain_recorder_still_records_without_checkpoint_configuration():
    recorder = BenchmarkRunRecorder()
    add_prediction(recorder, "a")
    assert recorder.count == 1
    assert recorder.checkpoint() == 0
    assert recorder.get_dataset_summary("dataset")["total"] == 1
