"""Exercise interruption recovery through real iterators, pipelines and scoring."""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gasbench.benchmarks import common
from gasbench.benchmarks._checkpoint import CheckpointError, RecorderCheckpoint
from gasbench.dataset.config import BenchmarkDatasetConfig


class Interrupted(BaseException):
    pass


class Session:
    def __init__(self, model_dir, stop_after=None, error=False):
        self.model_dir = model_dir
        self.stop_after = stop_after
        self.calls = []
        self.error = error

    def run(self, _, inputs):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise Interrupted()
        values = next(iter(inputs.values()))
        self.calls.extend(int(value.flat[0]) for value in values)
        if self.error:
            raise RuntimeError("model failed")
        return [np.tile([2.0, 0.0], (len(values), 1))]


@pytest.fixture(params=["image", "video", "audio", "audio-tensor"])
def benchmark(request, tmp_path, monkeypatch):
    modality = request.param.split("-")[0]
    module = importlib.import_module(f"gasbench.benchmarks.{modality}_bench")
    dataset = BenchmarkDatasetConfig("tiny", "test/repo", modality, "real")
    cache = tmp_path / "cache"
    dataset_dir = cache / "datasets" / dataset.name
    samples_dir = dataset_dir / "samples"
    samples_dir.mkdir(parents=True)
    metadata = {}
    for i in range(5):
        name = f"sample_{i}.bin"
        if request.param == "audio-tensor":
            import torch

            name = f"sample_{i}.pt"
            torch.save({
                "waveform": torch.full((8,), float(i)), "label": 0,
                # Embedded legacy metadata must not change the frozen sample ID.
                "metadata": {"path_in_archive": "different-embedded-path"},
            }, samples_dir / name)
        else:
            (samples_dir / name).write_bytes(bytes([i]))
        metadata[name] = {"source_file": name, "member_path": name}
    (dataset_dir / "sample_metadata.json").write_text(json.dumps(metadata))
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"hf_resolved_revision": "fixed-revision"})
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.py").write_text("fixture model")
    monkeypatch.setattr(common, "discover_benchmark_datasets", lambda **_: [dataset])
    monkeypatch.setattr(
        common, "calculate_weighted_dataset_sampling", lambda *_: {dataset.name: 5}
    )
    decoded = []

    def decode(sample, **kwargs):
        decoded.append(sample["source_file"])
        data = sample.get("image", sample.get("video_bytes", sample.get("audio_bytes")))
        shape = {"image": (2, 2, 3), "video": (2, 2, 2, 3), "audio": (8,)}[modality]
        return np.full(shape, data[0], dtype=np.float32), 0

    def augment(array, *args, **kwargs):
        return array, None, None, None

    if modality == "image":
        monkeypatch.setattr(module, "process_image_sample", decode)
        monkeypatch.setattr(module, "apply_random_augmentations", augment)
        monkeypatch.setattr(module, "apply_robustness_augmentations", augment)
    elif modality == "video":
        monkeypatch.setattr(module, "process_video_bytes_sample", decode)
        monkeypatch.setattr(module, "apply_random_augmentations", augment)
        monkeypatch.setattr(module, "apply_video_robustness_augmentations", augment)
    else:
        monkeypatch.setattr(module, "process_audio_sample", decode)
    specs = [SimpleNamespace(name="input", shape=[None, 3, 2, 2], type="tensor(float)")]

    def run(session, run_id="run", **extra):
        results = {"errors": []}
        asyncio.run(
            getattr(module, f"run_{modality}_benchmark")(
                session,
                specs,
                results,
                run_id=run_id,
                cache_dir=str(cache),
                batch_size=extra.pop("batch_size", 1),
                skip_missing=True,
                seed=7,
                **extra,
            )
        )
        return results

    return SimpleNamespace(
        run=run,
        model_dir=model_dir,
        cache=cache,
        samples_dir=samples_dir,
        dataset=dataset,
        decoded=decoded,
        modality=modality,
        module=module,
        checkpoint=cache / "runs" / "run" / "checkpoint",
    )


def checkpoint_rows(directory):
    manifest = RecorderCheckpoint.read_manifest(directory)
    return RecorderCheckpoint(directory, manifest).records


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resume_skips_committed_inputs_and_preserves_results(
    benchmark, tmp_path, batch_size
):
    b = benchmark
    uninterrupted = b.run(
        Session(b.model_dir),
        batch_size=batch_size,
        run_id="control",
        records_parquet_path=str(tmp_path / "control.parquet"),
    )
    with pytest.raises(Interrupted):
        b.run(Session(b.model_dir, stop_after=2), batch_size=batch_size)
    committed = checkpoint_rows(b.checkpoint)
    assert len(committed) == 2
    # Cached discovery changes and completed source files disappear. Resume must
    # consume the saved plan, without loading the completed inputs again.
    for row in committed:
        (b.samples_dir / row["source_file"]).unlink()
    (b.samples_dir / "new.bin").write_bytes(b"\xff")
    b.decoded.clear()
    resumed_session = Session(b.model_dir)
    resumed = b.run(
        resumed_session,
        batch_size=batch_size,
        records_parquet_path=str(tmp_path / "resumed.parquet"),
    )
    assert len(resumed_session.calls) == 3
    assert not {row["source_file"] for row in committed}.intersection(b.decoded)
    rows = checkpoint_rows(b.checkpoint)
    assert len(rows) == 5
    keys = ["dataset_name", "iteration_index", "sample_id", "aug_pass"]
    assert len({tuple(row[k] for k in keys) for row in rows}) == 5
    assert (
        resumed[f"{b.modality}_results"]["benchmark_score"]
        == uninterrupted[f"{b.modality}_results"]["benchmark_score"]
    )
    left, right = (
        pd.read_parquet(tmp_path / f"{name}.parquet") for name in ("control", "resumed")
    )
    columns = keys + ["label", "predicted", "correct"]
    pd.testing.assert_frame_equal(
        left[columns].sort_values(keys).reset_index(drop=True),
        right[columns].sort_values(keys).reset_index(drop=True),
    )
    complete_session = Session(b.model_dir)
    b.run(complete_session, batch_size=batch_size)
    assert complete_session.calls == []


def test_changed_pending_sample_aborts_instead_of_returning_partial_score(benchmark):
    b = benchmark
    with pytest.raises(Interrupted):
        b.run(Session(b.model_dir, stop_after=0))
    next(b.samples_dir.iterdir()).write_bytes(b"changed")
    with pytest.raises(CheckpointError, match="changed"):
        b.run(Session(b.model_dir))


def test_checkpoint_startup_never_reads_media_or_augmentation_payloads(benchmark, tmp_path, monkeypatch):
    """The old planner read every full video and augmentation before inference."""
    b = benchmark
    aug_dir = tmp_path / "augmentations"
    aug_dir.mkdir()
    original = common.create_tracker

    def create_tracker(*args, **kwargs):
        original_open = Path.open

        def guarded_open(path, *open_args, **open_kwargs):
            if b.samples_dir in path.parents or aug_dir in path.parents:
                pytest.fail("Checkpoint startup opened a media payload")
            return original_open(path, *open_args, **open_kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", guarded_open)
            tracker = original(*args, **kwargs)
        assert tracker.count == 0
        raise Interrupted()

    monkeypatch.setattr(b.module, "create_tracker", create_tracker)
    extra = {} if b.modality == "audio" else {
        "n_aug_per_dataset": 3, "aug_cache_dir": str(aug_dir),
    }
    with pytest.raises(Interrupted):
        b.run(Session(b.model_dir), **extra)
    manifest = RecorderCheckpoint.read_manifest(b.checkpoint)
    samples = manifest["benchmark"]["samples"]["tiny"]["base"]
    assert samples and all("file_metadata_sha256" in sample for sample in samples)


def test_metadata_identity_detects_same_size_edit_with_restored_mtime(tmp_path):
    import os

    path = tmp_path / "media.bin"
    path.write_bytes(b"before")
    before = path.stat()
    sample = {"video_path": str(path), "file_metadata_sha256": common.file_metadata_digest([path])}
    path.write_bytes(b"after!")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(CheckpointError, match="changed"):
        common.verify_sample(sample)


def test_changed_model_rejected_before_inference(benchmark):
    b = benchmark
    with pytest.raises(Interrupted):
        b.run(Session(b.model_dir, stop_after=1))
    (b.model_dir / "model.py").write_text("different model")
    session = Session(b.model_dir)
    with pytest.raises(CheckpointError, match="configuration or model"):
        b.run(session)
    assert session.calls == []


@pytest.mark.parametrize("model_error", [False, True])
def test_persistence_failure_aborts_every_modality(benchmark, model_error):
    b = benchmark
    session = Session(b.model_dir, error=model_error)
    storage_error = OSError("storage unavailable")

    def persist(directory):
        if list(Path(directory).glob("batch-*.json")):
            raise storage_error

    with pytest.raises(CheckpointError) as error:
        b.run(session, checkpoint_persist=persist)
    assert error.value.__cause__ is storage_error
    assert len(session.calls) == 1


@pytest.mark.parametrize("benchmark", ["audio-tensor"], indirect=True)
def test_unreadable_audio_preserves_pending_batch_and_remaining_samples(benchmark):
    b = benchmark
    selected = list(common.DatasetIterator(
        b.dataset, max_samples=5, cache_dir=str(b.cache), download=False,
        seed=7, metadata_only=True,
    ))
    # Corrupt an input before freezing the plan, with a valid sample ahead of it
    # waiting for its batch and more valid samples after it.
    bad_sample = selected[1]
    Path(bad_sample["audio_path"]).write_bytes(b"not a torch file")
    expected = {sample["source_file"] for sample in selected} - {bad_sample["source_file"]}
    session = Session(b.model_dir)
    result = b.run(session, batch_size=2)
    assert result["errors"]
    assert len(session.calls) == len(expected)
    assert {row["source_file"] for row in checkpoint_rows(b.checkpoint)} == expected
    resumed_session = Session(b.model_dir)
    b.run(resumed_session, batch_size=2)
    assert resumed_session.calls == []


def test_failed_inference_is_committed_and_not_retried(benchmark):
    b = benchmark
    b.run(Session(b.model_dir, error=True))
    assert all(row["status"] == "error" for row in checkpoint_rows(b.checkpoint))
    session = Session(b.model_dir)
    b.run(session)
    assert session.calls == []


def test_augmentation_resumes_separately_from_base_pass(benchmark):
    b = benchmark
    if b.modality == "audio":
        pytest.skip("Audio has no robustness pass")
    with pytest.raises(Interrupted):
        b.run(Session(b.model_dir, stop_after=6), n_aug_per_dataset=3)
    rows = checkpoint_rows(b.checkpoint)
    assert sum(row["aug_pass"] for row in rows) == 1
    session = Session(b.model_dir)
    b.run(session, n_aug_per_dataset=3)
    rows = checkpoint_rows(b.checkpoint)
    assert len(session.calls) == 2
    assert sum(row["aug_pass"] for row in rows) == 3
    assert sum(not row["aug_pass"] for row in rows) == 5


def test_top_level_api_checkpoints_by_default(benchmark, monkeypatch):
    import gasbench.benchmark as driver

    b = benchmark
    session = Session(b.model_dir)

    async def load_model(*args):
        return session, [
            SimpleNamespace(name="input", shape=[None, 3, 2, 2], type="tensor(float)")
        ]

    monkeypatch.setattr(driver, "load_model_for_benchmark", load_model)
    for _ in range(2):
        result = asyncio.run(
            driver.run_benchmark(
                str(b.model_dir),
                b.modality,
                cache_dir=str(b.cache),
                run_id="api",
                skip_missing=True,
                batch_size=2,
            )
        )
        assert result["benchmark_completed"]
    assert len(session.calls) == 5
    assert len(checkpoint_rows(Path(result["checkpoint_dir"]))) == 5


def test_augmentation_cache_cannot_change_across_attempts(benchmark, tmp_path):
    from gasbench.benchmarks.aug_cache import img_aug_cache_path, vid_aug_cache_path
    from gasbench.benchmarks.recording import build_sample_id

    b = benchmark
    if b.modality == "audio":
        pytest.skip("Audio has no robustness pass")
    aug_dir = tmp_path / "augmentations"
    cache_path = img_aug_cache_path if b.modality == "image" else vid_aug_cache_path
    for path in b.samples_dir.iterdir():
        sample = {
            "source_kind": "huggingface",
            "dataset_path": "test/repo",
            "source_file": path.name,
            "member_path": path.name,
        }
        artifact = Path(cache_path(str(aug_dir), build_sample_id(sample), (2, 2)))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            artifact, np.zeros((2, 2, 3) if b.modality == "image" else (2, 2, 2, 3))
        )
    with pytest.raises(Interrupted):
        b.run(
            Session(b.model_dir, stop_after=5),
            n_aug_per_dataset=3,
            aug_cache_dir=str(aug_dir),
        )
    for path in aug_dir.rglob("*.npy"):
        path.write_bytes(b"changed")
    with pytest.raises(CheckpointError, match="augmentation cache"):
        b.run(Session(b.model_dir), n_aug_per_dataset=3, aug_cache_dir=str(aug_dir))
