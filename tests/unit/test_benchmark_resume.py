"""Exercise interruption recovery through real iterators, pipelines and scoring."""

import asyncio
import builtins
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
        self.inputs = []
        self.error = error

    def run(self, _, inputs):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise Interrupted()
        values = next(iter(inputs.values()))
        self.inputs.extend(value.copy() for value in values)
        self.calls.extend(int(value.flat[0]) for value in values)
        if self.error:
            raise RuntimeError("model failed")
        signal = values.reshape(len(values), -1)[:, 0].astype(np.float32)
        return [np.column_stack((signal + 1, -signal))]


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
                "waveform": torch.full((96000,), float(i)), "label": 0,
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
        shape = {"image": (2, 2, 3), "video": (2, 2, 2, 3), "audio": (96000,)}[modality]
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
    shape = {"image": [None, 3, 2, 2], "video": [None, 2, 3, 2, 2], "audio": [None, 96000]}[modality]
    specs = [SimpleNamespace(name="input", shape=shape, type="tensor(float)")]

    def run(session, run_id="run", *, results=None, **extra):
        results = {"errors": []} if results is None else results
        asyncio.run(
            getattr(module, f"run_{modality}_benchmark")(
                session,
                specs,
                results,
                run_id=run_id,
                cache_dir=str(cache),
                batch_size=extra.pop("batch_size", 1),
                skip_missing=True,
                seed=extra.pop("seed", 7),
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
        specs=specs,
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
    columns = keys + ["label", "predicted", "correct", "probs"]
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
    populate_augmentation_cache(b, aug_dir)
    original = common.create_tracker

    def create_tracker(*args, **kwargs):
        def guarded_open(original_open):
            def open_file(path, *open_args, **open_kwargs):
                if isinstance(path, (str, Path)):
                    parents = Path(path).parents
                    if b.samples_dir in parents or aug_dir in parents:
                        pytest.fail("Checkpoint startup opened a media payload")
                return original_open(path, *open_args, **open_kwargs)
            return open_file

        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", guarded_open(Path.open))
            patch.setattr(builtins, "open", guarded_open(builtins.open))
            tracker = original(*args, **kwargs)
        assert tracker.count == 0
        raise Interrupted()

    monkeypatch.setattr("gasbench.benchmarks.runner.create_tracker", create_tracker)
    extra = {
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
def test_persistence_failure_aborts_every_modality(benchmark, model_error, monkeypatch):
    from gasbench.benchmarks import recording, timing

    b = benchmark
    session = Session(b.model_dir, error=model_error)
    storage_error = OSError("storage unavailable")
    clock = 0
    monkeypatch.setattr(timing, "perf_counter", lambda: clock)
    # Fail a grouped commit before the remaining inputs/augmentation can run.
    monkeypatch.setattr(recording, "_CHECKPOINT_MAX_PENDING_ROWS", 2)

    def persist(directory):
        nonlocal clock
        if list(Path(directory).glob("batch-*.json")):
            clock += 7
            raise storage_error

    results = {}
    with pytest.raises(CheckpointError) as error:
        b.run(session, checkpoint_persist=persist, results=results, n_aug_per_dataset=3)
    assert error.value.__cause__ is storage_error
    assert len(session.calls) == (1 if model_error else 2)
    base = results["metrics"]["performance"]["groups"]["base"]
    assert base["stages"]["checkpoint_persist"]["seconds"] == 7
    assert base["stages"]["checkpoint"]["seconds"] == 7
    assert base["counts"].get("committed_samples", 0) == 0


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
    assert result["audio_results"]["skipped_samples"] == 1
    assert len(session.calls) == len(expected)
    assert {row["source_file"] for row in checkpoint_rows(b.checkpoint) if row["status"] == "ok"} == expected
    resumed_session = Session(b.model_dir)
    b.run(resumed_session, batch_size=2)
    assert resumed_session.calls == []


def test_failed_inference_is_committed_and_not_retried(benchmark):
    # A failed base batch must stop the run before augmentation can begin.
    n_aug = 3
    b = benchmark
    from gasbench.benchmarks.errors import BenchmarkError

    with pytest.raises(BenchmarkError, match="Inference failed"):
        b.run(Session(b.model_dir, error=True), n_aug_per_dataset=n_aug)
    rows = checkpoint_rows(b.checkpoint)
    assert len(rows) == 1 and rows[0]["status"] == "error"
    session = Session(b.model_dir)
    with pytest.raises(BenchmarkError, match="Checkpoint contains failed inference"):
        b.run(session, n_aug_per_dataset=n_aug)
    assert session.calls == []


def test_augmentation_resumes_separately_from_base_pass(benchmark):
    b = benchmark
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


@pytest.mark.parametrize("n_aug", [0, 3])
def test_top_level_api_checkpoints_by_default(benchmark, monkeypatch, n_aug):
    import gasbench.benchmark as driver

    b = benchmark
    session = Session(b.model_dir)

    async def load_model(*args):
        return session, b.specs

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
                n_aug_per_dataset=n_aug,
            )
        )
        assert result["benchmark_completed"]
        metrics = result[f"{b.modality}_results"]
        assert metrics["total_samples"] == 5 + n_aug
        if n_aug:
            assert metrics["aug_total_samples"] == n_aug
            assert metrics["aug_paired_samples"] == n_aug
        else:
            assert "aug_total_samples" not in metrics
    assert len(session.calls) == 5 + n_aug
    assert len(checkpoint_rows(Path(result["checkpoint_dir"]))) == 5 + n_aug


def populate_augmentation_cache(b, aug_dir):
    from gasbench.benchmarks.aug_cache import aud_aug_cache_path, img_aug_cache_path, vid_aug_cache_path
    from gasbench.benchmarks.recording import build_sample_id

    cache_path = img_aug_cache_path if b.modality == "image" else vid_aug_cache_path
    selected = common.DatasetIterator(
        b.dataset, max_samples=3, cache_dir=str(b.cache), download=False,
        seed=7, metadata_only=True,
    )
    for index, sample in enumerate(selected, 1):
        if b.modality == "audio":
            artifact = Path(aud_aug_cache_path(str(aug_dir), build_sample_id(sample), 7 + index))
        else:
            artifact = Path(cache_path(str(aug_dir), build_sample_id(sample), (2, 2),
                                      **({"num_frames": 2} if b.modality == "video" else {})))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shape = {"image": (2, 2, 3), "video": (2, 2, 2, 3), "audio": (96000,)}[b.modality]
        np.save(artifact, np.zeros(shape))


def test_performance_separates_passes_cache_hits_and_resumed_work(benchmark, tmp_path, monkeypatch):
    from gasbench.benchmark import save_results_to_json
    from gasbench.benchmarks import recording

    # Neither threshold expires: only dataset/pass boundaries should flush.
    monkeypatch.setattr(recording, "monotonic", lambda: 0)
    monkeypatch.setattr(recording, "_CHECKPOINT_MAX_PENDING_ROWS", 100)

    b = benchmark
    aug_dir = tmp_path / "augmentations"
    populate_augmentation_cache(b, aug_dir)
    options = dict(n_aug_per_dataset=3, aug_cache_dir=str(aug_dir), batch_size=2)
    result = b.run(Session(b.model_dir), **options)
    performance = result["metrics"]["performance"]
    groups = performance["groups"]
    rows = checkpoint_rows(b.checkpoint)
    for pass_name, augmented in (("base", False), ("aug", True)):
        count = sum(row["aug_pass"] == augmented for row in rows)
        assert groups[pass_name]["counts"]["inferred_samples"] == count
        assert groups[pass_name]["counts"]["committed_samples"] == count
        assert groups[pass_name]["stages"]["checkpoint"]["calls"] == 1
    assert groups["aug"]["counts"]["augmentation_cache_hits"] == options["n_aug_per_dataset"]
    assert "decode" not in groups["aug"]["stages"]
    assert "source_read" not in groups["aug"]["stages"]
    assert groups["base"]["stages"]["source_read"]["calls"] == sum(not row["aug_pass"] for row in rows)
    exported = Path(save_results_to_json(result, output_dir=str(tmp_path)))
    assert json.loads(exported.read_text())["performance"] == performance

    resumed = b.run(Session(b.model_dir), **options)["metrics"]["performance"]
    assert resumed["scope"] == "current_attempt"
    assert resumed["groups"]["startup"]["counts"]["restored_samples"] == len(rows)
    for pass_name in ("base", "aug"):
        group = resumed["groups"][pass_name]
        assert "inference" not in group["stages"]
        assert "prepare" not in group["stages"]
        assert "checkpoint" not in group["stages"]
        assert group["counts"]["restored_samples"] == groups[pass_name]["counts"]["inferred_samples"]


def test_augmentation_cache_cannot_change_across_attempts(benchmark, tmp_path):
    b = benchmark
    aug_dir = tmp_path / "augmentations"
    populate_augmentation_cache(b, aug_dir)
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


@pytest.mark.parametrize("benchmark", ["audio", "audio-tensor"], indirect=True)
def test_audio_cache_matches_uncached_pass_and_respects_seed_and_readonly(
    benchmark, tmp_path, monkeypatch,
):
    b = benchmark
    aug_dir = tmp_path / "augmentations"
    augment = b.module.apply_audio_robustness_augmentations
    augmented_seeds = []

    def tracked_augment(*args, **kwargs):
        augmented_seeds.append(kwargs["seed"])
        return augment(*args, **kwargs)

    monkeypatch.setattr(b.module, "apply_audio_robustness_augmentations", tracked_augment)
    sessions = []
    # Read-only misses compute without writing; writable misses populate the
    # cache; subsequent runs load it without applying transforms a second time.
    for run_id, readonly in [("readonly", True), ("populate", False), ("reuse", True)]:
        session = Session(b.model_dir)
        result = b.run(
            session, run_id=run_id, n_aug_per_dataset=3,
            aug_cache_dir=str(aug_dir), aug_cache_readonly=readonly,
        )
        assert result["audio_results"]["aug_total_samples"] == 3
        assert not result["errors"]
        sessions.append(session)
        if run_id == "readonly":
            assert not list(aug_dir.rglob("*.npy"))
        else:
            assert len(list(aug_dir.rglob("*.npy"))) == 3
    assert len(augmented_seeds) == 6
    np.testing.assert_array_equal(sessions[0].inputs, sessions[1].inputs)
    np.testing.assert_array_equal(sessions[0].inputs, sessions[2].inputs)
    # A cache with no seed in its key would silently load the previous noise.
    b.run(
        Session(b.model_dir), run_id="different-seed", seed=91, n_aug_per_dataset=3,
        aug_cache_dir=str(aug_dir), aug_cache_readonly=True,
    )
    assert len(augmented_seeds) == 9
    assert set(augmented_seeds[-3:]).isdisjoint(augmented_seeds[:3])


@pytest.mark.parametrize("benchmark", ["audio", "audio-tensor"], indirect=True)
def test_audio_augmented_predictions_use_shared_scoring(benchmark, monkeypatch):
    b = benchmark
    # A conspicuous transform exposes accidentally augmenting the base pass or
    # bypassing augmentation for preprocessed tensors. The model then gives
    # deliberately worse predictions on the augmented inputs.
    monkeypatch.setattr(
        b.module, "apply_audio_robustness_augmentations",
        lambda waveform, **_: (waveform + 10, None, "test", {}),
    )

    class SensitiveSession(Session):
        def run(self, _, inputs):
            values = next(iter(inputs.values()))
            augmented = values[:, 0] >= 10
            return [np.column_stack((~augmented, augmented)).astype(np.float32)]

    weight = 0.35
    result = b.run(SensitiveSession(b.model_dir), n_aug_per_dataset=3, aug_weight=weight)
    metrics = result["audio_results"]
    assert metrics["total_samples"] == 8
    assert metrics["correct_predictions"] == 5
    assert metrics["aug_paired_samples"] == 3
    assert metrics["base_sn34_score"] > metrics["aug_sn34_score"]
    assert metrics["sn34_score"] == pytest.approx(
        (1 - weight) * metrics["base_sn34_score"] + weight * metrics["aug_sn34_score"]
    )


# Compatibility is implemented in the shared tracker, so one modality suffices.
@pytest.mark.parametrize("benchmark", ["image"], indirect=True)
@pytest.mark.parametrize("change", ["reader", "unapproved-evaluator", "model"])
def test_audited_reader_upgrade_preserves_resume_guards(benchmark, monkeypatch, tmp_path, change):
    b = benchmark
    original_fingerprint = common.fingerprint_files
    evaluator_root = Path(common.__file__).resolve().parents[1]
    identity = ["old"]

    def fingerprint(paths, root):
        return identity[0] if root == evaluator_root else original_fingerprint(paths, root)

    monkeypatch.setattr(common, "fingerprint_files", fingerprint)
    (tmp_path / "checkpoint_compatibility.json").write_text(json.dumps({"new": ["old"]}))
    compatible = common.compatible_evaluator_identity
    monkeypatch.setattr(common, "compatible_evaluator_identity", lambda root, current, previous: compatible(tmp_path, current, previous))
    with pytest.raises(Interrupted):
        b.run(Session(b.model_dir, stop_after=2))
    identity[0] = "unapproved" if change == "unapproved-evaluator" else "new"
    if change == "model":
        (b.model_dir / "model.py").write_text("changed inference implementation")
    resumed = Session(b.model_dir)
    if change != "reader":
        with pytest.raises(CheckpointError, match="differs"):
            b.run(resumed)
        assert not resumed.calls
    else:
        b.run(resumed)
        assert len(resumed.calls) == 3
        assert RecorderCheckpoint.read_manifest(b.checkpoint)["benchmark"]["evaluator_sha256"] == "old"
        assert not compatible(tmp_path, "new", "unrelated-old")


@pytest.mark.parametrize("benchmark", ["image", "video", "audio"], indirect=True)
def test_all_corrupt_sources_record_skips_but_cannot_produce_score(benchmark, monkeypatch):
    from gasbench.benchmarks.errors import BenchmarkError

    b = benchmark
    monkeypatch.setattr(b.module.PIPELINE, "_read_and_preprocess", lambda *_: None)
    session = Session(b.model_dir)
    with pytest.raises(BenchmarkError, match="No successful predictions"):
        b.run(session)
    assert not session.calls
    assert len(checkpoint_rows(b.checkpoint)) == 5
    assert all(row["status"] == "skipped" for row in checkpoint_rows(b.checkpoint))


@pytest.mark.parametrize("benchmark", ["image"], indirect=True)
def test_requested_holdouts_cannot_silently_disappear(benchmark, tmp_path):
    with pytest.raises(FileNotFoundError, match="Holdout config"):
        benchmark.run(Session(benchmark.model_dir), holdout_config=str(tmp_path / "missing.yaml"))


@pytest.mark.parametrize("empty", [False, True])
def test_public_driver_reports_completion_only_for_executed_plan(benchmark, monkeypatch, empty):
    from gasbench import benchmark as driver
    from gasbench.benchmarks.errors import BenchmarkError

    b = benchmark
    session = Session(b.model_dir)

    async def load_model(*_):
        return session, b.specs

    monkeypatch.setattr(driver, "load_model_for_benchmark", load_model)
    monkeypatch.setattr(driver, "configure_huggingface_cache", lambda *_: None)
    if empty:
        monkeypatch.setattr(common, "discover_benchmark_datasets", lambda **_: [])
    run = driver.run_benchmark(
        str(b.model_dir), b.modality, cache_dir=str(b.cache), skip_missing=True,
        batch_size=2, seed=7, n_aug_per_dataset=2,
    )
    if empty:
        with pytest.raises(BenchmarkError):
            asyncio.run(run)
        assert not session.calls
    else:
        result = asyncio.run(run)
        assert result["benchmark_completed"]
        assert result[f"{b.modality}_results"]["total_samples"] == 7
        assert result[f"{b.modality}_results"]["aug_total_samples"] == 2
        assert len(session.calls) == 7


@pytest.mark.parametrize("benchmark", ["image"], indirect=True)
def test_image_augmentation_cache_bypasses_source_decoder(benchmark, tmp_path):
    b = benchmark
    aug_dir = tmp_path / "aug"
    populate_augmentation_cache(b, aug_dir)
    result = b.run(Session(b.model_dir), n_aug_per_dataset=3, aug_cache_dir=str(aug_dir))
    assert result["image_results"]["aug_total_samples"] == 3
    assert len(b.decoded) == len(list(b.samples_dir.iterdir()))


@pytest.mark.parametrize("benchmark", ["image"], indirect=True)
@pytest.mark.parametrize("n_aug", [0, 3])
def test_augmentation_cannot_complete_run_without_selected_base_samples(benchmark, monkeypatch, n_aug):
    from gasbench.benchmarks.errors import BenchmarkError

    b = benchmark
    monkeypatch.setattr(common, "calculate_weighted_dataset_sampling", lambda *_: {b.dataset.name: 0})
    session = Session(b.model_dir)
    with pytest.raises(BenchmarkError, match="No cached base samples"):
        b.run(session, n_aug_per_dataset=n_aug)
    assert not session.calls
