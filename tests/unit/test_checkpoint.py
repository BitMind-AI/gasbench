"""Exercise crash recovery, replay safety, and persistence failures."""

import json
import os
import subprocess
import sys

import pytest

from gasbench.benchmarks._checkpoint import (
    CheckpointError,
    RecorderCheckpoint,
    prediction_key,
)


@pytest.fixture
def manifest():
    return {
        "run_id": "run-a",
        "model_hash": "model-a",
        "benchmark_version": "test",
        "seed": 7,
        "sample_plan": ["source@revision/sample-a", "source@revision/sample-b"],
        "settings": {"augmentation": True, "weight": 0.2},
    }


def record(sample_id, prediction=1):
    sample_id, _, pass_name = sample_id.partition("/")
    return {
        "run_id": "run-a",
        "dataset_name": "dataset",
        "sample_id": sample_id,
        "iteration_index": 1,
        "aug_pass": pass_name == "aug",
        "predicted": prediction,
        "probs": [0.1, 0.9],
    }


def test_resume_and_overlapping_replay_preserve_exact_records(tmp_path, manifest):
    first, augmented, second = [record(key) for key in ("a/base", "a/aug", "b/base")]
    store = RecorderCheckpoint(tmp_path, manifest)
    store.commit_batch([first, augmented])
    resumed = RecorderCheckpoint(tmp_path, manifest)
    assert resumed.completed_predictions == {
        prediction_key(first),
        prediction_key(augmented),
    }
    assert resumed.commit_batch([first, second]) == 1
    assert resumed.commit_batch([first, augmented, second]) == 0
    assert RecorderCheckpoint(tmp_path, manifest).records == [first, augmented, second]


def test_conflicting_replay_rejects_entire_batch(tmp_path, manifest):
    store = RecorderCheckpoint(tmp_path, manifest)
    store.commit_batch([record("a")])
    with pytest.raises(CheckpointError, match="Conflicting"):
        store.commit_batch([record("b"), record("a", prediction=0)])
    assert RecorderCheckpoint(tmp_path, manifest).records == [record("a")]


@pytest.mark.parametrize(
    "field",
    ["run_id", "model_hash", "benchmark_version", "seed", "sample_plan", "settings"],
)
def test_changed_run_identity_fails_closed(tmp_path, manifest, field):
    RecorderCheckpoint(tmp_path, manifest).commit_batch([record("a")])
    changed = {**manifest, field: "changed"}
    with pytest.raises(CheckpointError, match="manifest differs"):
        RecorderCheckpoint(tmp_path, changed)
    assert RecorderCheckpoint(tmp_path, manifest).records == [record("a")]


def test_callers_cannot_mutate_checkpoint_state(tmp_path, manifest):
    store = RecorderCheckpoint(tmp_path, manifest)
    row = record("a")
    store.commit_batch([row])
    row["probs"][0] = 100
    exposed = store.records
    exposed[0]["probs"][0] = 200
    assert (
        store.records == RecorderCheckpoint(tmp_path, manifest).records == [record("a")]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "truncated",
        "checksum",
        "missing_batch",
        "missing_tail",
        "head_checksum",
        "missing_manifest",
    ],
)
def test_committed_corruption_never_silently_restarts(tmp_path, manifest, damage):
    store = RecorderCheckpoint(tmp_path, manifest)
    store.commit_batch([record("a")])
    store.commit_batch([record("b")])
    first = sorted(tmp_path.glob("batch-*.json"))[0]
    if damage == "truncated":
        first.write_text('{"payload":')
    elif damage == "checksum":
        envelope = json.loads(first.read_text())
        envelope["payload"]["records"][0]["predicted"] = 0
        first.write_text(json.dumps(envelope))
    elif damage == "missing_batch":
        first.unlink()
    elif damage == "missing_tail":
        sorted(tmp_path.glob("batch-*.json"))[-1].unlink()
    elif damage == "head_checksum":
        head_path = tmp_path / "head.json"
        head = json.loads(head_path.read_text())
        head["payload"]["last_batch"] = -1
        head_path.write_text(json.dumps(head))
    else:
        (tmp_path / "manifest.json").unlink()
    with pytest.raises(CheckpointError):
        RecorderCheckpoint(tmp_path, manifest)


def test_batch_from_another_manifest_is_rejected(tmp_path, manifest):
    source = RecorderCheckpoint(tmp_path / "source", manifest)
    source.commit_batch([record("a")])
    RecorderCheckpoint(tmp_path / "target", {**manifest, "seed": 8})
    batch = next(source.directory.glob("batch-*.json"))
    (tmp_path / "target" / batch.name).write_bytes(batch.read_bytes())
    (tmp_path / "target" / "head.json").write_bytes(
        (source.directory / "head.json").read_bytes()
    )
    with pytest.raises(CheckpointError, match="Invalid checkpoint batch"):
        RecorderCheckpoint(tmp_path / "target", {**manifest, "seed": 8})


@pytest.mark.parametrize("crash_at", ["batch_rename", "head_rename", "after_commit"])
def test_killed_process_recovers_only_complete_batches(tmp_path, manifest, crash_at):
    RecorderCheckpoint(tmp_path, manifest).commit_batch([record("a")])
    program = """
import json, os, sys
from pathlib import Path
from gasbench.benchmarks._checkpoint import RecorderCheckpoint
store = RecorderCheckpoint(Path(sys.argv[1]), json.loads(sys.argv[2]))
replace = os.replace
def crash_replace(source, target):
    if sys.argv[3] == 'batch_rename' or (sys.argv[3] == 'head_rename' and target.name == 'head.json'):
        os._exit(71)
    replace(source, target)
os.replace = crash_replace
store.commit_batch([{'run_id': 'run-a', 'dataset_name': 'dataset', 'sample_id': 'b', 'iteration_index': 1, 'aug_pass': False, 'predicted': 1, 'probs': [0.1, 0.9]}])
os._exit(71)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), json.dumps(manifest), crash_at],
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 71
    resumed = RecorderCheckpoint(tmp_path, manifest)
    assert resumed.records == (
        [record("a")] if crash_at != "after_commit" else [record("a"), record("b")]
    )
    resumed.commit_batch([record("b"), record("c")])
    assert RecorderCheckpoint(tmp_path, manifest).records == [
        record("a"),
        record("b"),
        record("c"),
    ]


@pytest.mark.parametrize("failure", ["callback", "rename", "fsync"])
def test_failed_commit_is_not_acknowledged(tmp_path, manifest, monkeypatch, failure):
    calls = []
    storage_error = RuntimeError("remote storage unavailable") if failure == "callback" else OSError("local storage unavailable")

    def fail(*args):
        raise storage_error

    def persist(directory):
        calls.append(directory)
        if failure == "callback" and len(calls) > 2:
            fail()

    store = RecorderCheckpoint(tmp_path, manifest, persist=persist)
    if failure == "rename":
        monkeypatch.setattr(os, "replace", fail)
    elif failure == "fsync":
        monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(CheckpointError, match="storage unavailable") as error:
        store.commit_batch([record("a")])
    assert error.value.__cause__ is storage_error
    assert store.completed_predictions == set()
    with pytest.raises(CheckpointError, match="reopen"):
        store.commit_batch([record("b")])


def test_persistence_callback_sees_complete_checkpoint_before_ack(tmp_path, manifest):
    snapshots = []

    def persist(directory):
        if (directory / "head.json").exists():
            snapshots.append(RecorderCheckpoint(directory, manifest).records)

    store = RecorderCheckpoint(tmp_path, manifest, persist=persist)
    store.commit_batch([record("a")])
    assert snapshots == [[], [record("a")]]


@pytest.mark.parametrize(
    "rows",
    [
        [record("a"), record("a")],
        [{**record("a"), "sample_id": ""}],
        [{**record("a"), "iteration_index": 0}],
        [{**record("a"), "run_id": "other"}],
        [{**record("a"), "aug_pass": "false"}],
        [None],
        [],
    ],
)
def test_invalid_batch_never_changes_progress(tmp_path, manifest, rows):
    store = RecorderCheckpoint(tmp_path, manifest)
    with pytest.raises(CheckpointError):
        store.commit_batch(rows)
    assert RecorderCheckpoint(tmp_path, manifest).records == []


def test_saved_manifest_is_checked_before_any_predictions_exist(tmp_path, manifest):
    RecorderCheckpoint(tmp_path, manifest)
    assert RecorderCheckpoint.read_manifest(tmp_path) == manifest
    path = tmp_path / "manifest.json"
    saved = json.loads(path.read_text())
    saved["manifest"]["seed"] = "corrupted"
    path.write_text(json.dumps(saved))
    with pytest.raises(CheckpointError, match="manifest checksum"):
        RecorderCheckpoint.read_manifest(tmp_path)


def test_parallel_reads_restore_commit_order_and_ignore_uncommitted_tail(tmp_path, manifest, monkeypatch):
    from threading import Event
    from gasbench.benchmarks import _checkpoint as checkpoint

    store = RecorderCheckpoint(tmp_path, manifest)
    expected = [record(str(index)) for index in range(4)]
    for row in expected:
        store.commit_batch([row])
    tail = tmp_path / store._batch_name(len(expected))
    tail.write_text("uncommitted partial JSON")
    second_finished = Event()
    original = checkpoint._read
    completed = []

    def read(path):
        assert path != tail
        if path.name == store._batch_name(0):
            assert second_finished.wait(5), "Reads were serialized behind the first batch"
        result = original(path)
        if path.name.startswith("batch-"):
            completed.append(path.name)
        if path.name == store._batch_name(1):
            second_finished.set()
        return result

    monkeypatch.setattr(checkpoint, "_read", read)
    monkeypatch.setattr(checkpoint, "_READ_WORKERS", 2)
    assert RecorderCheckpoint(tmp_path, manifest).records == expected
    assert completed.index(store._batch_name(1)) < completed.index(store._batch_name(0))


@pytest.mark.parametrize("stop_early", [False, True])
def test_prefetch_is_bounded_and_releases_executor_on_consumer_failure(monkeypatch, stop_early):
    from pathlib import Path
    from gasbench.benchmarks import _checkpoint as checkpoint

    window = 3
    submitted = consumed = 0
    shutdown = []

    class Future:
        def __init__(self, fn, path):
            self.fn, self.path = fn, path

        def result(self):
            nonlocal consumed
            consumed += 1
            return self.fn(self.path)

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, path):
            nonlocal submitted
            submitted += 1
            assert submitted - consumed <= window
            return Future(fn, path)

        def shutdown(self, **kwargs):
            shutdown.append(kwargs)

    monkeypatch.setattr(checkpoint, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(checkpoint, "_READ_AHEAD", window)
    monkeypatch.setattr(checkpoint, "_read", lambda path: path.name)
    paths = [Path(str(i)) for i in range(11)]
    try:
        with checkpoint._prefetch_batches(paths) as loaded:
            for index, (path, value) in enumerate(loaded):
                assert path == paths[index] and value == path.name
                if stop_early:
                    raise CheckpointError("validation rejected batch")
    except CheckpointError:
        assert stop_early
    assert shutdown == [{"wait": True, "cancel_futures": True}]
    if not stop_early:
        assert submitted == consumed == len(paths)


def test_prefetched_read_failure_fails_closed(tmp_path, manifest, monkeypatch):
    from gasbench.benchmarks import _checkpoint as checkpoint

    store = RecorderCheckpoint(tmp_path, manifest)
    for index in range(5):
        store.commit_batch([record(str(index))])
    original = checkpoint._read

    def read(path):
        if path.name == store._batch_name(2):
            raise CheckpointError("read failed")
        return original(path)

    monkeypatch.setattr(checkpoint, "_read", read)
    with pytest.raises(CheckpointError, match="read failed"):
        RecorderCheckpoint(tmp_path, manifest)
