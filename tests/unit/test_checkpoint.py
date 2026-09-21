"""Exercise crash recovery, replay safety, and persistence failures."""

import json
import os
import subprocess
import sys

import pytest

from gasbench.checkpoint import CheckpointError, CheckpointStore


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


def record(work_id, prediction=1):
    return {"work_id": work_id, "predicted": prediction, "probs": [0.1, 0.9]}


def test_resume_and_overlapping_replay_preserve_exact_records(tmp_path, manifest):
    first, augmented, second = [record(key) for key in ("a/base", "a/aug", "b/base")]
    store = CheckpointStore(tmp_path, manifest)
    store.commit_batch([first, augmented])
    resumed = CheckpointStore(tmp_path, manifest)
    assert resumed.completed_ids == {"a/base", "a/aug"}
    assert resumed.commit_batch([first, second]) == 1
    assert resumed.commit_batch([first, augmented, second]) == 0
    assert CheckpointStore(tmp_path, manifest).records == [first, augmented, second]


def test_conflicting_replay_rejects_entire_batch(tmp_path, manifest):
    store = CheckpointStore(tmp_path, manifest)
    store.commit_batch([record("a")])
    with pytest.raises(CheckpointError, match="Conflicting"):
        store.commit_batch([record("b"), record("a", prediction=0)])
    assert CheckpointStore(tmp_path, manifest).records == [record("a")]


@pytest.mark.parametrize("field", ["run_id", "model_hash", "benchmark_version", "seed", "sample_plan", "settings"])
def test_changed_run_identity_fails_closed(tmp_path, manifest, field):
    CheckpointStore(tmp_path, manifest).commit_batch([record("a")])
    changed = {**manifest, field: "changed"}
    with pytest.raises(CheckpointError, match="manifest differs"):
        CheckpointStore(tmp_path, changed)
    assert CheckpointStore(tmp_path, manifest).records == [record("a")]


def test_callers_cannot_mutate_checkpoint_state(tmp_path, manifest):
    store = CheckpointStore(tmp_path, manifest)
    row = record("a")
    store.commit_batch([row])
    row["probs"][0] = 100
    exposed = store.records
    exposed[0]["probs"][0] = 200
    assert store.records == CheckpointStore(tmp_path, manifest).records == [record("a")]


@pytest.mark.parametrize("damage", ["truncated", "checksum", "missing_batch", "missing_tail", "head_checksum", "missing_manifest"])
def test_committed_corruption_never_silently_restarts(tmp_path, manifest, damage):
    store = CheckpointStore(tmp_path, manifest)
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
        CheckpointStore(tmp_path, manifest)


def test_batch_from_another_manifest_is_rejected(tmp_path, manifest):
    source = CheckpointStore(tmp_path / "source", manifest)
    source.commit_batch([record("a")])
    CheckpointStore(tmp_path / "target", {**manifest, "seed": 8})
    batch = next(source.directory.glob("batch-*.json"))
    (tmp_path / "target" / batch.name).write_bytes(batch.read_bytes())
    (tmp_path / "target" / "head.json").write_bytes((source.directory / "head.json").read_bytes())
    with pytest.raises(CheckpointError, match="Invalid checkpoint batch"):
        CheckpointStore(tmp_path / "target", {**manifest, "seed": 8})


@pytest.mark.parametrize("crash_at", ["batch_rename", "head_rename", "after_commit"])
def test_killed_process_recovers_only_complete_batches(tmp_path, manifest, crash_at):
    CheckpointStore(tmp_path, manifest).commit_batch([record("a")])
    program = """
import json, os, sys
from pathlib import Path
from gasbench.checkpoint import CheckpointStore
store = CheckpointStore(Path(sys.argv[1]), json.loads(sys.argv[2]))
replace = os.replace
def crash_replace(source, target):
    if sys.argv[3] == 'batch_rename' or (sys.argv[3] == 'head_rename' and target.name == 'head.json'):
        os._exit(71)
    replace(source, target)
os.replace = crash_replace
store.commit_batch([{'work_id': 'b', 'predicted': 1, 'probs': [0.1, 0.9]}])
os._exit(71)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), json.dumps(manifest), crash_at],
        env=os.environ.copy(), check=False,
    )
    assert result.returncode == 71
    resumed = CheckpointStore(tmp_path, manifest)
    assert resumed.records == ([record("a")] if crash_at != "after_commit" else [record("a"), record("b")])
    resumed.commit_batch([record("b"), record("c")])
    assert CheckpointStore(tmp_path, manifest).records == [record("a"), record("b"), record("c")]


def test_failed_remote_commit_is_not_acknowledged(tmp_path, manifest):
    calls = []

    def persist(directory):
        calls.append(directory)
        if len(calls) > 2:
            raise OSError("remote storage unavailable")

    store = CheckpointStore(tmp_path, manifest, persist=persist)
    with pytest.raises(OSError, match="remote storage"):
        store.commit_batch([record("a")])
    assert store.completed_ids == set()
    with pytest.raises(CheckpointError, match="reopen"):
        store.commit_batch([record("b")])


def test_persistence_callback_sees_complete_checkpoint_before_ack(tmp_path, manifest):
    snapshots = []

    def persist(directory):
        if (directory / "head.json").exists():
            snapshots.append(CheckpointStore(directory, manifest).records)

    store = CheckpointStore(tmp_path, manifest, persist=persist)
    store.commit_batch([record("a")])
    assert snapshots == [[], [record("a")]]


@pytest.mark.parametrize("rows", [[record("a"), record("a")], [{"work_id": ""}], [{"work_id": 1}], [None], []])
def test_invalid_batch_never_changes_progress(tmp_path, manifest, rows):
    store = CheckpointStore(tmp_path, manifest)
    with pytest.raises(CheckpointError):
        store.commit_batch(rows)
    assert CheckpointStore(tmp_path, manifest).records == []
