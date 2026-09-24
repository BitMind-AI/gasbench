"""Append-only inference checkpoints, independent of Modal and the ML stack.

One coordinator must own a run directory at a time. This class does not provide
distributed locking. The manifest must identify the run, model, evaluator,
complete sample plan (including source revisions), seeds, and scoring settings.
Records use BenchmarkRunRecorder fields: run_id, dataset_name, iteration_index,
sample_id, and aug_pass. No parallel prediction schema or identifier is introduced.

Local durability uses fsync and atomic rename. Distributed filesystems require a
``persist(directory)`` callback that commits the mounted filesystem before a
batch is acknowledged. A callback failure makes this instance unusable; recover
by reopening the directory from the durable filesystem, never by deleting it.
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional


class CheckpointError(RuntimeError):
    """Checkpoint is incompatible, corrupt, or cannot safely be persisted."""


def _encode(value, *, sort_keys: bool = True) -> bytes:
    return json.dumps(
        value, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _read(path: Path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"Cannot read checkpoint {path.name}") from exc


# Limit both active I/O and queued decoded batches on remote filesystems.
_READ_WORKERS = 16
_READ_AHEAD = 32


@contextmanager
def _prefetch_batches(paths):
    """Read ahead in parallel, but expose batches in commit order.

    Keep only a bounded window of futures alive. The context owns the executor
    so validation failure also cancels queued reads and joins active readers.
    """
    if not paths:
        yield iter(())
        return
    executor = ThreadPoolExecutor(max_workers=min(_READ_WORKERS, len(paths)))
    pending = deque()
    remaining = iter(paths)

    def submit_next():
        path = next(remaining, None)
        if path is not None:
            pending.append((path, executor.submit(_read, path)))

    def ordered():
        while pending:
            path, future = pending.popleft()
            yield path, future.result()
            submit_next()

    try:
        for _ in range(min(_READ_AHEAD, len(paths))):
            submit_next()
        yield ordered()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def prediction_key(row: Mapping) -> tuple:
    """Identity of one planned prediction, using the existing recorder schema."""
    fields = ("run_id", "dataset_name", "sample_id")
    if any(not isinstance(row.get(field), str) or not row[field] for field in fields):
        raise CheckpointError("Prediction requires run_id, dataset_name, and sample_id")
    index = row.get("iteration_index")
    aug_pass = row.get("aug_pass", False)
    if type(index) is not int or index < 1 or type(aug_pass) is not bool:
        raise CheckpointError(
            "Prediction requires a positive sample index and boolean aug_pass"
        )
    return (row["run_id"], row["dataset_name"], index, row["sample_id"], aug_pass)


class RecorderCheckpoint:
    """Internal persistence for BenchmarkRunRecorder rows.

    Reopening validates every committed batch. Temporary writes are ignored;
    corrupt or missing committed batches fail closed, including a missing tail.
    Replaying an identical record is a no-op; changing a completed prediction
    is rejected.
    """

    @staticmethod
    def read_manifest(directory: Path):
        """Read a saved plan before opening the recorder, which validates its batches."""
        path = Path(directory) / "manifest.json"
        if not path.exists():
            if any(Path(directory).glob("batch-*.json")) or (Path(directory) / "head.json").exists():
                raise CheckpointError("Checkpoint progress exists without a manifest")
            return None
        value = _read(path)
        if not isinstance(value, dict) or value.get("schema_version") != 2 or not isinstance(value.get("manifest"), dict):
            raise CheckpointError("Invalid checkpoint manifest")
        if value.get("sha256") != _digest(value["manifest"]):
            raise CheckpointError("Invalid checkpoint manifest checksum")
        return value["manifest"]

    def __init__(
        self,
        directory: Path,
        manifest: Mapping,
        *,
        persist: Optional[Callable[[Path], None]] = None,
    ):
        self.directory = Path(directory)
        self._persist = persist
        self._usable = True
        self._records = {}
        self._next_batch = 0
        if not isinstance(manifest, Mapping) or not manifest:
            raise ValueError("A nonempty run manifest is required")
        # Normalize JSON types and detach mutable caller-owned objects.
        expected = {
            "schema_version": 2,
            "manifest": json.loads(_encode(dict(manifest))),
        }
        expected["sha256"] = _digest(expected["manifest"])
        self.run_id = expected["manifest"].get("run_id")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Checkpoint requires the benchmark run_id")
        self._manifest_digest = _digest(expected)
        # Provision the run directory before execution. Sync its parent as well
        # so a local crash cannot lose a newly created directory entry.
        self.directory.mkdir(parents=True, exist_ok=True)
        parent_fd = os.open(self.directory.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        manifest_path = self.directory / "manifest.json"
        batches = sorted(self.directory.glob("batch-*.json"))
        if manifest_path.exists():
            if _encode(_read(manifest_path)) != _encode(expected):
                raise CheckpointError("Checkpoint manifest differs from this run")
        else:
            if batches:
                raise CheckpointError("Checkpoint batches exist without a manifest")
            self._write(manifest_path, expected)

        head_path = self.directory / "head.json"
        if not head_path.exists():
            if batches:
                raise CheckpointError("Checkpoint batches exist without a commit head")
            self._write_head(-1, None)
        head_envelope = _read(head_path)
        if not isinstance(head_envelope, dict) or not isinstance(
            head_envelope.get("payload"), dict
        ):
            raise CheckpointError("Invalid checkpoint commit head")
        head = head_envelope["payload"]
        try:
            head_valid = head_envelope.get("sha256") == _digest(head)
        except (TypeError, ValueError):
            head_valid = False
        if not head_valid:
            raise CheckpointError("Invalid checkpoint commit head checksum")
        if (
            not isinstance(head, dict)
            or type(head.get("last_batch")) is not int
            or head["last_batch"] < -1
            or head["last_batch"] >= len(batches)
            or (head["last_batch"] == -1 and head.get("sha256") is not None)
        ):
            raise CheckpointError("Invalid or incomplete checkpoint commit head")
        # Files newer than the head were never acknowledged. They can be safely
        # replaced when the interrupted batch is replayed.
        committed = batches[: head["last_batch"] + 1]
        for index, path in enumerate(committed):
            if path.name != self._batch_name(index):
                raise CheckpointError("Checkpoint batch sequence is incomplete")
        started = time.monotonic()
        with _prefetch_batches(committed) as loaded:
            for path, envelope in loaded:
                if not isinstance(envelope, dict):
                    raise CheckpointError(f"Invalid checkpoint batch {path.name}")
                payload = envelope.get("payload")
                try:
                    valid = (
                        isinstance(payload, dict)
                        and envelope.get("sha256") == _digest(payload)
                        and payload.get("manifest_sha256") == self._manifest_digest
                        and payload.get("batch_index") == self._next_batch
                    )
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise CheckpointError(f"Invalid checkpoint batch {path.name}")
                records = self._validate_records(payload.get("records"))
                if any(prediction_key(row) in self._records for row in records):
                    raise CheckpointError("Duplicate work in committed checkpoint batches")
                self._records.update((prediction_key(row), row) for row in records)
                self._next_batch += 1
        if (
            batches
            and head["last_batch"] >= 0
            and envelope["sha256"] != head.get("sha256")
        ):
            raise CheckpointError("Checkpoint tail differs from commit head")
        if committed:
            logging.getLogger(__name__).info(
                "Validated %s checkpoint batches (%s rows) in %.2fs",
                len(committed), len(self._records), time.monotonic() - started,
            )

    @staticmethod
    def _batch_name(index: int) -> str:
        return f"batch-{index:012d}.json"

    def _validate_records(self, records):
        if not isinstance(records, list) or not records:
            raise CheckpointError("Checkpoint batch must contain records")
        seen = set()
        for row in records:
            if not isinstance(row, dict):
                raise CheckpointError("Checkpoint record must be an object")
            key = prediction_key(row)
            if row.get("run_id") != self.run_id:
                raise CheckpointError("Prediction belongs to a different benchmark run")
            if key in seen:
                raise CheckpointError("Duplicate prediction in checkpoint batch")
            seen.add(key)
        return records

    @property
    def completed_predictions(self) -> frozenset:
        return frozenset(self._records)

    def contains_prediction(self, row: Mapping) -> bool:
        return prediction_key(row) in self._records

    @property
    def records(self) -> list:
        return deepcopy(list(self._records.values()))

    def commit_batch(self, records: Iterable[Mapping]) -> int:
        """Commit new records; return their count after persistence succeeds.

        The caller may replay an overlapping batch on recovery. All previously
        completed records must match exactly, and only new records are appended.
        """
        if not self._usable:
            raise CheckpointError("Persistence failed; reopen the durable checkpoint")
        rows = self._validate_records(
            json.loads(_encode(list(records), sort_keys=False))
        )
        new_rows = []
        for row in rows:
            old = self._records.get(prediction_key(row))
            if old is None:
                new_rows.append(row)
            elif _encode(old) != _encode(row):
                raise CheckpointError(
                    f"Conflicting completed work: {prediction_key(row)}"
                )
        if not new_rows:
            return 0
        payload = {
            "manifest_sha256": self._manifest_digest,
            "batch_index": self._next_batch,
            "records": new_rows,
        }
        self._write(
            self.directory / self._batch_name(self._next_batch),
            {"payload": payload, "sha256": _digest(payload)},
            persist=False,
        )
        self._write_head(self._next_batch, _digest(payload))
        self._records.update((prediction_key(row), row) for row in new_rows)
        self._next_batch += 1
        return len(new_rows)

    def _write_head(self, index: int, checksum: Optional[str]) -> None:
        payload = {"last_batch": index, "sha256": checksum}
        self._write(
            self.directory / "head.json",
            {"payload": payload, "sha256": _digest(payload)},
        )

    def _write(self, destination: Path, value, *, persist: bool = True) -> None:
        # Preserve recorder column order in stored rows; checksums still use
        # canonical key ordering so integrity does not depend on JSON layout.
        temp_path = None
        try:
            content = _encode(value, sort_keys=False)
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.directory, prefix=".pending-", delete=False
                ) as handle:
                    temp_path = Path(handle.name)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, destination)
                fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                if persist and self._persist is not None:
                    self._persist(self.directory)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        except BaseException as exc:
            self._usable = False
            # Sample/dataset handlers must never swallow a failed durable write.
            # Preserve process interruptions while normalizing storage errors.
            if isinstance(exc, Exception) and not isinstance(exc, CheckpointError):
                raise CheckpointError(
                    f"Cannot persist checkpoint {destination.name}: {exc}"
                ) from exc
            raise
