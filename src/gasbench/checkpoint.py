"""Append-only inference checkpoints, independent of Modal and the ML stack.

One coordinator must own a run directory at a time. This class does not provide
distributed locking. The manifest must identify the run, model, evaluator,
complete sample plan (including source revisions), seeds, and scoring settings.
Each record has a stable ``work_id`` that distinguishes base/augmentation passes.

Local durability uses fsync and atomic rename. Distributed filesystems require a
``persist(directory)`` callback that commits the mounted filesystem before a
batch is acknowledged. A callback failure makes this instance unusable; recover
by reopening the directory from the durable filesystem, never by deleting it.
"""

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional


class CheckpointError(RuntimeError):
    """Checkpoint is incompatible, corrupt, or cannot safely be persisted."""


def _encode(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _read(path: Path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"Cannot read checkpoint {path.name}") from exc


class CheckpointStore:
    """Persist completed prediction batches and restore them on process restart.

    Reopening validates every committed batch. Temporary writes are ignored;
    corrupt or missing committed batches fail closed, including a missing tail.
    Replaying an identical record is a no-op; changing a completed prediction
    is rejected.
    """

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
        expected = {"schema_version": 1, "manifest": json.loads(_encode(dict(manifest)))}
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
        if not isinstance(head_envelope, dict) or not isinstance(head_envelope.get("payload"), dict):
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
        for path in batches[:head["last_batch"] + 1]:
            if path.name != self._batch_name(self._next_batch):
                raise CheckpointError("Checkpoint batch sequence is incomplete")
            envelope = _read(path)
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
            if any(row["work_id"] in self._records for row in records):
                raise CheckpointError("Duplicate work in committed checkpoint batches")
            self._records.update((row["work_id"], row) for row in records)
            self._next_batch += 1
        if batches and head["last_batch"] >= 0 and envelope["sha256"] != head.get("sha256"):
            raise CheckpointError("Checkpoint tail differs from commit head")

    @staticmethod
    def _batch_name(index: int) -> str:
        return f"batch-{index:012d}.json"

    @staticmethod
    def _validate_records(records):
        if not isinstance(records, list) or not records:
            raise CheckpointError("Checkpoint batch must contain records")
        seen = set()
        for row in records:
            if not isinstance(row, dict):
                raise CheckpointError("Checkpoint record must be an object")
            work_id = row.get("work_id")
            if not isinstance(work_id, str) or not work_id or work_id in seen:
                raise CheckpointError("Checkpoint work IDs must be nonempty and unique")
            seen.add(work_id)
        return records

    @property
    def completed_ids(self) -> frozenset:
        return frozenset(self._records)

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
        rows = self._validate_records(json.loads(_encode(list(records))))
        new_rows = []
        for row in rows:
            old = self._records.get(row["work_id"])
            if old is None:
                new_rows.append(row)
            elif _encode(old) != _encode(row):
                raise CheckpointError(f"Conflicting completed work: {row['work_id']}")
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
        self._records.update((row["work_id"], row) for row in new_rows)
        self._next_batch += 1
        return len(new_rows)

    def _write_head(self, index: int, checksum: Optional[str]) -> None:
        payload = {"last_batch": index, "sha256": checksum}
        self._write(
            self.directory / "head.json",
            {"payload": payload, "sha256": _digest(payload)},
        )

    def _write(self, destination: Path, value, *, persist: bool = True) -> None:
        content = _encode(value)
        temp_path = None
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
        except BaseException:
            self._usable = False
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
