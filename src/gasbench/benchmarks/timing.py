"""Bounded, thread-safe stage totals for one benchmark attempt.

Durations are inclusive: preparation overlaps inference, and checkpoint writes
are part of checkpoint time. Worker totals must not be added to wall time.
No individual samples, predictions, or historical attempt timings are retained.
"""

from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class StageTimings:
    def __init__(self):
        self._lock = Lock()
        self._stages = {}
        self._counts = {}

    @contextmanager
    def measure(self, stage):
        started = perf_counter()
        try:
            yield
        finally:
            self.observe(stage, perf_counter() - started)

    def observe(self, stage, seconds):
        with self._lock:
            entry = self._stages.setdefault(
                stage, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0}
            )
            entry["calls"] += 1
            entry["seconds"] += seconds
            entry["max_seconds"] = max(entry["max_seconds"], seconds)

    def count(self, name, amount=1):
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + amount

    def snapshot(self):
        with self._lock:
            return {
                "stages": {name: dict(value) for name, value in self._stages.items()},
                "counts": dict(self._counts),
            }

    def merge(self, snapshot):
        with self._lock:
            for name, value in snapshot["stages"].items():
                entry = self._stages.setdefault(
                    name, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0}
                )
                entry["calls"] += value["calls"]
                entry["seconds"] += value["seconds"]
                entry["max_seconds"] = max(entry["max_seconds"], value["max_seconds"])
            for name, count in snapshot["counts"].items():
                self._counts[name] = self._counts.get(name, 0) + count
