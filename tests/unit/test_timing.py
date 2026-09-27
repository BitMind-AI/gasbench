"""Timing must retain concurrent work and failures without retaining samples."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gasbench.benchmarks import timing


def test_worker_totals_include_failures_and_merge_without_sharing_mutable_state(monkeypatch):
    clock = threading.local()
    monkeypatch.setattr(timing, "perf_counter", lambda: getattr(clock, "now", 0))
    timings = timing.StageTimings()
    durations = list(range(1, 21))

    def prepare(duration):
        clock.now = 0
        try:
            with timings.measure("prepare"):
                clock.now = duration
                timings.count("samples")
                raise ValueError("decode failed")
        except ValueError:
            pass

    with timings.measure("wall"):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(prepare, durations))
        clock.now = 1
    snapshot = timings.snapshot()
    assert snapshot["stages"]["prepare"] == {
        "calls": len(durations), "seconds": sum(durations), "max_seconds": max(durations),
    }
    assert snapshot["counts"]["samples"] == len(durations)
    # Worker totals may exceed wall time and must never be summed into it.
    assert snapshot["stages"]["wall"]["seconds"] == 1
    combined = timing.StageTimings()
    combined.merge(snapshot)
    combined.merge(snapshot)
    assert combined.snapshot()["stages"]["prepare"]["seconds"] == 2 * sum(durations)
    assert combined.snapshot()["stages"]["prepare"]["max_seconds"] == max(durations)
    snapshot["stages"]["prepare"]["seconds"] = -1
    assert timings.snapshot()["stages"]["prepare"]["seconds"] == sum(durations)


def test_timer_preserves_process_interruptions(monkeypatch):
    clock = iter([4, 9])
    monkeypatch.setattr(timing, "perf_counter", lambda: next(clock))
    timings = timing.StageTimings()
    interruption = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as error:
        with timings.measure("inference"):
            raise interruption
    assert error.value is interruption
    assert timings.snapshot()["stages"]["inference"]["seconds"] == 5
