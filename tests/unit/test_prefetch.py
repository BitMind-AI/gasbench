"""The shared producer must preserve outcomes, surface failures and stop cleanly."""

import threading
from types import SimpleNamespace

import pytest

from gasbench.benchmarks.prefetch import PrefetchPipeline


class Samples(list):
    config = SimpleNamespace(name="tiny")


class Pipeline(PrefetchPipeline):
    def __init__(self, samples, prepare, **kwargs):
        self.prepare = prepare
        super().__init__(Samples(samples), (2, 2), 1, 42, **kwargs)

    def _read_and_preprocess(self, sample, index, name):
        return self.prepare(sample)


def test_prefetch_preserves_plan_order_and_explicit_decode_skips():
    second_ready = threading.Event()

    def prepare(sample):
        if sample == 0:
            assert second_ready.wait(2)
        else:
            second_ready.set()
        return {"value": sample} if sample != 2 else None

    with Pipeline(range(3), prepare, num_workers=2) as pipeline:
        outcomes = [item for batch in pipeline for item in batch]
    assert [item.get("value") for item in outcomes] == [0, 1, None]
    assert outcomes[2]["sample_index"] == 3
    assert outcomes[2]["sample"] == 2
    assert outcomes[2]["skip_reason"]


def test_preparation_error_reaches_consumer():
    def prepare(sample):
        raise OSError("disk unavailable")

    with Pipeline([0], prepare) as pipeline:
        with pytest.raises(OSError, match="disk unavailable"):
            next(pipeline)


def test_prefetch_timeout_is_not_end_of_dataset():
    release = threading.Event()
    pipeline = Pipeline([0], lambda _: release.wait(2), batch_timeout=0.02)
    try:
        with pytest.raises(TimeoutError):
            next(pipeline)
    finally:
        release.set()
        pipeline.close()


def test_closing_full_queue_does_not_deadlock_producer():
    pipeline = Pipeline(range(100), lambda sample: {"value": sample}, max_queue_size=1)
    # First output means the producer has started; leave subsequent work unread.
    next(pipeline)
    closed = threading.Event()
    closer = threading.Thread(
        target=lambda: (pipeline.close(), closed.set()), daemon=True
    )
    closer.start()
    assert closed.wait(3), "Producer remained blocked on an abandoned output queue"
    closer.join()
    assert not pipeline.producer_thread.is_alive()
