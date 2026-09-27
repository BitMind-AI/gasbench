"""Bounded sample preparation with one owner for workers, errors and shutdown."""

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Empty, Full, Queue
from typing import Optional, TypedDict, Union

import numpy as np

from .timing import StageTimings


class SampleContext(TypedDict):
    sample: dict
    sample_index: int
    dataset_name: str


class PreparedSample(SampleContext):
    data: np.ndarray
    label: int
    sample_seed: Optional[int]


class SkippedSample(SampleContext):
    skip_reason: str


class PrefetchPipeline:
    def __init__(
        self,
        dataset_iterator,
        target_size,
        batch_size,
        seed,
        augment_level=0,
        crop_prob=0.0,
        num_workers=4,
        max_queue_size=6,
        robustness_pass=False,
        aug_cache_dir=None,
        aug_cache_readonly=False,
        tracker=None,
        batch_timeout=300,
        timings=None,
    ):
        if batch_size < 1 or num_workers < 1 or max_queue_size < 1:
            raise ValueError("Batch size, worker count and queue size must be positive")
        self.dataset_iterator = dataset_iterator
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed
        self.augment_level = augment_level
        self.crop_prob = crop_prob
        self.num_workers = num_workers
        self.robustness_pass = robustness_pass
        self.aug_cache_dir = aug_cache_dir
        self.aug_cache_readonly = aug_cache_readonly
        self.tracker = tracker
        self.batch_timeout = batch_timeout
        self.timings = timings if timings is not None else StageTimings()
        self.batch_queue = Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.error = None
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.producer_thread = threading.Thread(target=self._produce, daemon=True)
        self.producer_thread.start()

    def _put(self, batch):
        while not self.stop_event.is_set():
            try:
                self.batch_queue.put(batch, timeout=0.1)
                return
            except Full:
                continue

    def _read_and_preprocess(
        self, sample, index, dataset_name
    ) -> Optional[PreparedSample]:
        raise NotImplementedError

    def _prepare(
        self, sample, index, dataset_name
    ) -> Union[PreparedSample, SkippedSample]:
        with self.timings.measure("prepare"):
            result = self._read_and_preprocess(sample, index, dataset_name)
        if result is None:
            return {
                "sample": sample,
                "sample_index": index,
                "dataset_name": dataset_name,
                "skip_reason": "media-decode-failed",
            }
        return result

    def _produce(self):
        pending = deque()
        try:
            name = self.dataset_iterator.config.name
            samples = enumerate(self.dataset_iterator, 1)
            exhausted = False
            batch = []
            while not self.stop_event.is_set():
                while (
                    len(pending) < self.num_workers * 3
                    and not exhausted
                    and not self.stop_event.is_set()
                ):
                    try:
                        index, sample = next(samples)
                    except StopIteration:
                        exhausted = True
                        break
                    if self.tracker is not None and self.tracker.is_checkpointed(
                        dataset_name=name,
                        sample_index=index,
                        sample=sample,
                        aug_pass=self.robustness_pass,
                    ):
                        self.timings.count("restored_samples")
                        continue
                    pending.append(
                        self.executor.submit(self._prepare, sample, index, name)
                    )
                if not pending:
                    break
                # Prepare concurrently, but preserve the frozen plan's order so
                # worker scheduling cannot change inference batch composition.
                done, _ = wait([pending[0]], timeout=0.1)
                if done:
                    batch.append(pending.popleft().result())
                    if len(batch) >= self.batch_size:
                        self._put(batch)
                        batch = []
            if batch:
                self._put(batch)
        except BaseException as exc:
            self.error = exc
        finally:
            for future in pending:
                future.cancel()
            self._put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.error is not None:
            raise self.error
        try:
            with self.timings.measure("input_wait"):
                batch = self.batch_queue.get(timeout=self.batch_timeout)
        except Empty as exc:
            self.error = TimeoutError("Timed out waiting for sample preparation")
            raise self.error from exc
        if self.error is not None:
            raise self.error
        if batch is None:
            raise StopIteration
        return batch

    def close(self, *, aborted=False):
        self.stop_event.set()
        # The producer's puts and waits are interruptible even if the consumer
        # stops with a full queue. It must stop submitting before shutdown.
        self.producer_thread.join()
        # Running thread work cannot be forcibly killed. Do not hide a failure
        # behind its completion; cancel queued work and let in-flight reads end.
        self.executor.shutdown(wait=not (aborted or self.error), cancel_futures=True)
        while True:
            try:
                self.batch_queue.get_nowait()
            except Empty:
                break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(aborted=exc_type is not None)
