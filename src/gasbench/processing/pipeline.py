"""Multiprocessing pipeline for efficient data loading and preprocessing.

This module provides a producer-consumer pattern where:
- Worker processes handle I/O and CPU-intensive preprocessing (loading, augmentation)
- Main process consumes preprocessed samples for GPU inference
- Queue stays full to prevent GPU starvation
"""

import multiprocessing as mp
import os
import queue
import time
from typing import Dict, Any, Optional, Iterator
import traceback

import numpy as np

from ..logger import get_logger
from ..processing.media import process_image_sample, process_video_bytes_sample
from ..processing.transforms import (
    apply_random_augmentations,
    compress_image_jpeg_pil,
    compress_video_frames_jpeg_torchvision,
)

logger = get_logger(__name__)


def get_optimal_worker_count() -> int:
    """Determine optimal number of worker processes for this hardware.

    Strategy:
    - Check environment variable for override (GASBENCH_WORKERS)
    - Use cpu_count() - 1 to leave one core for main process
    - Minimum of 1 worker, maximum of 16 (diminishing returns)

    Returns:
        Number of worker processes to use
    """
    env_workers = os.environ.get("GASBENCH_WORKERS")
    if env_workers:
        try:
            return max(1, int(env_workers))
        except ValueError:
            pass

    cpu_count = mp.cpu_count()
    return max(1, min(cpu_count - 1, 16))


class PreprocessingPipeline:
    """Parallel preprocessing pipeline with producer-consumer pattern."""

    def __init__(
        self,
        preprocess_fn,
        num_workers: Optional[int] = None,
        queue_size: int = 32,
        timeout: float = 5.0,
    ):
        """Initialize the preprocessing pipeline.

        Args:
            preprocess_fn: Function to preprocess samples. Should be picklable.
                Signature: preprocess_fn(sample) -> processed_sample or None
            num_workers: Number of worker processes. Defaults to cpu_count() - 1
            queue_size: Size of the output queue (prefetch buffer)
            timeout: Timeout for queue operations in seconds
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        self.preprocess_fn = preprocess_fn
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.timeout = timeout

        self.input_queue = None
        self.output_queue = None
        self.workers = []
        self.running = False

        logger.info(
            f"Initialized preprocessing pipeline with {num_workers} workers, queue size {queue_size}"
        )

    def start(self):
        """Start the worker processes."""
        if self.running:
            logger.warning("Pipeline already running")
            return

        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue(maxsize=self.queue_size * 2)
        self.output_queue = ctx.Queue(maxsize=self.queue_size)

        # Start workers
        self.workers = []
        for worker_id in range(self.num_workers):
            worker = ctx.Process(
                target=self._worker_loop,
                args=(
                    worker_id,
                    self.input_queue,
                    self.output_queue,
                    self.preprocess_fn,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        self.running = True
        logger.info(f"Started {len(self.workers)} worker processes")

    def stop(self):
        """Stop all worker processes gracefully."""
        if not self.running:
            return

        logger.info("Stopping preprocessing pipeline...")

        # Send stop signals to all workers
        for _ in range(self.num_workers):
            try:
                self.input_queue.put(None, timeout=self.timeout)
            except queue.Full:
                pass

        # Wait for workers to finish with timeout
        for worker in self.workers:
            worker.join(timeout=self.timeout)
            if worker.is_alive():
                logger.warning(
                    f"Worker {worker.pid} did not stop gracefully, terminating"
                )
                worker.terminate()
                worker.join(timeout=1.0)

        self.workers = []
        self.running = False
        self._clear_queue(self.input_queue)
        self._clear_queue(self.output_queue)

        logger.info("Pipeline stopped")

    def _clear_queue(self, q):
        """Clear all items from a queue."""
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except (queue.Empty, AttributeError):
            pass

    @staticmethod
    def _worker_loop(worker_id, input_queue, output_queue, preprocess_fn):
        """Worker process loop that processes samples.

        Each worker:
        1. Gets raw sample from input queue
        2. Applies preprocessing function
        3. Puts result in output queue
        4. Continues until None sentinel is received
        """
        processed_count = 0
        error_count = 0

        try:
            while True:
                try:
                    # Get next sample with timeout
                    item = input_queue.get(timeout=1.0)
                    if item is None:
                        break

                    sample_idx, sample = item

                    try:
                        result = preprocess_fn(sample)

                        # Put preprocessed result in output queue
                        output_queue.put((sample_idx, result, None), timeout=5.0)
                        processed_count += 1

                    except Exception as e:
                        error_msg = f"Worker {worker_id} preprocessing error: {str(e)}"
                        output_queue.put((sample_idx, None, error_msg), timeout=5.0)
                        error_count += 1

                except queue.Empty:
                    # No sample available, continue waiting
                    continue

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Worker {worker_id} crashed: {e}\n{traceback.format_exc()}")

    def process_iterator(self, sample_iterator: Iterator) -> Iterator[tuple]:
        """Process samples from an iterator using the worker pool.

        Args:
            sample_iterator: Iterator that yields raw samples

        Yields:
            Tuples of (sample_idx, processed_sample, error_msg)
            - If successful: (idx, processed_sample, None)
            - If error: (idx, None, error_msg)
        """
        if not self.running:
            raise RuntimeError("Pipeline not started. Call start() first.")

        samples_sent = 0
        samples_received = 0
        feeder_done = False

        # Feed samples into input queue (runs in main process)
        def feed_samples():
            nonlocal samples_sent, feeder_done
            try:
                for sample_idx, sample in enumerate(sample_iterator):
                    # Put sample in input queue (blocks if queue is full)
                    self.input_queue.put((sample_idx, sample), timeout=self.timeout)
                    samples_sent += 1
            except Exception as e:
                logger.error(f"Error feeding samples: {e}")
            finally:
                feeder_done = True

        # Start feeding in a background thread
        import threading

        feeder_thread = threading.Thread(target=feed_samples, daemon=True)
        feeder_thread.start()

        # Yield processed samples as they become available
        # maintain order by tracking which indices we've seen
        pending_results = {}
        next_idx_to_yield = 0

        while samples_received < samples_sent or not feeder_done:
            try:
                # Get next result from output queue
                sample_idx, result, error_msg = self.output_queue.get(timeout=0.1)
                samples_received += 1

                # Store result temporarily
                pending_results[sample_idx] = (result, error_msg)

                # Yield results in order
                while next_idx_to_yield in pending_results:
                    result, error_msg = pending_results.pop(next_idx_to_yield)
                    yield (next_idx_to_yield, result, error_msg)
                    next_idx_to_yield += 1

            except queue.Empty:
                # No results ready yet, continue waiting
                if feeder_done and samples_received >= samples_sent:
                    break
                continue

        # Wait for feeder thread to complete
        feeder_thread.join(timeout=1.0)

        # Yield any remaining results
        for idx in sorted(pending_results.keys()):
            result, error_msg = pending_results[idx]
            yield (idx, result, error_msg)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def preprocess_image_sample_worker(sample: Dict) -> Optional[Dict]:
    """Worker function that preprocesses a single image sample.

    This function is designed to be run in a worker process.
    It performs all CPU-intensive operations:
    - PIL to numpy conversion
    - Augmentations (rotation, flips, distortions)
    - JPEG compression
    - Transpose operations

    Args:
        sample: Raw sample dict with PIL image and metadata

    Returns:
        Processed sample dict with preprocessed image array and metadata,
        or None if preprocessing failed
    """
    try:
        image_array, true_label_multiclass = process_image_sample(sample)

        if image_array is None or true_label_multiclass is None:
            return None

        try:
            chw = image_array[0]
            hwc = np.transpose(chw, (1, 2, 0))
            aug_hwc, _, _, _ = apply_random_augmentations(hwc)
            aug_hwc = compress_image_jpeg_pil(aug_hwc, quality=75)
            aug_chw = np.transpose(aug_hwc, (2, 0, 1))
            image_array = np.expand_dims(aug_chw, 0)
        except Exception:
            # If augmentation fails, continue with unaugmented image
            pass

        return {
            "image_array": image_array,
            "true_label_multiclass": true_label_multiclass,
            "sample_metadata": sample,
        }

    except Exception:
        return None


def preprocess_video_sample_worker(sample: Dict) -> Optional[Dict]:
    """Worker function that preprocesses a single video sample.

    This function is designed to be run in a worker process.
    It performs all CPU-intensive operations:
    - Video decoding from bytes
    - Frame extraction
    - Augmentations (rotation, flips, distortions)
    - JPEG compression per frame
    - Transpose operations

    Args:
        sample: Raw sample dict with video bytes and metadata

    Returns:
        Processed sample dict with preprocessed video array and metadata,
        or None if preprocessing failed
    """
    try:
        # Decode video bytes and extract frames
        video_array, true_label_multiclass = process_video_bytes_sample(sample)

        if video_array is None or true_label_multiclass is None:
            return None

        try:
            tchw = video_array[0]
            thwc = np.transpose(tchw, (0, 2, 3, 1))
            aug_thwc, _, _, _ = apply_random_augmentations(thwc)
            aug_thwc = compress_video_frames_jpeg_torchvision(aug_thwc, quality=75)
            aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2))
            video_array = np.expand_dims(aug_tchw, 0)
        except Exception:
            # If augmentation fails, continue with unaugmented video
            pass

        return {
            "video_array": video_array,
            "true_label_multiclass": true_label_multiclass,
            "sample_metadata": sample,
        }

    except Exception:
        return None
