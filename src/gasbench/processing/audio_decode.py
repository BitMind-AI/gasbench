"""Reusable, killable audio decoding worker with a wall-clock deadline."""

import atexit
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

AUDIO_DECODE_TIMEOUT = 30
AUDIO_DECODE_WORKERS = 2


def _decode(audio_bytes, target_sr):
    # Imports stay inside the worker, which never inherits a CUDA context.
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(audio_bytes, sample_rate=target_sr, num_channels=1)
        return decoder.get_all_samples().data.squeeze(0).numpy()
    except Exception:
        from .media import _decode_audio_waveform_ffmpeg_cli

        return _decode_audio_waveform_ffmpeg_cli(audio_bytes, target_sr).numpy()


def _worker(connection):
    import torch

    torch.set_num_threads(1)
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                result = (True, _decode(*request))
            except Exception as exc:
                result = (False, str(exc))
            connection.send(result)
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def _start_worker(worker, connection):
    if os.name == "posix":
        os.setsid()
    worker(connection)


class AudioDecoderWorker:
    """Serialize requests; terminate and replace the process after a timeout."""

    def __init__(self, worker=_worker):
        self._worker = worker
        self._process = None
        self._connection = None
        self._lock = threading.Lock()

    def _stop(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            # Kill the process group even if the Python worker exited first:
            # its ffmpeg child can otherwise outlive it.
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                if self._process.is_alive():
                    self._process.terminate()
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process.close()
            self._process = None

    def close(self):
        with self._lock:
            self._stop()

    def decode(self, audio_bytes, sample_rate, timeout=AUDIO_DECODE_TIMEOUT):
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._stop()
                context = multiprocessing.get_context("spawn")
                self._connection, child = context.Pipe()
                self._process = context.Process(
                    target=_start_worker, args=(self._worker, child), daemon=True
                )
                self._process.start()
                child.close()
            connection = self._connection

            def exchange():
                connection.send((audio_bytes, sample_rate))
                return connection.recv()

            # Include IPC in the deadline: large inputs can block send(), and
            # poll() alone does not bound receiving a partially written result.
            with ThreadPoolExecutor(max_workers=1) as transport:
                try:
                    ok, value = transport.submit(exchange).result(timeout=timeout)
                    if not ok:
                        raise ValueError(value)
                    return value
                except FutureTimeoutError as exc:
                    self._stop()
                    raise TimeoutError("Audio decoder exceeded its deadline") from exc
                except BaseException:
                    # Killing the peer unblocks IPC before joining its thread.
                    self._stop()
                    raise


class AudioDecoderPool:
    """Bound concurrent decoders and include pool waits in each request deadline."""

    def __init__(self, max_workers=AUDIO_DECODE_WORKERS, *, worker=_worker):
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("Audio decoder pool requires a positive worker count")
        self._workers = [AudioDecoderWorker(worker=worker) for _ in range(max_workers)]
        self._available = list(self._workers)
        self._condition = threading.Condition()
        self._closed = False

    def decode(self, audio_bytes, sample_rate, timeout=AUDIO_DECODE_TIMEOUT):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._available and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for an audio decoder")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("Audio decoder pool is closed")
            decoder = self._available.pop()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Audio decoder exceeded its deadline")
            return decoder.decode(audio_bytes, sample_rate, timeout=remaining)
        finally:
            with self._condition:
                self._available.append(decoder)
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while len(self._available) < len(self._workers):
                self._condition.wait()
        for decoder in self._workers:
            decoder.close()


_decoder = AudioDecoderPool()
atexit.register(_decoder.close)


def decode_audio(audio_bytes, target_sr, timeout=AUDIO_DECODE_TIMEOUT):
    import torch

    return torch.from_numpy(_decoder.decode(audio_bytes, target_sr, timeout))
