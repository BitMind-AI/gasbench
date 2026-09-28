"""Native decoder hangs must be killable, and the next request must recover."""

import io
import os
import shutil
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from gasbench.processing.audio_decode import AudioDecoderPool, AudioDecoderWorker


def simulated_decoder(connection):
    while True:
        payload, _ = connection.recv()
        if payload == b"hang":
            time.sleep(60)
        connection.send((True, np.array([0.25], dtype=np.float32)))


def barrier_decoder(connection):
    while True:
        payload, _ = connection.recv()
        directory = Path(payload)
        (directory / str(os.getpid())).touch()
        while not (directory / "release").exists():
            time.sleep(0.01)
        connection.send((True, os.getpid()))


def test_pool_runs_independently_and_bounds_queued_request_deadlines(tmp_path):
    pool = AudioDecoderPool(2, worker=barrier_decoder)
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        calls = [executor.submit(pool.decode, str(tmp_path), 16000, timeout=10) for _ in range(2)]
        deadline = time.monotonic() + 8
        while len(list(tmp_path.iterdir())) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(list(tmp_path.iterdir())) == 2, "The pool serialized independent requests"
        with pytest.raises(TimeoutError, match="waiting"):
            pool.decode(str(tmp_path), 16000, timeout=0.02)
        (tmp_path / "release").touch()
        pids = {call.result() for call in calls}
        assert len(pids) == 2
        assert pool.decode(str(tmp_path), 16000, timeout=2) in pids
    finally:
        (tmp_path / "release").touch()
        executor.shutdown(wait=True)
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.decode(str(tmp_path), 16000)


def test_seeded_crop_does_not_modify_global_rng(monkeypatch):
    import torch
    from gasbench.processing import media

    waveform = torch.arange(100, dtype=torch.float32)
    monkeypatch.setattr(media, "_decode_audio_with_timeout", lambda *_: waveform)
    state = torch.random.get_rng_state()
    options = dict(target_sr=10, target_duration_seconds=2, use_random_crop=True, seed=7)
    sample = {"audio_bytes": b"input", "media_type": "real"}
    a, _ = media.process_audio_sample(sample, **options)
    b, _ = media.process_audio_sample(sample, **options)
    assert torch.equal(a, b)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_timeout_stops_running_decoder_and_next_request_restarts_it():
    worker = AudioDecoderWorker(worker=simulated_decoder)
    try:
        expected = worker.decode(b"ok", 16000, timeout=10)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            worker.decode(b"hang", 16000, timeout=0.05)
        assert time.monotonic() - started < 3
        assert worker._process is None
        np.testing.assert_array_equal(worker.decode(b"ok", 16000, timeout=10), expected)
    finally:
        worker.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_worker_decodes_real_wav_and_reuses_process():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.full(1600, 8192, dtype="<i2").tobytes())
    worker = AudioDecoderWorker()
    try:
        process = None
        for _ in range(2):
            result = worker.decode(buffer.getvalue(), 16000, timeout=15)
            np.testing.assert_allclose(result, np.full(1600, 0.25), atol=1e-4)
            if process is None:
                process = worker._process
            else:
                assert worker._process is process
    finally:
        worker.close()
