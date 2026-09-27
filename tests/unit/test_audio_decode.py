"""Native decoder hangs must be killable, and the next request must recover."""

import io
import shutil
import time
import wave

import numpy as np
import pytest

from gasbench.processing.audio_decode import AudioDecoderWorker


def simulated_decoder(connection):
    while True:
        payload, _ = connection.recv()
        if payload == b"hang":
            time.sleep(60)
        connection.send((True, np.array([0.25], dtype=np.float32)))


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
