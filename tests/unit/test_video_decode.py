"""Frame selection, bounded reads, and failure handling for real video inputs."""

from fractions import Fraction
from io import BytesIO
from types import SimpleNamespace

import av
import numpy as np
import pytest

from gasbench.processing import video_decode
from gasbench.processing.media import process_video_bytes_sample


def encode(frames, *, variable_rate=False):
    output = BytesIO()
    with av.open(output, "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "bgr0"
        stream.time_base = Fraction(1, 1000)
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            if variable_rate:
                frame.time_base = Fraction(1, 1000)
                frame.pts = index * 50 + (index // 3) * 100
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return output.getvalue()


@pytest.mark.parametrize("variable_rate", [False, True])
@pytest.mark.parametrize("frame_rate,num_frames", [(None, 7), (6, 8), (60, 28)])
def test_rgb_selection_and_short_clip_padding(tmp_path, variable_rate, frame_rate, num_frames):
    # Frame-specific pixels expose RGB swaps, off-by-one selection, reordering,
    # timestamp-based resampling, and padding from the wrong selected frame.
    frames = np.random.default_rng(3).integers(0, 256, (23, 24, 32, 3), dtype=np.uint8)
    data = encode(frames, variable_rate=variable_rate)
    path = tmp_path / "clip.mkv"
    path.write_bytes(data)
    with av.open(BytesIO(data)) as container:
        stream = container.streams.best("video")
        assert stream.frames == 0  # This container has no total-frame metadata.
        fps = float(stream.average_rate)
    step = max(1, round(fps / frame_rate)) if frame_rate else 1
    selected = list(frames[::step][:num_frames])
    selected.extend([selected[-1]] * (num_frames - len(selected)))
    for source in (path, data):
        actual = video_decode.decode_video(source, num_frames, frame_rate)
        np.testing.assert_array_equal(actual, np.stack(selected))
        assert actual.dtype == np.uint8


def test_prefix_read_does_not_materialize_entire_source():
    # A long lossless stream makes accidental whole-file reads observable
    # without a wall-clock threshold or an external FFmpeg executable.
    frames = np.random.default_rng(7).integers(0, 256, (120, 96, 160, 3), dtype=np.uint8)
    data = encode(frames)

    class CountedSource(BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            chunk = super().read(size)
            self.bytes_read += len(chunk)
            return chunk

    source = CountedSource(data)
    actual = video_decode.decode_video(source, num_frames=3)
    np.testing.assert_array_equal(actual, frames[:3])
    assert source.bytes_read < len(data) // 2


def test_mpeg_prefix_starts_at_first_frame_despite_nonzero_timestamp():
    # A seek after indexing skipped the first GOP in a real MPEG program
    # stream. Brightness identifies frames without depending on codec bytes.
    output = BytesIO()
    levels = np.arange(36) * 6
    with av.open(output, "w", format="mpeg") as container:
        stream = container.add_stream("mpeg1video", rate=30)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 12
        for level in levels:
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), level, dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    data = output.getvalue()
    with av.open(BytesIO(data)) as container:
        assert container.streams.best("video").start_time > 0
    actual = video_decode.decode_video(data, num_frames=16)
    assert np.max(np.abs(actual[:, 8, 8].astype(int) - levels[:16, None])) <= 3


@pytest.mark.parametrize("matrix,kr,kb", [(1, 0.2126, 0.0722), (5, 0.299, 0.114)])
@pytest.mark.parametrize("full_range", [False, True])
def test_yuv_color_matrix_and_range_are_preserved(matrix, kr, kb, full_range):
    # A constant YUV patch has a known RGB conversion. The defaults previously
    # ignored BT.709 and full-range metadata, changing otherwise valid inputs.
    output = BytesIO()
    with av.open(output, "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.colorspace = matrix
        stream.codec_context.color_range = 2 if full_range else 1
        frame = av.VideoFrame(16, 16, "yuv420p")
        frame.colorspace = matrix
        frame.color_range = stream.codec_context.color_range
        for plane, value in zip(frame.planes, (100, 150, 170)):
            plane.update(bytes([value]) * plane.buffer_size)
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    y, u, v = (100, 22, 42) if full_range else (84 * 255 / 219, 22 * 255 / 224, 42 * 255 / 224)
    expected = np.array([
        y + (2 - 2 * kr) * v,
        y - kb * (2 - 2 * kb) / (1 - kr - kb) * u - kr * (2 - 2 * kr) / (1 - kr - kb) * v,
        y + (2 - 2 * kb) * u,
    ])
    pixels = video_decode.decode_video(output.getvalue(), num_frames=1)
    # swscale uses fixed-point coefficients and integer rounding.
    assert np.max(np.abs(pixels.astype(float) - expected)) < 3


@pytest.mark.parametrize("rotation", [0, 90, -90, 180])
def test_stops_at_last_requested_frame_and_closes_container(monkeypatch, rotation):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    seen = []

    class Container:
        closed = False
        streams = SimpleNamespace(best=lambda _: SimpleNamespace(
            average_rate=24, codec_context=SimpleNamespace(thread_count=0)))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def decode(self, _):
            for index in range(4):
                seen.append(index)
                yield SimpleNamespace(
                    rotation=rotation, colorspace=2, color_range=0,
                    to_ndarray=lambda **_: pixels,
                )
            raise AssertionError("Read beyond the requested prefix")

    container = Container()
    monkeypatch.setattr(video_decode.av, "open", lambda _: container)
    actual = video_decode.decode_video("clip", num_frames=2, frame_rate=8)
    assert seen == [0, 1, 2, 3]
    assert container.closed
    np.testing.assert_array_equal(actual, np.stack([np.rot90(pixels, rotation // 90)] * 2))


def test_decode_error_closes_container_and_skips_sample(monkeypatch):
    class Container:
        closed = False
        streams = SimpleNamespace(best=lambda _: SimpleNamespace(
            average_rate=24, codec_context=SimpleNamespace(thread_count=0)))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def decode(self, _):
            raise ValueError("Truncated video")

    container = Container()
    monkeypatch.setattr(video_decode.av, "open", lambda _: container)
    assert process_video_bytes_sample({"video_bytes": b"invalid"}) == (None, None)
    assert container.closed


@pytest.mark.parametrize("source", [b"", b"invalid container"])
def test_unreadable_media_is_skipped(source):
    assert process_video_bytes_sample({"video_bytes": source}) == (None, None)
