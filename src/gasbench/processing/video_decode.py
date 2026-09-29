"""Decode a sampled video prefix without indexing or copying the entire file."""

import math
from io import BytesIO

import av
import numpy as np
from av.video.reformatter import Colorspace


def decode_video(source, num_frames=16, frame_rate=None):
    """Return RGB uint8 THWC frames from a path, bytes, or seekable stream.

    Sampling retains the existing frame-index stride based on average FPS,
    including for variable-rate videos. Short clips repeat the last selected
    frame. Container metadata may require seeking, but decoding stops as soon
    as the requested frames have been collected; no frame-count scan is needed.
    """
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    if frame_rate is not None and (not math.isfinite(frame_rate) or frame_rate <= 0):
        raise ValueError("frame_rate must be positive and finite")
    if isinstance(source, bytes):
        source = BytesIO(source)

    with av.open(source) as container:
        # Match FFmpeg's stream selection rather than selecting a thumbnail or
        # preview track solely because it appears first in the container.
        stream = container.streams.best("video")
        if stream is None:
            raise ValueError("No video stream")
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or 30)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        step = 1 if frame_rate is None else max(1, round(fps / frame_rate))
        frames = []
        for index, frame in enumerate(container.decode(stream)):
            if index % step:
                continue
            # Apply the source matrix/range explicitly. PyAV's default matching
            # source/destination matrices can otherwise bypass color conversion
            # setup and treat BT.709 as BT.601. RGB output has no YUV range.
            pixels = frame.to_ndarray(
                format="rgb24", src_colorspace=frame.colorspace,
                dst_colorspace=Colorspace.DEFAULT,
                src_color_range=frame.color_range,
            )
            # PyAV exposes display rotation but does not apply it. Preserve the
            # previous decoder's handling of quarter-turn display metadata.
            rotation = frame.rotation % 360
            if rotation in (90, 180, 270):
                pixels = np.rot90(pixels, rotation // 90)
            frames.append(pixels)
            if len(frames) == num_frames:
                break

    if not frames:
        raise ValueError("No frames decoded")
    frames.extend([frames[-1]] * (num_frames - len(frames)))
    return np.stack(frames)
