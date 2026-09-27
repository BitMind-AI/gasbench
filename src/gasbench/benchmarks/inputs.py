"""Effective preprocessing contracts shared by planning and execution."""

import math

from ..constants import AUDIO_DURATION_SECONDS, AUDIO_SAMPLE_RATE, MAX_VIDEO_NUM_FRAMES
from ..processing.transforms import extract_num_frames_from_input_specs


def validate_audio_preprocessing(preprocessing):
    if (
        preprocessing.get("sample_rate", AUDIO_SAMPLE_RATE) != AUDIO_SAMPLE_RATE
        or preprocessing.get("duration_seconds", AUDIO_DURATION_SECONDS)
        != AUDIO_DURATION_SECONDS
    ):
        raise ValueError(
            "Audio models must accept mono 16 kHz audio for six seconds (96000 samples)"
        )


def video_preprocessing(session, input_specs):
    config = (
        session.get_preprocessing_config()
        if hasattr(session, "get_preprocessing_config")
        else {}
    )
    frames = extract_num_frames_from_input_specs(input_specs)
    frames = config.get("num_frames", 16) if frames is None else frames
    if type(frames) is not int or not 1 <= frames <= MAX_VIDEO_NUM_FRAMES:
        raise ValueError(
            f"Video frame count must be between 1 and {MAX_VIDEO_NUM_FRAMES}"
        )
    rate = config.get("frame_rate")
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError("Video frame_rate must be positive and finite")
    return {"num_frames": frames, "frame_rate": rate}
