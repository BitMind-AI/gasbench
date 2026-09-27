import numpy as np

from gasbench.benchmarks.frame_cache import (
    cache_covers_prefix,
    frame_indices,
    load_frame_cache,
    select_cached_frames,
    write_frame_cache,
)
from gasbench.constants import MAX_VIDEO_NUM_FRAMES


def test_leading_frames_match_a_model_that_takes_the_first_n():
    stored = np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)
    selected = select_cached_frames(stored, total_frames=30, fps=24.0, num_frames=4, frame_rate=None)
    assert selected.shape == (4, 2, 2, 3)
    assert np.array_equal(selected, stored[:4])


def test_short_clip_repeats_its_last_real_frame():
    stored = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    selected = select_cached_frames(stored, total_frames=3, fps=24.0, num_frames=5, frame_rate=None)
    assert selected.shape == (5, 2, 2, 3)
    assert np.array_equal(selected[-1], stored[-1])


def test_frame_rate_misses_when_the_prefix_is_too_short():
    stored = np.zeros((8, 2, 2, 3), dtype=np.uint8)
    indices = frame_indices(total_frames=300, fps=30.0, num_frames=16, frame_rate=1.0)
    assert max(indices) >= stored.shape[0]
    assert select_cached_frames(stored, 300, 30.0, 16, 1.0) is None


def test_missing_fps_uses_the_same_30fps_fallback_as_live_decode():
    assert frame_indices(90, None, 3, 10.0) == frame_indices(90, 30.0, 3, 10.0)


def test_roundtrip_keeps_source_order_and_a_partial_cache_is_incomplete(tmp_path):
    frames = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    write_frame_cache(str(tmp_path), "abcdef", frames, total_frames=40, fps=None)
    loaded, total, fps = load_frame_cache(str(tmp_path), "abcdef")
    assert total == 40 and fps is None
    assert np.array_equal(loaded, frames)
    assert not cache_covers_prefix(str(tmp_path), "abcdef", max_frames=MAX_VIDEO_NUM_FRAMES)
    assert cache_covers_prefix(str(tmp_path), "abcdef", max_frames=4)
