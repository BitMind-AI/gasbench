import logging
import os
from typing import Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_IMG_AUG_VERSION = "img_v1"
# Decoder changes must not reuse tensors produced by the previous backend.
_VID_AUG_VERSION = "vid_v3"
# Includes the fixed mono 16 kHz / six-second audio preprocessing contract.
_AUD_AUG_VERSION = "aud_v1"


def img_aug_cache_path(
    cache_dir: str, sample_id: str, target_size: Tuple[int, int]
) -> str:
    H, W = target_size
    return os.path.join(
        cache_dir, "img", sample_id[:2], f"{sample_id}_{_IMG_AUG_VERSION}_{H}x{W}.npy"
    )


def vid_aug_cache_path(
    cache_dir: str,
    sample_id: str,
    target_size: Tuple[int, int],
    num_frames: int = 16,
    frame_rate: float = None,
) -> str:
    H, W = target_size
    rate = "native" if frame_rate is None else str(float(frame_rate))
    return os.path.join(
        cache_dir,
        "vid",
        sample_id[:2],
        f"{sample_id}_{_VID_AUG_VERSION}_{H}x{W}_t{num_frames}_fps{rate}.npy",
    )


def aud_aug_cache_path(cache_dir: str, sample_id: str, seed: int) -> str:
    # Noise depends on the sample seed, so a different run seed cannot reuse it.
    return os.path.join(
        cache_dir,
        "aud",
        sample_id[:2],
        f"{sample_id}_{_AUD_AUG_VERSION}_seed{seed}.npy",
    )


def write_aug_cache(path: str, array: np.ndarray) -> None:
    """Atomically write an augmented array to the cache via temp-file + os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import tempfile

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(path), prefix=".pending-", delete=False
        ) as stream:
            tmp = stream.name
            np.save(stream, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
