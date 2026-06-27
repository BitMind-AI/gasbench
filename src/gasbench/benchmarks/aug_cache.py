import logging
import os
from typing import Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_IMG_AUG_VERSION = "img_v1"
_VID_AUG_VERSION = "vid_v1"


def img_aug_cache_path(cache_dir: str, sample_id: str, target_size: Tuple[int, int]) -> str:
    H, W = target_size
    return os.path.join(cache_dir, "img", sample_id[:2], f"{sample_id}_{_IMG_AUG_VERSION}_{H}x{W}.npy")


def vid_aug_cache_path(cache_dir: str, sample_id: str, target_size: Tuple[int, int]) -> str:
    H, W = target_size
    return os.path.join(cache_dir, "vid", sample_id[:2], f"{sample_id}_{_VID_AUG_VERSION}_{H}x{W}.npy")


def write_aug_cache(path: str, array: np.ndarray) -> None:
    """Atomically write an augmented array to the cache via temp-file + os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
    except Exception as e:
        _logger.error("write_aug_cache failed for %s: %s", path, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
