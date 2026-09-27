"""Determinism guarantees for the augmentation pipeline.

The prefetch pipelines run augmentation in parallel worker threads. These tests
pin the two properties that must hold for reproducible scoring:
  1. A given (sample, seed) produces the same output regardless of concurrency.
  2. Different seeds produce different augmentations (the seed is actually used).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gasbench.processing.transforms import (
    apply_random_augmentations,
    apply_robustness_augmentations,
)

TARGET = (224, 224)


def _img(seed=0, h=260, w=300):
    return (np.random.RandomState(seed).rand(h, w, 3) * 255).astype(np.uint8)


def test_different_seed_different_output():
    img = _img()
    a, *_ = apply_random_augmentations(
        img.copy(), TARGET, level=3, crop_prob=0.5, seed=1
    )
    b, *_ = apply_random_augmentations(
        img.copy(), TARGET, level=3, crop_prob=0.5, seed=2
    )
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_deterministic_under_thread_pool(level):
    """The core guarantee: seeding is per-call, not global, so concurrent workers
    cannot clobber each other's RNG state."""
    samples = [(_img(seed=i), 1000 + i) for i in range(8)]

    def one(item):
        img, seed = item
        out, *_ = apply_random_augmentations(
            img.copy(), TARGET, level=level, crop_prob=0.5, seed=seed
        )
        return out

    serial = [one(s) for s in samples]
    with ThreadPoolExecutor(max_workers=8) as ex:
        concurrent = list(ex.map(one, samples))

    for i, (s, c) in enumerate(zip(serial, concurrent)):
        assert np.array_equal(s, c), (
            f"sample {i} differed between serial and concurrent runs"
        )


def test_robustness_pass_is_deterministic_and_changes_the_image():
    img = _img()
    a, *_ = apply_robustness_augmentations(img.copy(), TARGET, seed=42)
    b, *_ = apply_robustness_augmentations(img.copy(), TARGET, seed=42)
    assert np.array_equal(a, b)
    base, *_ = apply_random_augmentations(img, TARGET, level=0, crop_prob=0)
    assert a.shape == (*TARGET, 3)
    assert a.dtype == np.uint8
    assert not np.array_equal(a, base)
