"""Compatibility adapters to the shared dataset cache reader."""

from pathlib import Path

from ..cache import cache_state


def _is_dataset_cached(dataset, cache_dir="/.cache/gasbench"):
    return cache_state(Path(cache_dir) / "datasets" / dataset.name)["complete"]


def _load_dataset_from_cache(dataset, cache_dir="/.cache/gasbench"):
    # Import at call time: the iterator also owns remote download orchestration.
    from io import BytesIO
    from PIL import Image
    from ..iterator import DatasetIterator

    for sample in DatasetIterator(dataset, cache_dir=cache_dir, max_samples=-1, download=False):
        if dataset.modality == "image":
            image = Image.open(BytesIO(sample["image"]))
            image.load()
            sample = {**sample, "image": image}
        yield sample
