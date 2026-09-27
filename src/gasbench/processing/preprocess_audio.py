"""Convert cached audio to tensors using a recoverable metadata-index update."""

import json
import os
import tempfile
from pathlib import Path

import torch

from ..dataset.cache import atomic_json, fsync_directory, load_audio_sample
from ..dataset.config import discover_benchmark_audio_datasets
from ..logger import get_logger
from .media import process_audio_sample

logger = get_logger(__name__)


def _preprocess_cache(directory: Path, config) -> bool:
    metadata_path = directory / "sample_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    samples = directory / "samples"
    errors = 0
    for filename in list(metadata):
        if Path(filename).name != filename:
            raise ValueError(f"Invalid cached filename: {filename}")
        if filename.endswith(".pt"):
            if not (samples / filename).is_file():
                errors += 1
                logger.warning("Missing preprocessed audio: %s", samples / filename)
            continue
        source = samples / filename
        if not source.exists():
            source = directory / "originals" / filename
        # Retain the full filename: clip.wav and clip.mp3 are distinct inputs.
        target = samples / f"{filename}.pt"
        pending = None
        try:
            sample = load_audio_sample(
                {
                    **metadata[filename],
                    "audio_path": str(source),
                    "media_type": config.media_type,
                }
            )
            waveform, label = process_audio_sample(sample)
            if waveform is None or label is None:
                raise ValueError("Audio decoding failed")
            with tempfile.NamedTemporaryFile(
                dir=samples, prefix=".pending-", delete=False
            ) as stream:
                pending = Path(stream.name)
                torch.save(
                    {
                        "waveform": waveform.cpu(),
                        "label": label,
                        "metadata": metadata[filename],
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, target)
            fsync_directory(samples)
            replacement = dict(metadata)
            replacement[target.name] = replacement.pop(filename)
            atomic_json(metadata_path, replacement)
            metadata = replacement
            # Once indexed, the tensor is authoritative. Keep the source so a
            # crash, decoder bug or future preprocessing change cannot lose it.
            originals = directory / "originals"
            originals.mkdir(exist_ok=True)
            backup = originals / filename
            if not backup.exists():
                source.rename(backup)
                fsync_directory(originals)
                fsync_directory(samples)
                fsync_directory(directory)
        except Exception as exc:
            errors += 1
            logger.warning("Failed to preprocess %s: %s", source, exc)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    return bool(metadata) and errors == 0


def preprocess_dataset(dataset_name: str, cache_dir: str = "/.cache/gasbench") -> bool:
    """Convert every indexed raw sample; retain originals and existing tensors."""
    config = next(
        (
            d
            for d in discover_benchmark_audio_datasets(mode="full")
            if d.name == dataset_name
        ),
        None,
    )
    if config is None:
        logger.error("Unknown audio dataset: %s", dataset_name)
        return False
    directory = Path(cache_dir) / "datasets" / dataset_name
    indexes = sorted(directory.rglob("sample_metadata.json"))
    if not indexes:
        logger.error("Dataset is not cached: %s", dataset_name)
        return False
    results = [_preprocess_cache(index.parent, config) for index in indexes]
    return all(results)


def preprocess_all_datasets(cache_dir: str = "/.cache/gasbench") -> bool:
    results = [
        preprocess_dataset(d.name, cache_dir)
        for d in discover_benchmark_audio_datasets(mode="full")
    ]
    return bool(results) and all(results)
