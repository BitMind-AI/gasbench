"""Cache recovery must preserve samples and their source identity."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from gasbench.dataset.cache import cache_state, load_audio_sample
from gasbench.dataset.config import BenchmarkDatasetConfig
from gasbench.dataset.iterator import DatasetIterator
from gasbench.processing import preprocess_audio


def make_cache(tmp_path, names):
    directory = tmp_path / "datasets" / "tiny"
    samples = directory / "samples"
    samples.mkdir(parents=True)
    metadata = {}
    for i, name in enumerate(names):
        (samples / name).write_bytes(bytes([i + 1]))
        metadata[name] = {"source_file": name, "member_path": f"original/{name}"}
    (directory / "sample_metadata.json").write_text(json.dumps(metadata))
    (directory / "dataset_info.json").write_text("{}")
    return directory


def test_audio_conversion_is_recoverable_idempotent_and_preserves_every_source(
    tmp_path, monkeypatch
):
    directory = make_cache(tmp_path, ["clip.wav", "clip.mp3", "last.wav"])
    metadata_path = directory / "sample_metadata.json"
    original = json.loads(metadata_path.read_text())
    config = SimpleNamespace(media_type="real")
    monkeypatch.setattr(
        preprocess_audio,
        "process_audio_sample",
        lambda sample: (
            torch.full((96000,), float(sample["audio_bytes"][0])),
            0,
        ),
    )
    publish = preprocess_audio.atomic_json

    def interrupted(*_):
        raise OSError("interrupted")

    monkeypatch.setattr(preprocess_audio, "atomic_json", interrupted)
    assert not preprocess_audio._preprocess_cache(directory, config)
    assert json.loads(metadata_path.read_text()) == original
    assert all((directory / "samples" / name).exists() for name in original)
    monkeypatch.setattr(preprocess_audio, "atomic_json", publish)
    for _ in range(2):
        assert preprocess_audio._preprocess_cache(directory, config)
        metadata = json.loads(metadata_path.read_text())
        assert len(metadata) == len(original)
        for i, (name, provenance) in enumerate(original.items(), 1):
            tensor = load_audio_sample(
                {
                    "audio_path": str(directory / "samples" / f"{name}.pt"),
                    "media_type": "synthetic",
                }
            )
            assert torch.all(tensor["preprocessed_waveform"] == i)
            assert (
                tensor["label"] == 1
            )  # Current taxonomy takes priority over cached label.
            assert tensor["member_path"] == provenance["member_path"]
            assert metadata[f"{name}.pt"] == provenance
            assert (directory / "originals" / name).read_bytes() == bytes([i])


def test_partial_cache_replays_remote_source_without_duplicating_existing_samples(
    tmp_path, monkeypatch
):
    directory = make_cache(tmp_path, ["aud_000000.wav", "aud_000001.wav"])
    # A missing indexed sample and an unindexed file must not be mistaken for
    # complete data or overwritten when a partial cache is repaired.
    (directory / "samples" / "aud_000001.wav").unlink()
    (directory / "samples" / "aud_000009.wav").write_bytes(b"orphan")
    provenance = {"source_file": "shard.parquet", "source_column": "audio"}
    (directory / "sample_metadata.json").write_text(
        json.dumps(
            {
                "aud_000000.wav": provenance,
                "aud_000001.wav": provenance,
            }
        )
    )
    called = []

    def remote(config, **kwargs):
        called.append(kwargs["force_download"])
        for value in (1, 2, 3):
            yield {**provenance, "audio_bytes": bytes([value]), "media_type": "real"}

    monkeypatch.setattr("gasbench.dataset.iterator.download_and_extract", remote)
    config = BenchmarkDatasetConfig("tiny", "repo", "audio", "real")
    iterator = DatasetIterator(config, cache_dir=str(tmp_path), max_samples=-1, seed=42)
    assert called == [True]
    assert sorted(s["audio_bytes"] for s in iterator) == [b"\x01", b"\x02", b"\x03"]
    assert (directory / "samples" / "aud_000009.wav").read_bytes() == b"orphan"
    assert cache_state(directory)["complete"]
    assert (
        len(list(DatasetIterator(config, cache_dir=str(tmp_path), max_samples=-1))) == 3
    )
    assert len(called) == 1


def test_seeded_selection_does_not_depend_on_directory_order(tmp_path, monkeypatch):
    make_cache(tmp_path, [f"sample_{i}.wav" for i in range(9)])
    config = BenchmarkDatasetConfig("tiny", "repo", "audio", "real")

    def select(limit):
        return [
            s["source_file"]
            for s in DatasetIterator(
                config,
                cache_dir=str(tmp_path),
                download=False,
                max_samples=limit,
                seed=7,
                metadata_only=True,
            )
        ]

    selected = select(3)
    listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: list(reversed(listdir(path))))
    assert select(3) == selected
    assert len(select(-1)) == 9
    assert select(0) == []


@pytest.mark.parametrize("entry", ["clip.mp4", "clip_frames"])
def test_cache_completeness_counts_files_and_frame_directories(tmp_path, entry):
    directory = make_cache(tmp_path, [entry])
    path = directory / "samples" / entry
    if entry.endswith("frames"):
        path.unlink()
        path.mkdir()
    (directory / ".download_complete").touch()
    assert cache_state(directory)["complete"]
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    assert not cache_state(directory)["complete"]


def test_failed_remote_repair_cannot_mark_partial_cache_complete(tmp_path, monkeypatch):
    directory = make_cache(tmp_path, ["aud_000000.wav", "aud_000001.wav"])
    (directory / "samples" / "aud_000001.wav").unlink()
    (directory / ".download_complete").touch()

    def unavailable(*args, **kwargs):
        raise OSError("remote unavailable")

    monkeypatch.setattr(
        "gasbench.dataset.download.core._list_remote_dataset_files", unavailable
    )
    config = BenchmarkDatasetConfig(
        "tiny", "repo", "audio", "real", source_format="wav"
    )
    with pytest.raises(OSError, match="remote unavailable"):
        DatasetIterator(config, cache_dir=str(tmp_path))
    assert not (directory / ".download_complete").exists()
    assert not cache_state(directory)["complete"]
