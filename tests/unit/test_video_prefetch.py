"""Cached robustness inputs must bypass decoding without bypassing identity checks."""

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gasbench.benchmarks import common, video_bench
from gasbench.benchmarks._checkpoint import CheckpointError
from gasbench.benchmarks.aug_cache import vid_aug_cache_path
from gasbench.benchmarks.recording import build_sample_id
from gasbench.constants import media_type_to_label


class Samples(list):
    config = SimpleNamespace(name="tiny")


def preprocess(sample, aug_dir, **kwargs):
    pipeline = video_bench.VideoPrefetchPipeline(
        Samples([sample]), target_size=(2, 2), batch_size=1, seed=7,
        augment_level=0, crop_prob=0, num_workers=1,
        robustness_pass=kwargs.pop("robustness_pass", True),
        aug_cache_dir=str(aug_dir), **kwargs,
    )
    try:
        return [item for batch in pipeline for item in batch]
    finally:
        pipeline.close()


@pytest.fixture(params=["video_path", "video_frames"])
def sample(request, tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"source")
    return {
        request.param: str(path) if request.param == "video_path" else [str(path)],
        "source_file": path.name, "dataset_name": "tiny", "media_type": "real",
        "file_metadata_sha256": common.file_metadata_digest([path]),
    }


def cache_path(sample, aug_dir):
    return Path(vid_aug_cache_path(str(aug_dir), build_sample_id(sample), (2, 2)))


def install_decoder(monkeypatch):
    def decode(sample, **_):
        return np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3), media_type_to_label(
            sample["media_type"], "video"
        )

    decoder = Mock(side_effect=decode)
    monkeypatch.setattr(video_bench, "process_video_bytes_sample", decoder)
    monkeypatch.setattr(video_bench, "process_video_frames_sample", decoder)
    monkeypatch.setattr(
        video_bench, "apply_video_robustness_augmentations",
        lambda frames, *_, seed: (frames + seed, None, None, None),
    )
    return decoder


@pytest.mark.parametrize("media_type", ["real", "synthetic", "semisynthetic"])
def test_cache_hit_matches_uncached_output_without_source_reads(
    sample, tmp_path, monkeypatch, media_type,
):
    sample["media_type"] = media_type
    aug_dir = tmp_path / "aug"
    decoder = install_decoder(monkeypatch)
    uncached = preprocess(sample, aug_dir)
    assert len(uncached) == 1
    decoder.assert_called_once()
    sample["augmentation_cache_metadata_sha256"] = common.file_metadata_digest(
        [cache_path(sample, aug_dir)]
    )
    decoder.reset_mock()

    original_open = builtins.open
    source_paths = set(common.sample_files(sample))

    def guarded_open(path, *args, **kwargs):
        if Path(path) in source_paths:
            raise AssertionError("Cache hit read source media")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    cached = preprocess(sample, aug_dir, aug_cache_readonly=True)
    assert len(cached) == 1
    decoder.assert_not_called()
    np.testing.assert_array_equal(cached[0]["video"], uncached[0]["video"])
    for key in ("label", "sample_seed", "sample_index", "dataset_name"):
        assert cached[0][key] == uncached[0][key]
    assert build_sample_id(cached[0]["sample"]) == build_sample_id(uncached[0]["sample"])


@pytest.mark.parametrize("changed", ["source", "cache", "missing-cache"])
def test_cache_hit_rejects_changed_frozen_inputs(sample, tmp_path, changed):
    aug_dir = tmp_path / "aug"
    path = cache_path(sample, aug_dir)
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((2, 2, 2, 3), dtype=np.uint8))
    sample["augmentation_cache_metadata_sha256"] = common.file_metadata_digest([path])
    if changed == "source":
        common.sample_files(sample)[0].write_bytes(b"changed")
    elif changed == "cache":
        path.write_bytes(b"changed")
    else:
        path.unlink()
    with pytest.raises(CheckpointError, match="changed|disappeared"):
        preprocess(sample, aug_dir)


@pytest.mark.parametrize("robustness_pass", [False, True])
def test_unselected_cache_cannot_replace_original_input(
    sample, tmp_path, monkeypatch, robustness_pass,
):
    # A cache created after the run's selection was frozen must stay unused.
    aug_dir = tmp_path / "aug"
    path = cache_path(sample, aug_dir)
    path.parent.mkdir(parents=True)
    np.save(path, np.full((2, 2, 2, 3), 200, dtype=np.uint8))
    sample["augmentation_cache_metadata_sha256"] = None
    decoder = install_decoder(monkeypatch)
    monkeypatch.setattr(
        video_bench, "apply_random_augmentations",
        lambda frames, *_, **__: (frames, None, None, None),
    )
    rows = preprocess(sample, aug_dir, robustness_pass=robustness_pass, aug_cache_readonly=True)
    assert len(rows) == 1
    decoder.assert_called_once()
    assert not np.all(rows[0]["video"] == 200)
    np.testing.assert_array_equal(np.load(path), np.full((2, 2, 2, 3), 200, dtype=np.uint8))
