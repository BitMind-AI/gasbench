import json

from gasbench.dataset.config import BenchmarkDatasetConfig
from gasbench.dataset.iterator import CACHE_MAX_SAMPLES
from gasbench.download_manager import _is_dataset_cached_for_mode


def _dataset() -> BenchmarkDatasetConfig:
    return BenchmarkDatasetConfig(
        name="example",
        path="owner/repo",
        modality="audio",
        media_type="real",
        source_format="wav",
        media_per_archive=-1,
        archives_per_dataset=-1,
    )


def _write_metadata(dataset_dir, count: int) -> None:
    dataset_dir.mkdir()
    metadata = {f"aud_{index:06d}.wav": {} for index in range(count)}
    (dataset_dir / "sample_metadata.json").write_text(json.dumps(metadata))


def test_completion_marker_marks_exhausted_source_complete(tmp_path):
    dataset_dir = tmp_path / "example"
    _write_metadata(dataset_dir, 12)
    (dataset_dir / ".download_complete").write_text("{}")

    assert _is_dataset_cached_for_mode(dataset_dir, _dataset())


def test_iterator_sample_cap_marks_cache_complete(tmp_path):
    dataset_dir = tmp_path / "example"
    _write_metadata(dataset_dir, CACHE_MAX_SAMPLES)

    assert _is_dataset_cached_for_mode(dataset_dir, _dataset())


def test_partial_cache_below_iterator_cap_remains_incomplete(tmp_path):
    dataset_dir = tmp_path / "example"
    _write_metadata(dataset_dir, CACHE_MAX_SAMPLES - 1)

    assert not _is_dataset_cached_for_mode(dataset_dir, _dataset())
