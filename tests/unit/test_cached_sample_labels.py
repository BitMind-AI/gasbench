import json

import pytest

from gasbench.constants import media_type_to_label
from gasbench.dataset.config import BenchmarkDatasetConfig
from gasbench.dataset.download.cache_io import _load_dataset_from_cache
from gasbench.dataset.iterator import DatasetIterator


@pytest.mark.parametrize("loader", ["iterator", "download"])
@pytest.mark.parametrize("payload", ["bytes", "frames", "lazy"])
def test_current_video_label_overrides_stale_cache(tmp_path, loader, payload):
    config = BenchmarkDatasetConfig(
        name="example", path="owner/repo", modality="video", media_type="real",
        source_format="mp4", media_per_archive=-1, archives_per_dataset=-1,
    )
    directory = tmp_path / "datasets" / config.name
    samples = directory / "samples"
    samples.mkdir(parents=True)
    filename = "clip" if payload == "frames" else "clip.mp4"
    if payload == "frames":
        (samples / filename).mkdir()
        (samples / filename / "frame.jpg").write_bytes(b"frame")
    else:
        (samples / filename).write_bytes(b"video")
    (directory / "sample_metadata.json").write_text(json.dumps({filename: {
        "media_type": "rendered", "source_file": "original.mp4",
    }}))
    (directory / "dataset_info.json").write_text("{}")
    if loader == "iterator":
        iterator = DatasetIterator(config, cache_dir=str(tmp_path), download=False,
                                   lazy_read=payload == "lazy")
        result = list(iterator)
    else:
        result = list(_load_dataset_from_cache(config, str(tmp_path)))
    assert len(result) == 1
    assert media_type_to_label(result[0]["media_type"], "video") == 0
    assert result[0]["source_file"] == "original.mp4"
    assert json.loads((directory / "sample_metadata.json").read_text())[filename]["media_type"] == "rendered"
