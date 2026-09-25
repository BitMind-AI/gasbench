from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.gasbench.dataset.config import (
    BenchmarkDatasetConfig,
    _obfuscate_holdout_names,
    validate_dataset_config,
)
from src.gasbench.dataset.download import core
from src.gasbench.dataset.download.listing import _list_remote_dataset_files
from src.gasbench.dataset.utils import s3_utils


def test_s3_parent_discovers_nested_shards_and_filters_sibling_datasets(monkeypatch):
    keys = ["release/corpus-shard-1/a.jpg", "release/corpus-shard-2/deep/b.png", "release/other/c.jpg"]

    def paginate(**kwargs):
        assert kwargs == {"Bucket": "bucket", "Prefix": "release/"}
        for key in keys:
            yield {"Contents": [{"Key": key}]}

    monkeypatch.setattr(s3_utils, "_get_s3_client", lambda: SimpleNamespace(
        get_paginator=lambda name: SimpleNamespace(paginate=paginate),
    ))
    assert _list_remote_dataset_files(
        "bucket/release", ["jpg", "png"], source="s3",
        include_paths=["release/corpus-"],
    ) == keys[:2]


def test_mixed_formats_download_all_matching_shards(monkeypatch, tmp_path):
    dataset = BenchmarkDatasetConfig(
        name="mixed", path="bucket/corpus", modality="image", media_type="real",
        source="s3", source_format=["jpg", "png"],
    )
    monkeypatch.setattr(core, "_list_remote_dataset_files", lambda *a, **k: [
        "corpus/shard-a/sample.jpg", "corpus/shard-b/sample.png", "corpus/backup.zip",
    ])
    monkeypatch.setattr(core, "_get_download_urls", lambda path, files, *a: files)

    def download(paths, root, **kwargs):
        for key in paths:
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            yield path

    monkeypatch.setattr(core, "_stream_downloads", download)
    monkeypatch.setattr(core, "yield_media_from_source", lambda path, *a: iter([path.suffix]))
    samples = list(core.download_and_extract(
        dataset, media_per_archive=-1, archives_per_dataset=-1,
        temp_dir=str(tmp_path), force_download=True,
    ))
    assert set(samples) == {".jpg", ".png"}
    assert core._calculate_files_to_download(dataset, ["jpg", "png"], 3, 2) == 6


def test_format_order_does_not_change_holdout_identity():
    dataset = BenchmarkDatasetConfig(
        name="mixed", path="bucket/corpus", modality="image", media_type="real",
        source_format=["jpg", "png"],
    )
    def name(fmt):
        return _obfuscate_holdout_names([replace(dataset, source_format=fmt)])[0][0].name

    assert name(["jpg", "png"]) == name(["png", "jpg", "png"])
    assert name("jpg") == name(["jpg"])
    assert name("jpg") != name(["jpg", "png"])


@pytest.mark.parametrize("formats", [[], [None], [""], 42])
def test_invalid_format_lists_are_rejected(formats):
    assert any("source_format" in error for error in validate_dataset_config({
        "name": "bad", "path": "bucket/corpus", "modality": "image",
        "media_type": "real", "source_format": formats,
    }))
