from types import SimpleNamespace

import pytest

from src.gasbench.dataset.download.listing import _list_remote_dataset_files
from src.gasbench.dataset.utils import s3_utils


def mock_pages(monkeypatch, pages):
    calls = []
    def paginate(**kwargs):
        prefix = kwargs["Prefix"]
        calls.append(prefix)
        yield from pages(prefix)
    monkeypatch.setattr(s3_utils, "_get_s3_client", lambda: SimpleNamespace(
        get_paginator=lambda _: SimpleNamespace(paginate=paginate)))
    return calls


def test_prefixes_keep_partial_shard_names_deduplicate_and_stop(monkeypatch):
    def pages(prefix):
        yield {"Contents": [{"Key": prefix + "001/no.txt"},
                            {"Key": prefix + "001/excluded.jpg"},
                            {"Key": prefix + "002/keep.PNG"}]}
        raise AssertionError("Fetched a page beyond the matching limit")
    calls = mock_pages(monkeypatch, pages)
    assert _list_remote_dataset_files(
        "bucket/root", ["jpg", "png"], source="s3",
        s3_prefixes=["root/corpus-", "root/corpus-001/", "root/later-"],
        exclude_paths=["excluded"], max_files=1,
    ) == ["root/corpus-002/keep.PNG"]
    assert calls == ["root/corpus-"]


def test_substring_filters_still_match_nested_keys(monkeypatch):
    mock_pages(monkeypatch, lambda _: iter([
        {"Contents": [{"Key": "root/other/no.jpg"}]},
        {"Contents": [{"Key": "root/deep/corpus/a.jpg"}]},
    ]))
    assert _list_remote_dataset_files(
        "bucket/root", "jpg", source="s3", include_paths=["corpus"], max_files=1
    ) == ["root/deep/corpus/a.jpg"]


def test_week_filter_runs_before_limit(monkeypatch):
    mock_pages(monkeypatch, lambda _: iter([
        {"Contents": [{"Key": "gasstation/old/a.jpg"}]},
        {"Contents": [{"Key": "gasstation/desired/b.jpg"}]},
    ]))
    assert _list_remote_dataset_files(
        "bucket/gasstation", "jpg", source="s3", target_week="desired", max_files=1,
    ) == ["gasstation/desired/b.jpg"]


@pytest.mark.parametrize("prefixes", [[], ["outside/"], ["root-other/"], [""]])
def test_prefix_cannot_escape_dataset_path(prefixes):
    with pytest.raises(ValueError, match="prefixes"):
        s3_utils.list_s3_files("bucket/root", prefixes=prefixes)


def test_prefix_config_is_loaded_and_changes_cache_identity():
    from src.gasbench.dataset.config import _dataset_dict_to_config, _obfuscate_holdout_names
    row = {"name": "corpus", "path": "bucket/root", "source": "s3",
           "modality": "image", "media_type": "real", "source_format": "jpg"}
    original = _dataset_dict_to_config(row)
    narrowed = _dataset_dict_to_config({**row, "s3_prefixes": ["root/corpus-"]})
    assert narrowed.s3_prefixes == ["root/corpus-"]
    before, _ = _obfuscate_holdout_names([original])
    after, _ = _obfuscate_holdout_names([narrowed])
    assert before[0].name != after[0].name
