from pathlib import Path
from urllib.parse import quote

from gasbench.dataset.download import fetch


def test_huggingface_url_parser_accepts_a_root_level_file():
    assert fetch._parse_huggingface_dataset_url(
        "https://huggingface.co/datasets/example/repo/resolve/main/data.zip"
    ) == ("example/repo", "main", "data.zip")


def test_huggingface_download_preserves_selected_file_and_revision(
    tmp_path, monkeypatch
):
    revision = "feature/data v2"
    filename = "nested folder/chosen shard.zip"
    url = (
        "https://huggingface.co/datasets/example/huge-dataset/resolve/"
        f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
    )
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        downloaded = Path(kwargs["local_dir"]) / kwargs["filename"]
        downloaded.parent.mkdir(parents=True)
        downloaded.write_bytes(b"selected shard")
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    result = fetch.download_single_file(url, tmp_path, chunk_size=8192, hf_token="token")

    assert result is not None
    assert result.read_bytes() == b"selected shard"
    assert len(calls) == 1
    assert calls[0] == {
        "repo_id": "example/huge-dataset",
        "filename": filename,
        "repo_type": "dataset",
        "revision": revision,
        "token": "token",
        "local_dir": str(tmp_path),
    }


def test_non_huggingface_urls_do_not_use_hub_download(tmp_path, monkeypatch):
    hub_calls = []

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: hub_calls.append(kwargs),
    )

    assert fetch._parse_huggingface_dataset_url(
        "https://example.com/datasets/example/repo/resolve/main/file.zip"
    ) is None
    assert hub_calls == []
