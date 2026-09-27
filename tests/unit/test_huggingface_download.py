from pathlib import Path
from urllib.parse import quote
from unittest.mock import MagicMock, Mock

import pytest

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


def response(status, data):
    result = MagicMock(status_code=status, headers={"content-length": str(len(data))})
    result.__enter__.return_value = result
    result.iter_content.return_value = iter([data])
    return result


@pytest.mark.parametrize("host", ["example.com", "huggingface.co"])
def test_http_routing_and_hub_failure_fallback(tmp_path, monkeypatch, host):
    url = f"https://{host}/datasets/example/repo/resolve/main/file.zip"
    hub_download = Mock(side_effect=OSError("hub unavailable"))
    http_get = Mock(return_value=response(200, b"downloaded content"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub_download)
    monkeypatch.setattr(fetch.requests, "get", http_get)

    result = fetch.download_single_file(url, tmp_path, chunk_size=8192)

    assert result.read_bytes() == b"downloaded content"
    assert hub_download.call_count == (1 if host == "huggingface.co" else 0)
    assert http_get.call_args.args == (url,)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("resume_status", [200, 206, 416])
def test_interrupted_http_download_resumes_or_restarts_without_duplicate_bytes(
    tmp_path, monkeypatch, resume_status,
):
    def interrupted_stream():
        yield b"abc"
        raise fetch.requests.ConnectionError("interrupted")

    first = response(200, b"abcdef")
    first.iter_content.return_value = interrupted_stream()
    resumed = response(resume_status, b"def" if resume_status == 206 else b"abcdef")
    responses = iter([first, resumed, response(200, b"abcdef")])
    headers = []

    def get(url, **kwargs):
        headers.append(kwargs["headers"].copy())
        return next(responses)

    monkeypatch.setattr(fetch.requests, "get", get)
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    result = fetch.download_single_file("https://example.com/file.zip", tmp_path, chunk_size=8192)

    assert result.read_bytes() == b"abcdef"
    assert headers[:2] == [{}, {"Range": "bytes=3-"}]
    if resume_status == 416:
        assert headers[2] == {}
    assert not list(tmp_path.glob("*.partial"))
