from types import SimpleNamespace
from unittest.mock import patch

from src.gasbench.dataset.download.listing import (
    _get_huggingface_urls,
    list_hf_files,
)


def test_fallback_files_do_not_crow_out_preferred_format():
    remote_files = [
        "archives/a.zip",
        "archives/b.zip",
        "videos/a.mp4",
        "videos/b.mp4",
    ]

    with patch(
        "src.gasbench.dataset.download.listing.hf_hub.HfApi.list_repo_tree",
        return_value=(SimpleNamespace(path=path) for path in remote_files),
    ) as list_repo_tree:
        files = list_hf_files(
            "org/repo",
            extension=(".mp4", ".zip"),
            preferred_extension=".mp4",
            max_files=2,
        )

    assert files == remote_files
    list_repo_tree.assert_called_once()


def test_fallback_listing_retains_only_requested_count_per_format():
    remote_files = [*(f"archives/{i}.zip" for i in range(10)), "README.md"]

    with patch(
        "src.gasbench.dataset.download.listing.hf_hub.HfApi.list_repo_tree",
        return_value=(SimpleNamespace(path=path) for path in remote_files),
    ):
        files = list_hf_files(
            "org/repo",
            extension=(".mp4", ".zip"),
            preferred_extension=".mp4",
            max_files=2,
        )

    assert files == ["archives/0.zip", "archives/1.zip"]


def test_hf_listing_targets_pinned_subfolder():
    with patch(
        "src.gasbench.dataset.download.listing.hf_hub.HfApi.list_repo_tree",
        return_value=iter([SimpleNamespace(path="pedestrian/session/video/front.mp4")]),
    ) as list_repo_tree:
        files = list_hf_files(
            "org/repo",
            extension=".mp4",
            revision="abc123",
            subfolders=["pedestrian/session"],
        )

    assert files == ["pedestrian/session/video/front.mp4"]
    list_repo_tree.assert_called_once_with(
        repo_id="org/repo",
        path_in_repo="pedestrian/session",
        recursive=True,
        expand=False,
        revision="abc123",
        repo_type="dataset",
        token=None,
    )


def test_hf_download_url_uses_pinned_revision():
    assert _get_huggingface_urls("org/repo", ["video/a.mp4"], "abc123") == [
        "https://huggingface.co/datasets/org/repo/resolve/abc123/video/a.mp4"
    ]
