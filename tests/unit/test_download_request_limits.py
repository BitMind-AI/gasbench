from unittest.mock import patch

from src.gasbench.dataset.download.listing import list_hf_files


def test_fallback_files_do_not_crow_out_preferred_format():
    remote_files = [
        "archives/a.zip",
        "archives/b.zip",
        "videos/a.mp4",
        "videos/b.mp4",
    ]

    with patch(
        "src.gasbench.dataset.download.listing.hf_hub.list_repo_files",
        return_value=iter(remote_files),
    ) as list_repo_files:
        files = list_hf_files(
            "org/repo",
            extension=(".mp4", ".zip"),
            preferred_extension=".mp4",
            max_files=2,
        )

    assert files == remote_files
    list_repo_files.assert_called_once()


def test_fallback_listing_retains_only_requested_count_per_format():
    remote_files = [*(f"archives/{i}.zip" for i in range(10)), "README.md"]

    with patch(
        "src.gasbench.dataset.download.listing.hf_hub.list_repo_files",
        return_value=iter(remote_files),
    ):
        files = list_hf_files(
            "org/repo",
            extension=(".mp4", ".zip"),
            preferred_extension=".mp4",
            max_files=2,
        )

    assert files == ["archives/0.zip", "archives/1.zip"]
