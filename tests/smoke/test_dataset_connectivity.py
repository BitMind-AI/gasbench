"""Optional live check of the download listing path for one dataset per modality."""

import pytest

from gasbench.dataset.config import load_benchmark_datasets_from_yaml
from gasbench.dataset.download.listing import list_hf_files

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_representative_dataset_has_downloadable_files(modality):
    dataset = next((
        item for item in load_benchmark_datasets_from_yaml()[modality]
        if item.source == "huggingface" and item.source_format
        and item.source_format != "frames" and "gasstation" not in item.name.lower()
    ), None)
    if dataset is None:
        pytest.skip(f"No file-based HF dataset configured for {modality}")
    formats = dataset.source_format
    if isinstance(formats, str):
        formats = [formats]
    files = list_hf_files(
        dataset.path,
        extension=tuple(f".{fmt.lstrip('.')}" for fmt in formats),
        revision=dataset.hf_revision,
        subfolders=dataset.hf_subfolders,
        include_paths=dataset.include_paths,
        exclude_paths=dataset.exclude_paths,
        max_files=1,
    )
    assert files, f"No downloadable files found for {dataset.name} ({dataset.path})"
