"""Small, explicit network smoke test for representative HF datasets."""

import pytest
from huggingface_hub import list_repo_files
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from src.gasbench.dataset.config import load_benchmark_datasets_from_yaml


pytestmark = pytest.mark.slow

REPRESENTATIVE_DATASETS = (
    ("image", "pica-100k"),
    ("video", "vap-data"),
    ("audio", "deepfake-urdu-real"),
)


@pytest.mark.parametrize("modality,dataset_name", REPRESENTATIVE_DATASETS)
def test_representative_dataset_is_accessible_and_has_expected_files(
    modality, dataset_name
):
    configs = load_benchmark_datasets_from_yaml()
    dataset = next(
        (item for item in configs[modality] if item.name == dataset_name), None
    )
    assert dataset is not None, (
        f"representative dataset {dataset_name} is not registered"
    )

    try:
        files = list(list_repo_files(dataset.path, repo_type="dataset"))
    except GatedRepoError:
        pytest.skip(f"dataset {dataset.path} requires authentication")
    except RepositoryNotFoundError:
        pytest.fail(f"dataset {dataset.path} does not exist")

    assert files, f"dataset {dataset.path} contains no files"
    if dataset.source_format and dataset.source_format != "frames":
        extension = f".{dataset.source_format.lstrip('.')}"
        assert any(filename.lower().endswith(extension) for filename in files), (
            f"dataset {dataset.path} has no {extension} files"
        )
