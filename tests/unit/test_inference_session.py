from pathlib import Path

import pytest

from gasbench.benchmarks.utils.inference import create_inference_session


def test_rejects_model_file(tmp_path: Path):
    model_file = tmp_path / "model.bin"
    model_file.touch()

    with pytest.raises(ValueError, match="custom PyTorch model directory"):
        create_inference_session(str(model_file), "image")


def test_rejects_directory_without_config(tmp_path: Path):
    with pytest.raises(ValueError, match="missing model_config.yaml"):
        create_inference_session(str(tmp_path), "image")
