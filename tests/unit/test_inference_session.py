from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_loads_weights_and_runs_real_model_with_native_input_dtype(tmp_path, monkeypatch, modality):
    """Catches config/weight routing, implicit input normalization, and train mode."""
    import sys

    import torch
    import yaml
    from safetensors.torch import save_file

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # The loader registers this module globally; restore it after the test.
    monkeypatch.setitem(sys.modules, "custom_model", None)
    num_classes = 2 if modality == "audio" else 3
    preproc = (
        {"sample_rate": 8, "duration_seconds": 1}
        if modality == "audio" else {"resize": [2, 3], "num_frames": 2}
    )
    input_dtype = "float32" if modality == "audio" else "uint8"
    config = {
        "dtype": "bfloat16" if modality == "video" else "float32",
        "preprocessing": preproc,
        "model": {"num_classes": num_classes, "weights_file": "weights.safetensors",
                  "input_dtype": input_dtype},
    }
    (tmp_path / "model_config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "model.py").write_text('''
import torch
from safetensors.torch import load_file

class Detector(torch.nn.Module):
    def __init__(self, num_classes, input_dtype):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))
        self.input_dtype = input_dtype

    def forward(self, x):
        assert str(x.dtype) == "torch." + self.input_dtype
        assert not self.training and not torch.is_grad_enabled()
        signal = x.float().flatten(1).mean(1, keepdim=True).to(self.bias.dtype)
        return signal + self.bias

def load_model(weights_path, num_classes, input_dtype):
    model = Detector(num_classes, input_dtype)
    model.load_state_dict(load_file(weights_path))
    return model
''')
    bias = torch.arange(num_classes, dtype=torch.float32) / 4
    save_file({"bias": bias}, tmp_path / "weights.safetensors")
    session = create_inference_session(str(tmp_path), modality)
    assert session.model.bias.dtype == (torch.bfloat16 if modality == "video" else torch.float32)
    expected_shape = {"image": [None, 3, 2, 3], "video": [None, 2, 3, 2, 3], "audio": [None, 8]}[modality]
    input_spec = session.get_inputs()[0]
    assert input_spec.shape == expected_shape
    assert session.get_outputs()[0].shape == [None, num_classes]
    values = np.array([0.5, -0.25] if modality == "audio" else [12, 24], dtype=input_dtype)
    batch = np.stack([np.full(expected_shape[1:], v, dtype=input_dtype) for v in values])
    (output,) = session.run(None, {input_spec.name: batch})
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, values[:, None] + bias.numpy())


@pytest.mark.parametrize("logits,expected", [
    ([0.0], [0.5]),
    ([np.log(3)], [0.75]),
    ([1000, 1000 + np.log(3)], [0.25, 0.75]),
    ([1000, 1000, 1000 + np.log(2)], [0.25, 0.25, 0.5]),
])
def test_logits_are_converted_to_probabilities_without_overflow(logits, expected):
    from gasbench.benchmarks.utils.inference import process_model_output

    predicted, probabilities = process_model_output(np.array(logits))
    np.testing.assert_allclose(probabilities, expected)
    assert predicted == (int(expected[0] > 0.5) if len(expected) == 1 else int(np.argmax(expected)))
