"""Model inference utilities for custom PyTorch models."""

import numpy as np
from pathlib import Path
from typing import Tuple

from ...logger import get_logger

logger = get_logger(__name__)


def create_inference_session(model_path: str, model_type: str):
    """
    Create an inference session for a custom PyTorch model.

    Args:
        model_path: Directory containing model_config.yaml, model.py, and weights,
            or a path to its YAML/JSON config file
        model_type: Type of model (image, video, audio)

    Returns:
        PyTorch inference session
    """
    model_path = Path(model_path)

    if model_path.is_dir():
        if (model_path / "model_config.yaml").exists():
            from .pytorch_session import PyTorchInferenceSession
            logger.info(f"Detected custom PyTorch model in {model_path}")
            return PyTorchInferenceSession(str(model_path), model_type)
        else:
            raise ValueError(f"Directory {model_path} missing model_config.yaml")

    if model_path.suffix.lower() in (".yaml", ".yml", ".json"):
        from .pytorch_session import PyTorchInferenceSession
        logger.info(f"Detected custom PyTorch model config at {model_path}")
        return PyTorchInferenceSession(str(model_path.parent), model_type)

    raise ValueError(
        "Model path must be a custom PyTorch model directory containing "
        "model_config.yaml, model.py, and a weights file"
    )


def process_model_output(logits: np.ndarray) -> Tuple[int, np.ndarray]:
    """Process model output logits into a class prediction and probabilities.

    Handles 4-class, 3-class, 2-class, and single-output model formats.
    The returned class index is argmax (0=real for all multimodal heads).
    Binary SN34 collapse (real vs not-real) happens in Metrics.update.

    Args:
        logits: Raw model output logits (1D array from single sample)

    Returns:
        Tuple of (predicted_class, probabilities)
        - predicted_class: argmax class, or 0/1 for a single-logit head
        - probabilities: sigmoid (len 1) or softmax (len 2/3/4)
    """
    logits = np.atleast_1d(logits).flatten()

    if len(logits) == 1:
        p = 1.0 / (1.0 + np.exp(-logits[0]))
        pred_probs = np.array([p], dtype=np.float64)
        predicted_class = int(p > 0.5)
        return predicted_class, pred_probs

    exp_x = np.exp(logits - np.max(logits))
    pred_probs = exp_x / np.sum(exp_x)
    predicted_class = int(np.argmax(pred_probs))
    return predicted_class, pred_probs
