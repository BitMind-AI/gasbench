"""Model inference utilities - supports ONNX and custom PyTorch models."""

import time
import numpy as np
from pathlib import Path
from typing import Tuple

import onnxruntime as ort

from ...logger import get_logger

logger = get_logger(__name__)


def create_inference_session(model_path: str, model_type: str):
    """
    Create inference session - auto-detects ONNX vs custom PyTorch.

    Args:
        model_path: Path to .onnx file OR directory containing model.py + weights
        model_type: Type of model (image, video, audio)

    Returns:
        Inference session (ONNX or PyTorch wrapper)
    """
    model_path = Path(model_path)

    # Check if it's a directory (custom PyTorch model)
    if model_path.is_dir():
        # Look for model_config.yaml to confirm it's a custom model
        if (model_path / "model_config.yaml").exists():
            from .pytorch_session import PyTorchInferenceSession
            logger.info(f"Detected custom PyTorch model in {model_path}")
            return PyTorchInferenceSession(str(model_path), model_type)
        else:
            raise ValueError(f"Directory {model_path} missing model_config.yaml")

    # Check for config file path (yaml/json)
    if model_path.suffix.lower() in ('.yaml', '.yml', '.json'):
        # Config file - load from parent directory
        from .pytorch_session import PyTorchInferenceSession
        logger.info(f"Detected custom PyTorch model config at {model_path}")
        return PyTorchInferenceSession(str(model_path.parent), model_type)

    # Default: ONNX model
    return _create_onnx_session(str(model_path), model_type)


def _create_onnx_session(model_path: str, model_type: str):
    """
    Create and configure ONNX inference session with GPU support.
    
    Args:
        model_path: Path to ONNX model file
        model_type: Type of model ("image", "video", "audio") for logging

    Returns:
        ONNX InferenceSession configured with CUDA/CPU providers
    """
    providers = [
        ("CUDAExecutionProvider", {'device_id': 0}),
        "CPUExecutionProvider"
    ]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    load_message = f"Loading {model_type} detector (ONNX)"
    if model_type == "video":
        load_message += " (this may take 30-60s for large models)"

    logger.info(load_message)
    load_start = time.time()
    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )
    load_time = time.time() - load_start
    logger.info(f"Loaded {model_type} detector in {load_time:.2f} seconds")

    return session


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

