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
    """
    Process model output logits into a binary prediction and probabilities.
    
    Handles 3-class, 2-class, and single-output model formats.
    For 3-class models, synthetic and semisynthetic are conflated into AI-generated.
    
    Args:
        logits: Raw model output logits (1D array from single sample)
    
    Returns:
        Tuple of (binary_prediction, probabilities)
        - binary_prediction: 0=real, 1=AI-generated
        - probabilities: softmax probabilities as 1D array (length 1,2, or 3)
    """
    logits = np.atleast_1d(logits).flatten()
    exp_x = np.exp(logits - np.max(logits))
    pred_probs = exp_x / np.sum(exp_x)
    
    if len(pred_probs) == 3:
        predicted_binary = 0 if np.argmax(pred_probs) == 0 else 1
    elif len(pred_probs) == 2:
        predicted_binary = int(np.argmax(pred_probs))
    else:
        predicted_binary = int(pred_probs[0] > 0.5)
    
    return predicted_binary, pred_probs

