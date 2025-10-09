"""ONNX model inference utilities."""

import time
import numpy as np
from typing import Tuple

import onnxruntime as ort

from ..logger import get_logger

logger = get_logger(__name__)


def create_inference_session(model_path: str, model_type: str):
    """
    Create and configure ONNX inference session with GPU support.
    
    Args:
        model_path: Path to ONNX model file
        model_type: Type of model ("image" or "video") for logging

    Returns:
        ONNX InferenceSession configured with CUDA/CPU providers
    """
    providers = [
        ("CUDAExecutionProvider", {'device_id': 0}),
        "CPUExecutionProvider"
    ]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    load_message = f"Loading {model_type} detector"
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


def process_model_output(logits: np.ndarray) -> Tuple[int, int]:
    """
    Process model output logits into binary and multiclass predictions.
    
    Handles 3-class, 2-class, and single-output model formats.
    
    Args:
        logits: Raw model output logits
    
    Returns:
        Tuple of (binary_prediction, multiclass_prediction)
        - binary: 0=real, 1=AI-generated
        - multiclass: 0=real, 1=synthetic, 2=semisynthetic
    """
    exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    if probabilities.shape[-1] == 3:
        pred_probs = probabilities[0]
        predicted_multiclass = int(np.argmax(pred_probs))
        predicted_binary = 0 if predicted_multiclass == 0 else 1
    elif probabilities.shape[-1] == 2:
        pred_probs = probabilities[0]
        predicted_binary = int(np.argmax(pred_probs))
        predicted_multiclass = predicted_binary
    else:
        predicted_binary = int(probabilities.flatten()[0] > 0.5)
        predicted_multiclass = predicted_binary
    
    return predicted_binary, predicted_multiclass

