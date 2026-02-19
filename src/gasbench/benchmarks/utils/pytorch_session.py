"""PyTorch inference session with ONNX-compatible interface."""

import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Any, Optional

from .custom_model_loader import load_custom_model, load_model_config
from ...logger import get_logger

logger = get_logger(__name__)


class InputSpec:
    """Mock ONNX InputSpec for interface compatibility."""
    
    def __init__(self, name: str, shape: List, type: str):
        self.name = name
        self.shape = shape
        self.type = type


class OutputSpec:
    """Mock ONNX OutputSpec for interface compatibility."""
    
    def __init__(self, name: str, shape: List):
        self.name = name
        self.shape = shape


class PyTorchInferenceSession:
    """
    Wrapper providing ONNX-like interface for PyTorch models.

    This allows custom PyTorch models to be used interchangeably with
    ONNX models in the benchmark code.
    """

    def __init__(self, model_dir: str, model_type: str):
        """
        Initialize PyTorch inference session.

        Args:
            model_dir: Path to directory containing model_config.yaml, model.py, weights
            model_type: Type of model (image, video, audio) for logging
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type

        load_message = f"Loading {model_type} detector (PyTorch custom model)"
        logger.info(load_message)
        load_start = time.time()

        # Load model and config
        self.model, self.config = load_custom_model(self.model_dir)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        load_time = time.time() - load_start
        logger.info(f"Loaded {model_type} detector in {load_time:.2f} seconds")

        # Log device info
        if self.device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")

        # Setup input/output specs
        self._input_name = "input"
        self._input_shape = self._infer_input_shape()
        self._num_classes = self.config.get("model", {}).get("num_classes", 2)

    def _infer_input_shape(self) -> List:
        """Infer input shape from preprocessing config."""
        preproc = self.config.get("preprocessing", {})
        
        # For audio models
        if "sample_rate" in preproc and "duration_seconds" in preproc:
            sr = preproc["sample_rate"]
            duration = preproc["duration_seconds"]
            samples = int(sr * duration)
            return [None, samples]  # Batch, Samples
        
        # For image/video models with resize config
        if "resize" in preproc:
            h, w = preproc["resize"]
            if self.model_type == "video":
                num_frames = preproc.get("num_frames", preproc.get("max_frames", 16))
                return [None, num_frames, 3, h, w]  # NTCHW format
            return [None, 3, h, w]  # NCHW format

        # Defaults
        if self.model_type == "video":
            return [None, 16, 3, 224, 224]
        return [None, 3, 224, 224]

    def get_preprocessing_config(self) -> dict:
        """Return the preprocessing section of model_config.yaml."""
        return self.config.get("preprocessing", {})

    def get_inputs(self) -> List[InputSpec]:
        """Return input specifications (ONNX-compatible interface)."""
        return [InputSpec(name=self._input_name, shape=self._input_shape, type="float32")]

    def get_outputs(self) -> List[OutputSpec]:
        """Return output specifications (ONNX-compatible interface)."""
        return [OutputSpec(name="output", shape=[None, self._num_classes])]

    def get_providers(self) -> List[str]:
        """Return execution providers (ONNX-compatible interface)."""
        if self.device.type == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, input_dict: dict) -> List[np.ndarray]:
        """
        Run inference (ONNX-compatible interface).

        Args:
            output_names: Ignored (for ONNX compatibility)
            input_dict: Dict mapping input name to numpy array

        Returns:
            List containing output numpy array
        """
        data = list(input_dict.values())[0]
        
        # Convert to tensor and move to device
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(self.device)
        else:
            tensor = data.to(self.device)
        
        # Ensure float32
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        with torch.no_grad():
            output = self.model(tensor)

        # Handle various output formats
        if hasattr(output, "logits"):
            output = output.logits  # HuggingFace models
        elif isinstance(output, tuple):
            output = output[0]  # Some models return tuples
        elif isinstance(output, dict):
            # Some models return dicts with 'logits' key
            output = output.get("logits", output.get("output", list(output.values())[0]))

        return [output.cpu().numpy()]
