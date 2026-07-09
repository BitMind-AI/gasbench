"""PyTorch inference session with ONNX-compatible interface."""

import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Any, Optional

from .custom_model_loader import load_custom_model, load_model_config
from ...logger import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


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

        # Resolve dtype from config (top-level 'dtype' key)
        dtype_str = self.config.get("dtype", "float32")
        self.dtype = _DTYPE_MAP.get(str(dtype_str).lower())
        if self.dtype is None:
            raise ValueError(
                f"Unsupported dtype '{dtype_str}' in model_config.yaml. "
                f"Valid options: {list(_DTYPE_MAP.keys())}"
            )

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=self.device, dtype=self.dtype).eval()

        load_time = time.time() - load_start
        logger.info(f"Loaded {model_type} detector in {load_time:.2f} seconds")

        # Log device and dtype info
        if self.device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")
        logger.info(f"Inference dtype: {self.dtype}")

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
        
        # Deliver uint8 tensor to device — models cast and normalise in forward().
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(self.device)
        else:
            tensor = data.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        if hasattr(output, "logits"):
            output = output.logits
        elif isinstance(output, dict):
            if "logits" in output:
                output = output["logits"]
            elif "output" in output:
                output = output["output"]
            elif len(output) == 1:
                output = next(iter(output.values()))
            else:
                raise ValueError(
                    f"Model returned a dict without a 'logits' or 'output' key; "
                    f"cannot determine which of {sorted(output.keys())} is the logits tensor. "
                    f"Return a logits tensor or key it as 'logits'."
                )
        elif isinstance(output, (tuple, list)):
            if len(output) == 1:
                output = output[0]
            else:
                raise ValueError(
                    f"Model returned a {type(output).__name__} of length {len(output)}; "
                    f"cannot determine which element is the logits tensor. "
                    f"Return a logits tensor rather than a {type(output).__name__}."
                )

        last_dim = output.shape[-1]
        if last_dim not in (1, 2, 3) and last_dim != self._num_classes:
            raise ValueError(
                f"Model output last dimension is {last_dim}, expected 1, 2, 3, "
                f"or num_classes ({self._num_classes}). Check that model.py returns "
                f"logits and that model_config.yaml declares the correct num_classes."
            )

        return [output.cpu().to(torch.float32).numpy()]
