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

        # Embedding extraction via self.classifier(s) pre-hook.
        # Supports single-head models (self.classifier) and ensembles
        # (self.classifiers as nn.ModuleList). Captures each head's input
        # and concatenates them into one embedding.
        self._captured_embeddings: List[torch.Tensor] = []
        self._embeddings_available = False
        self._embedding_dim: Optional[int] = None
        self._setup_embedding_hook()

        # Setup input/output specs
        self._input_name = "input"
        self._input_shape = self._infer_input_shape()
        self._num_classes = self.config.get("model", {}).get("num_classes", 2)

    def _setup_embedding_hook(self):
        """Register forward pre-hooks on all classification heads.

        Supports single-head models (one nn.Linear), ensemble models
        (self.classifiers as nn.ModuleList), and heuristic fallback.
        Hooks all heads and concatenates their input embeddings.
        """
        n_classes = self.config.get("model", {}).get("num_classes", 2)

        heads = self._find_classifier_explicit(n_classes)
        if heads is None:
            heads = self._find_classifier_heuristic(n_classes)

        if not heads:
            logger.warning(
                "No classification head(s) found. Model should expose a "
                "self.classifier of type nn.Linear for embedding extraction. "
                "Generalization coefficient will be set to the floor value."
            )
            return

        self._embedding_dim = sum(h["in_features"] for h in heads)

        # Clear accumulated embeddings before each forward pass
        self.model.register_forward_pre_hook(
            lambda _m, _inp: self._captured_embeddings.clear()
        )

        for h in heads:
            target = h["module"]
            target.register_forward_pre_hook(
                lambda _mod, inp, _t=target: self._captured_embeddings.append(
                    inp[0].detach().cpu()
                )
            )

        self._embeddings_available = True
        sources = ", ".join(h["source"] for h in heads)
        logger.info(
            f"Embedding extraction enabled via {len(heads)} head(s): "
            f"{sources}. Total embedding dim: {self._embedding_dim}"
        )

    def _find_classifier_explicit(self, n_classes: int) -> Optional[list]:
        """Explicit: self.classifiers (ModuleList) or self.classifier (single)."""
        classifiers = getattr(self.model, "classifiers", None)
        if isinstance(classifiers, torch.nn.ModuleList):
            heads = []
            for i, head in enumerate(classifiers):
                if isinstance(head, torch.nn.Linear) and head.out_features == n_classes:
                    heads.append(dict(module=head, in_features=head.in_features,
                                      source=f"self.classifiers[{i}] (explicit)"))
            if heads:
                return heads
        head = getattr(self.model, "classifier", None)
        if isinstance(head, torch.nn.Linear) and head.out_features == n_classes:
            return [dict(module=head, in_features=head.in_features,
                         source="self.classifier (explicit)")]
        return None

    def _find_classifier_heuristic(self, n_classes: int) -> Optional[list]:
        """Fallback: all nn.Linear(out_features==n_classes) at max depth."""
        cands: list[dict] = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == n_classes:
                cands.append(dict(module=module, in_features=module.in_features,
                                  depth=name.count("."), name=name))
        if not cands:
            return None
        max_depth = max(c["depth"] for c in cands)
        deepest = [c for c in cands if c["depth"] == max_depth]
        logger.warning(
            f"Model has no self.classifier — using heuristic fallback: "
            f"{len(deepest)} head(s) at depth {max_depth}: "
            f"{', '.join(c['name'] for c in deepest)}. "
            f"Miners should add an explicit self.classifier(s)."
        )
        return [dict(module=c["module"], in_features=c["in_features"],
                     source=f"{c['name']} (heuristic)") for c in deepest]

    @property
    def embeddings_available(self) -> bool:
        return self._embeddings_available

    def get_last_embedding(self) -> Optional[np.ndarray]:
        """Return concatenated embeddings from all classification heads.

        Returns a numpy array of shape (batch_size, total_embedding_dim),
        or None if embeddings are not available or not yet captured.
        """
        if not self._captured_embeddings:
            return None
        return torch.cat(self._captured_embeddings, dim=1).to(torch.float32).numpy()

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

        # Handle various output formats
        if hasattr(output, "logits"):
            output = output.logits  # HuggingFace models
        elif isinstance(output, tuple):
            output = output[0]  # Some models return tuples
        elif isinstance(output, dict):
            # Some models return dicts with 'logits' key
            output = output.get("logits", output.get("output", list(output.values())[0]))

        return [output.cpu().to(torch.float32).numpy()]
