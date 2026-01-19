"""Load custom model architectures from model.py files."""

import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from ...logger import get_logger

logger = get_logger(__name__)


def load_model_config(config_path: Path) -> Dict[str, Any]:
    """Load model configuration from YAML.
    
    Args:
        config_path: Path to model_config.yaml
        
    Returns:
        Parsed configuration dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_custom_model(model_dir: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a custom model from a directory containing model.py + weights.

    Args:
        model_dir: Directory containing model_config.yaml, model.py, and weights

    Returns:
        Tuple of (loaded PyTorch model, config dict)
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.yaml not found in {model_dir}")

    config = load_model_config(config_path)
    logger.info(f"Loaded model config: {config.get('name', 'unnamed')}")

    # Load model.py module
    model_py_path = model_dir / "model.py"
    if not model_py_path.exists():
        raise FileNotFoundError(f"model.py not found in {model_dir}")

    # Dynamically import the model module
    spec = importlib.util.spec_from_file_location("custom_model", model_py_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to create module spec from {model_py_path}")
    
    module = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules so imports within model.py work
    sys.modules["custom_model"] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Failed to execute model.py: {e}")

    # Check for required load_model function
    if not hasattr(module, "load_model"):
        raise ValueError(
            "model.py must define a load_model(weights_path, num_classes) function"
        )

    # Get weights path and num_classes from config
    weights_file = config.get("model", {}).get("weights_file", "model.safetensors")
    weights_path = model_dir / weights_file
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    num_classes = config.get("model", {}).get("num_classes", 2)

    logger.info(f"Loading model with {num_classes} classes from {weights_path}")

    # Call the miner's load_model function
    try:
        model = module.load_model(str(weights_path), num_classes=num_classes)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    if not isinstance(model, torch.nn.Module):
        raise ValueError(
            f"load_model() must return a torch.nn.Module, got {type(model)}"
        )

    return model, config


def validate_model_directory(model_dir: Path) -> bool:
    """
    Validate that a directory contains required files for a custom model.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        True if valid, raises exception otherwise
    """
    model_dir = Path(model_dir)
    
    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")
    
    config_path = model_dir / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.yaml not found in {model_dir}")
    
    model_py_path = model_dir / "model.py"
    if not model_py_path.exists():
        raise FileNotFoundError(f"model.py not found in {model_dir}")
    
    # Load config to check weights file
    config = load_model_config(config_path)
    weights_file = config.get("model", {}).get("weights_file", "model.safetensors")
    weights_path = model_dir / weights_file
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    return True
