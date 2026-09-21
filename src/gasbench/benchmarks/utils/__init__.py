"""Lazy exports, matching gasbench: recorder recovery needs no inference backend."""

from importlib import import_module

_LAZY_EXPORTS = {
    "Metrics": (".metrics", "Metrics"),
    "update_generator_stats": (".metrics", "update_generator_stats"),
    "calculate_per_source_accuracy": (".metrics", "calculate_per_source_accuracy"),
    "create_inference_session": (".inference", "create_inference_session"),
    "process_model_output": (".inference", "process_model_output"),
    "PyTorchInferenceSession": (".pytorch_session", "PyTorchInferenceSession"),
    "load_custom_model": (".custom_model_loader", "load_custom_model"),
    "validate_model_directory": (".custom_model_loader", "validate_model_directory"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr)
