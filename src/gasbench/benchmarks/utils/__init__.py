"""Utilities for benchmark processing."""

from .metrics import (
    ConfusionMatrix,
    multiclass_to_binary,
    update_generator_stats,
    calculate_per_source_accuracy,
)
from .misclassifications import (
    generate_sample_id,
    should_track_sample,
    create_misclassification_record,
    aggregate_misclassification_stats,
)

__all__ = [
    "ConfusionMatrix",
    "multiclass_to_binary",
    "update_generator_stats",
    "calculate_per_source_accuracy",
    "generate_sample_id",
    "should_track_sample",
    "create_misclassification_record",
    "aggregate_misclassification_stats",
]

