"""Constants used throughout the gasbench package."""

# Media type to label mapping for binary classification
# 0 = real, 1 = AI-generated (synthetic or semisynthetic)
# Note: Dataset configs still distinguish synthetic vs semisynthetic for provenance tracking
MEDIA_TYPE_TO_LABEL = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1,
}

