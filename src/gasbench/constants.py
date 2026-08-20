"""Constants used throughout the gasbench package."""

# Per-modality class indices. Class 0 is always real so binary collapse is 1 - p[0].
#
# Image is 3-class (no rendered): 0=real, 1=synthetic, 2=semisynthetic.
# Video is 4-class:                0=real, 1=synthetic, 2=semisynthetic, 3=rendered.
# Audio stays binary:              0=real, 1=synthetic (semisynthetic collapsed).
#
# Image has no rendered class: CGI/game-engine stills are excluded from image configs.
IMAGE_MEDIA_TYPE_TO_LABEL = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 2,
}
VIDEO_MEDIA_TYPE_TO_LABEL = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 2,
    "rendered": 3,
}
AUDIO_MEDIA_TYPE_TO_LABEL = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1,
}

VALID_MEDIA_TYPES = {
    "image": frozenset(IMAGE_MEDIA_TYPE_TO_LABEL),
    "video": frozenset(VIDEO_MEDIA_TYPE_TO_LABEL),
    "audio": frozenset({"real", "synthetic", "semisynthetic"}),
}

_MODALITY_LABELS = {
    "image": IMAGE_MEDIA_TYPE_TO_LABEL,
    "video": VIDEO_MEDIA_TYPE_TO_LABEL,
    "audio": AUDIO_MEDIA_TYPE_TO_LABEL,
}

# Legacy binary map (real vs not-real). Prefer media_type_to_label(media_type, modality).
MEDIA_TYPE_TO_LABEL = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1,
}


def media_type_to_label(media_type: str, modality: str) -> int:
    """Map a dataset media_type to the integer class index for this modality."""
    table = _MODALITY_LABELS.get(modality)
    if table is None:
        raise KeyError(f"Unknown modality '{modality}'")
    try:
        return table[media_type]
    except KeyError:
        raise KeyError(
            f"Invalid media_type '{media_type}' for modality '{modality}'. "
            f"Valid: {sorted(table)}"
        ) from None


# Video evaluation limits
# Caps num_frames to prevent submitted models from overwhelming the eval system.
MAX_VIDEO_NUM_FRAMES = 64

