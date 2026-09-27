"""Class order is a public model interface, independent of dataset membership."""

import pytest

from gasbench.constants import media_type_to_label


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_model_class_order_and_invalid_labels(modality):
    assert media_type_to_label("real", modality) == 0
    assert media_type_to_label("synthetic", modality) == 1
    assert media_type_to_label("semisynthetic", modality) == (1 if modality == "audio" else 2)
    with pytest.raises(KeyError):
        media_type_to_label("unknown", modality)
    with pytest.raises(KeyError):
        media_type_to_label("rendered", modality)


def test_unknown_modality_is_rejected():
    with pytest.raises(KeyError, match="modality"):
        media_type_to_label("real", "unknown")
