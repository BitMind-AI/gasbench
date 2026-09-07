"""Multiclass label mapping and dataset-taxonomy guards."""

import numpy as np
import pytest

from src.gasbench.constants import media_type_to_label
from src.gasbench.dataset.config import load_benchmark_datasets_from_yaml


EXCLUDED = {
    "justweirdimages",
    "cg-fake-id",
    "deepfake-insight",
    "artifact-bench",
    "pica-100k-src",
    "pica-100k-tgt",
}


def _load_metrics():
    """Load Metrics without importing gasbench.benchmarks (needs cv2)."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    if "src.gasbench.logger" not in sys.modules:
        logmod = types.ModuleType("src.gasbench.logger")

        def get_logger(_name):
            class _L:
                def warning(self, *a, **k):
                    pass

                def info(self, *a, **k):
                    pass

                def error(self, *a, **k):
                    pass

                def debug(self, *a, **k):
                    pass

            return _L()

        logmod.get_logger = get_logger
        sys.modules["src.gasbench.logger"] = logmod

    path = (
        Path(__file__).resolve().parents[2] / "src/gasbench/benchmarks/utils/metrics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "src.gasbench.benchmarks.utils.metrics_isolated", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Metrics


class TestLabelMaps:
    def test_image_three_class(self):
        assert media_type_to_label("real", "image") == 0
        assert media_type_to_label("synthetic", "image") == 1
        assert media_type_to_label("semisynthetic", "image") == 2
        with pytest.raises(KeyError):
            media_type_to_label("rendered", "image")

    def test_video_four_class(self):
        assert media_type_to_label("real", "video") == 0
        assert media_type_to_label("synthetic", "video") == 1
        assert media_type_to_label("semisynthetic", "video") == 2
        assert media_type_to_label("rendered", "video") == 3

    def test_audio_binary(self):
        assert media_type_to_label("real", "audio") == 0
        assert media_type_to_label("synthetic", "audio") == 1
        assert media_type_to_label("semisynthetic", "audio") == 1


class TestMetricsCollapse:
    def test_sn34_collapses_non_real_to_positive(self):
        Metrics = _load_metrics()
        m = Metrics()
        # semisynthetic label=2, predicted class=2 → binary TP
        m.update(label=2, pred=2, pred_probs=np.array([0.1, 0.2, 0.7]))
        # rendered label=3, predicted class=3 → binary TP
        m.update(label=3, pred=3, pred_probs=np.array([0.05, 0.1, 0.15, 0.7]))
        # real correctly predicted
        m.update(label=0, pred=0, pred_probs=np.array([0.9, 0.05, 0.05]))
        assert m.true_positives == 2.0
        assert m.true_negatives == 1.0
        assert m.false_positives == 0.0
        assert m.false_negatives == 0.0
        assert m.calculate_binary_mcc() == pytest.approx(1.0)


class TestYamlTaxonomy:
    def test_code_rendered_images_are_non_generative(self):
        configs = load_benchmark_datasets_from_yaml()
        datasets = {dataset.name: dataset for dataset in configs["image"]}
        assert datasets["cosyn-400k"].media_type == "real"

    def test_excluded_datasets_absent(self):
        configs = load_benchmark_datasets_from_yaml()
        names = {d.name for ds in configs.values() for d in ds}
        present = EXCLUDED & names
        assert not present, f"excluded datasets still in registry: {present}"

    def test_video_has_all_four_classes(self):
        configs = load_benchmark_datasets_from_yaml()
        types = {d.media_type for d in configs["video"]}
        assert types == {"real", "synthetic", "semisynthetic", "rendered"}

    def test_pica_is_synthetic_not_split(self):
        configs = load_benchmark_datasets_from_yaml()
        pica = next(d for d in configs["image"] if d.name == "pica-100k")
        assert pica.media_type == "synthetic"
        assert pica.data_columns == ["src_img", "tgt_img"]

    def test_gemini_is_synthetic_generation(self):
        configs = load_benchmark_datasets_from_yaml()
        names = {d.name: d for d in configs["image"]}
        for name in ("gemini31-flash-lite-train", "gemini31-flash-lite-val"):
            assert names[name].media_type == "synthetic"

    def test_full_frame_neural_edits_are_synthetic(self):
        configs = load_benchmark_datasets_from_yaml()
        names = {d.name: d for d in configs["image"] + configs["video"]}
        for name in (
            "receipts-i2i",
            "AttGAN",
            "STARGAN",
            "STGAN_CelebA",
            "gpt-image-edit-1-5m-hqedit",
            "dagan",
            "fomm",
            "lia",
            "mcnet",
            "mraa",
            "oneshot",
            "pirender",
            "facevid2vid",
            "tpsm",
            "v15-human-vid-mavos-dd-english_echomimic",
            "hyperreenact",
            "fakeparts-stylechange",
            "senorita-controllable-videos",
            "senorita-style-transfer",
        ):
            assert names[name].media_type == "synthetic", name

    def test_localized_digifakeav_is_semisynthetic(self):
        configs = load_benchmark_datasets_from_yaml()
        names = {d.name: d for d in configs["video"]}
        assert (
            names["v15-human-vid-digifakeavfvfa_with_audio"].media_type
            == "semisynthetic"
        )

    def test_cgi_videos_are_rendered(self):
        configs = load_benchmark_datasets_from_yaml()
        names = {d.name: d for d in configs["video"]}
        for name in (
            "abot-world-explorer",
            "physicalai-autonomous-driving-pedestrian",
            "bedlam-closeup-suburb-a",
            "nvidia-sdg-synhuman-shard-7",
            "cs2-10k-data-ancient-part-01",
            "ByteDance_Synthetic_Videos",
            "scene-decoupled-video-dataset",
            "synwts",
        ):
            assert names[name].media_type == "rendered", name
