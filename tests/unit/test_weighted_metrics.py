"""
Unit tests for weighted metrics and score composition.

Covers:
- Weighted Metrics semantics: weight=w is exactly equivalent to w duplicate samples.
- Back-compat: default weights reproduce the historical unweighted behavior.
- Provenance classification and target-share weight derivation.
- compute_metrics_from_df with score_composition (incl. absent-class renormalization).
"""

import numpy as np
import pandas as pd
import pytest

from src.gasbench.benchmarks.utils.metrics import Metrics
from src.gasbench.benchmarks.recording import (
    classify_sample_provenance,
    compute_metrics_from_df,
    derive_provenance_weights,
)


def make_df(rows):
    """rows: list of (dataset_name, label, predicted, p_synthetic)."""
    return pd.DataFrame(
        [
            {
                "status": "ok",
                "dataset_name": name,
                "label": label,
                "predicted": pred,
                "correct": int(label == pred),
                "probs": [1.0 - p, p],
                "inference_time_ms": 10.0,
            }
            for (name, label, pred, p) in rows
        ]
    )


class TestWeightedMetrics:
    def test_default_weight_matches_unweighted(self):
        """weight omitted == historical behavior on a hand-checked case."""
        m = Metrics()
        m.update(1, 1, [0.1, 0.9])
        m.update(0, 0, [0.8, 0.2])
        m.update(1, 0, [0.6, 0.4])
        m.update(0, 1, [0.3, 0.7])

        assert m.true_positives == 1.0
        assert m.true_negatives == 1.0
        assert m.false_negatives == 1.0
        assert m.false_positives == 1.0
        # MCC of a fully balanced confusion matrix is 0
        assert m.calculate_binary_mcc() == pytest.approx(0.0)
        # Brier = mean((p - y)^2)
        expected_brier = np.mean(
            [(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.4 - 1) ** 2, (0.7 - 0) ** 2]
        )
        assert m.calculate_brier() == pytest.approx(expected_brier)

    def test_weight_equals_duplication(self):
        """A sample with weight=w must contribute exactly like w copies."""
        samples = [
            (1, 1, [0.2, 0.8]),
            (0, 0, [0.7, 0.3]),
            (1, 0, [0.55, 0.45]),
            (0, 1, [0.4, 0.6]),
            (1, 1, [0.05, 0.95]),
        ]
        weights = [3.0, 1.0, 2.0, 1.0, 4.0]

        weighted = Metrics()
        for (label, pred, probs), w in zip(samples, weights):
            weighted.update(label, pred, probs, weight=w)

        duplicated = Metrics()
        for (label, pred, probs), w in zip(samples, weights):
            for _ in range(int(w)):
                duplicated.update(label, pred, probs)

        assert weighted.calculate_binary_mcc() == pytest.approx(
            duplicated.calculate_binary_mcc()
        )
        assert weighted.calculate_brier() == pytest.approx(duplicated.calculate_brier())
        assert weighted.calculate_binary_cross_entropy() == pytest.approx(
            duplicated.calculate_binary_cross_entropy()
        )
        assert weighted.compute_sn34_score() == pytest.approx(
            duplicated.compute_sn34_score()
        )

    def test_weights_are_scale_invariant(self):
        """Scaling all weights by a constant must not change any metric."""
        samples = [(1, 1, [0.1, 0.9]), (0, 1, [0.2, 0.8]), (0, 0, [0.9, 0.1])]
        a, b = Metrics(), Metrics()
        for label, pred, probs in samples:
            a.update(label, pred, probs, weight=1.0)
            b.update(label, pred, probs, weight=7.5)
        assert a.calculate_binary_mcc() == pytest.approx(b.calculate_binary_mcc())
        assert a.calculate_brier() == pytest.approx(b.calculate_brier())
        assert a.compute_sn34_score() == pytest.approx(b.compute_sn34_score())


class TestProvenance:
    def test_classification(self):
        assert classify_sample_provenance("real-image-holdout-a1b2c3d4") == "holdout"
        assert classify_sample_provenance("gasstation-image-w23") == "gasstation"
        assert classify_sample_provenance("GASSTATION-VIDEO") == "gasstation"
        assert classify_sample_provenance("34data/pica-100k") == "public"

    def test_weight_derivation_hits_target_shares(self):
        counts = {"public": 800, "holdout": 150, "gasstation": 50}
        comp = {"public": 0.5, "holdout": 0.3, "gasstation": 0.2}
        w = derive_provenance_weights(counts, comp)
        total = sum(counts.values())
        for cls in counts:
            realized = w[cls] * counts[cls] / total
            assert realized == pytest.approx(comp[cls])
        # mean sample weight is 1.0
        assert sum(w[c] * counts[c] for c in counts) == pytest.approx(total)

    def test_absent_class_renormalizes(self):
        counts = {"public": 900, "holdout": 100}  # no gasstation (e.g. audio)
        comp = {"public": 0.5, "holdout": 0.3, "gasstation": 0.2}
        w = derive_provenance_weights(counts, comp)
        assert "gasstation" not in w
        total = sum(counts.values())
        # targets renormalize to public 0.625, holdout 0.375
        assert w["public"] * 900 / total == pytest.approx(0.625)
        assert w["holdout"] * 100 / total == pytest.approx(0.375)

    def test_zero_targets_fall_back_to_uniform(self):
        w = derive_provenance_weights({"public": 10}, {"holdout": 1.0})
        assert w == {"public": 1.0}


class TestComputeMetricsFromDf:
    def _mixed_df(self):
        # Model is perfect on public, wrong + overconfident on holdout
        rows = []
        for i in range(8):
            label = i % 2
            rows.append(("public-set", label, label, 0.9 if label else 0.1))
        for i in range(2):
            label = i % 2
            wrong = 1 - label
            rows.append(
                ("real-image-holdout-abc123", label, wrong, 0.9 if wrong else 0.1)
            )
        return make_df(rows)

    def test_no_composition_is_backward_compatible(self):
        """score_composition=None must reproduce the legacy pooled metrics."""
        df = self._mixed_df()
        result = compute_metrics_from_df(df)
        assert "score_composition" not in result
        assert result["benchmark_score"] == pytest.approx(0.8)
        # Pooled Brier: 8 samples at (0.9-1)^2=0.01-equivalents, 2 at 0.81
        expected_brier = (8 * 0.01 + 2 * 0.81) / 10
        assert result["binary_brier"] == pytest.approx(expected_brier)

    def test_composition_shifts_score_toward_holdout(self):
        df = self._mixed_df()
        pooled = compute_metrics_from_df(df)
        weighted = compute_metrics_from_df(
            df, score_composition={"public": 0.5, "holdout": 0.5}
        )
        # Holdout (where the model fails) now carries 50% instead of 20%
        assert weighted["sn34_score"] < pooled["sn34_score"]
        assert weighted["binary_brier"] > pooled["binary_brier"]
        assert weighted["benchmark_score"] == pytest.approx(0.5)
        assert weighted["realized_composition"]["holdout"] == pytest.approx(0.2)
        assert weighted["score_composition"]["holdout"] == pytest.approx(0.5)
        assert weighted["provenance_weights"]["holdout"] == pytest.approx(2.5)
        assert weighted["provenance_weights"]["public"] == pytest.approx(0.625)

    def test_uniform_composition_matches_pooled(self):
        """Composition equal to realized shares must equal the pooled metrics."""
        df = self._mixed_df()
        pooled = compute_metrics_from_df(df)
        same = compute_metrics_from_df(
            df, score_composition={"public": 0.8, "holdout": 0.2}
        )
        assert same["sn34_score"] == pytest.approx(pooled["sn34_score"])
        assert same["binary_mcc"] == pytest.approx(pooled["binary_mcc"])
        assert same["binary_brier"] == pytest.approx(pooled["binary_brier"])
        assert same["benchmark_score"] == pytest.approx(pooled["benchmark_score"])

    def test_legacy_holdout_weight_only_affects_accuracy(self):
        df = self._mixed_df()
        pooled = compute_metrics_from_df(df)
        legacy = compute_metrics_from_df(df, holdout_weight=4.0)
        assert legacy["benchmark_score"] < pooled["benchmark_score"]
        assert legacy["sn34_score"] == pytest.approx(pooled["sn34_score"])
        assert legacy["binary_brier"] == pytest.approx(pooled["binary_brier"])

    def test_empty_df_defaults(self):
        result = compute_metrics_from_df(pd.DataFrame())
        assert result["sn34_score"] == 0.0
        assert result["binary_brier"] == 0.25
