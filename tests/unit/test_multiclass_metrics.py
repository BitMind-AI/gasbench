"""Unit tests for multiclass SN34 scoring (Gorodkin MCC + multiclass Brier).

The critical property is that num_classes=2 reduces EXACTLY to the previous
binary behaviour, so switching a modality to multiclass is the only thing that
can change a score.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gasbench.benchmarks.utils.metrics import Metrics, calculate_per_source_accuracy
from gasbench.benchmarks.recording import compute_per_dataset_from_df
from sklearn.metrics import matthews_corrcoef


class TestReports:
    def test_per_dataset_report_preserves_all_video_prediction_classes(self):
        df = pd.DataFrame(
            {
                "status": ["ok"] * 3,
                "aug_pass": [False] * 3,
                "dataset_name": ["video-set"] * 3,
                "predicted": [0, 1, 2],
                "correct": [True, True, True],
            }
        )

        per_dataset = compute_per_dataset_from_df(df)
        per_source = calculate_per_source_accuracy(
            [SimpleNamespace(name="video-set", media_type="real")],
            per_dataset,
        )
        predictions = per_source["real"]["video-set"]

        assert predictions == {
            "real": 1,
            "synthetic": 1,
            "semisynthetic": 1,
        }


class TestBinaryReduction:
    def test_two_class_metrics_and_score_reduce_to_binary(self):
        # Include errors and unequal weights, with a score above the floor.
        # Random, uncorrelated predictions can make both scores zero even if
        # their formulas disagree.
        m = Metrics(num_classes=2)
        for label, p, weight in [(0, 0.1, 2), (1, 0.85, 3), (0, 0.6, 1), (1, 0.7, 2)]:
            m.update(label, int(p > 0.5), [1 - p, p], weight=weight)
        assert m.calculate_multiclass_mcc() == pytest.approx(m.calculate_binary_mcc())
        assert m.calculate_multiclass_brier() == pytest.approx(2 * m.calculate_brier())
        assert Metrics(num_classes=2).compute_sn34_score() < m.compute_sn34_score() < 1
        assert m.compute_sn34_score(multiclass=True) == pytest.approx(m.compute_sn34_score())


@pytest.mark.parametrize("num_classes", [2, 3, 4])
def test_mcc_matches_independent_reference(num_classes):
    labels = np.repeat(np.arange(num_classes), 3)
    predictions = labels.copy()
    predictions[::4] = (predictions[::4] + 1) % num_classes
    weights = np.arange(1, len(labels) + 1)
    metrics = Metrics(num_classes=num_classes)
    for label, pred, weight in zip(labels, predictions, weights):
        metrics.update(label, pred, np.eye(num_classes)[pred], weight=weight)
    assert metrics.calculate_multiclass_mcc() == pytest.approx(
        matthews_corrcoef(labels, predictions, sample_weight=weights)
    )


class TestBinaryDecisionCollapse:
    """binary_pred comes from collapsed not-real mass, not argmax over K."""

    def test_wide_head_not_penalised_on_split_mass(self):
        # p[real]=0.4, not-real mass 0.6 split evenly -> argmax would say "real".
        p = np.array([0.4, 0.2, 0.2, 0.2])
        m = Metrics(num_classes=4)
        m.update(label=1, pred=int(np.argmax(p)), pred_probs=p)
        # Collapsed mass is 0.6 > 0.5, so this counts as a true positive.
        assert m.true_positives == 1.0
        assert m.false_negatives == 0.0

    def test_two_class_head_matches_argmax_including_ties(self):
        for p in ([0.75, 0.25], [0.5, 0.5], [0.25, 0.75]):
            pred = int(np.argmax(p))
            m = Metrics(num_classes=2)
            m.update(label=1, pred=pred, pred_probs=p)
            assert m.true_positives == pred
            assert m.false_negatives == 1 - pred

    def test_all_non_real_labels_collapse_to_positive(self):
        m = Metrics(num_classes=4)
        for label in (1, 2, 3):
            m.update(label, label, np.eye(4)[label])
        m.update(0, 0, np.eye(4)[0])
        assert m.true_positives == 3
        assert m.true_negatives == 1
        assert m.calculate_binary_mcc() == pytest.approx(1)

    def test_falls_back_to_argmax_without_probs(self):
        m = Metrics(num_classes=4)
        m.update(label=1, pred=3, pred_probs=None)
        assert m.true_positives == 1.0
        m2 = Metrics(num_classes=4)
        m2.update(label=1, pred=0, pred_probs=None)
        assert m2.false_negatives == 1.0


class TestEndpoints:
    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_perfect_is_one_and_random_is_zero(self, K):
        perfect = Metrics(num_classes=K)
        rand = Metrics(num_classes=K)
        for y in range(K):
            oh = np.zeros(K)
            oh[y] = 1.0
            perfect.update(y, y, oh)
            rand.update(y, (y + 1) % K, np.ones(K) / K)
        assert perfect.compute_sn34_score(multiclass=True) == pytest.approx(1.0)
        # compute_sn34_score floors the geomean at max(1e-12, ...) ** 0.5 = 1e-6,
        # so a uniform guesser bottoms out there rather than at exactly 0.
        assert rand.compute_sn34_score(multiclass=True) < 1e-5
        assert rand.multiclass_random_baseline() == pytest.approx((K - 1) / K)


class TestHeadWidthHandling:
    def test_narrow_head_padded_with_zeros(self):
        # 2-wide head on a 4-class run: no mass on semisynthetic/rendered.
        m = Metrics(num_classes=4)
        m.update(label=2, pred=1, pred_probs=np.array([0.3, 0.7]))
        # Brier = 0.3^2 + 0.7^2 + (0-1)^2 + 0^2
        assert m.calculate_multiclass_brier() == pytest.approx(0.09 + 0.49 + 1.0)

    def test_single_logit_head_expanded(self):
        m = Metrics(num_classes=2)
        m.update(label=1, pred=1, pred_probs=np.array([0.8]))
        # [0.8] -> [0.2, 0.8]; Brier = 0.2^2 + (0.8-1)^2
        assert m.calculate_multiclass_brier() == pytest.approx(0.04 + 0.04)

    def test_out_of_range_pred_falls_back_to_best_valid_class(self):
        # 4-wide head predicting rendered=3 on a 3-class image run. Valid-class
        # mass is [0.1, 0.2, 0.3] so the best valid guess is class 2.
        m = Metrics(num_classes=3)
        m.update(label=0, pred=3, pred_probs=np.array([0.1, 0.2, 0.3, 0.4]))
        assert m.confusion.shape == (3, 3)
        assert m.confusion[0, 2] == 1.0

    def test_out_of_range_pred_is_not_scored_correct_by_clipping(self):
        # Regression: clipping pred 3 -> K-1 == 2 previously landed on the
        # diagonal when the true label was also 2, scoring a prediction of a
        # class that does not exist in this modality as correct.
        m = Metrics(num_classes=3)
        m.update(label=2, pred=3, pred_probs=np.array([0.5, 0.05, 0.05, 0.4]))
        # Best valid class is 0 (0.5), so this must be off the diagonal.
        assert m.confusion[2, 2] == 0.0
        assert m.confusion[2, 0] == 1.0
        assert m.calculate_multiclass_mcc() <= 0.0

    def test_out_of_range_pred_without_probs_cannot_be_correct(self):
        for K in (3, 4):
            for label in range(K):
                m = Metrics(num_classes=K)
                m.update(label=label, pred=K + 5, pred_probs=None)
                assert m.confusion[label, label] == 0.0, (K, label)
                assert m.confusion[label].sum() == 1.0


class TestPerClassRecall:
    def test_recall_reports_each_class(self):
        m = Metrics(num_classes=4)
        for y in (0, 0, 1, 2, 3):
            m.update(y, y, np.eye(4)[y])
        m.update(3, 1, np.eye(4)[1])
        recall = m.per_class_recall()
        assert recall == pytest.approx({0: 1, 1: 1, 2: 1, 3: 0.5})
