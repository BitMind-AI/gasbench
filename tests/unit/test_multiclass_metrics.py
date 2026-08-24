"""Unit tests for multiclass SN34 scoring (Gorodkin MCC + multiclass Brier).

The critical property is that num_classes=2 reduces EXACTLY to the previous
binary behaviour, so switching a modality to multiclass is the only thing that
can change a score.
"""

import numpy as np
import pytest

from src.gasbench.benchmarks.utils.metrics import Metrics
from src.gasbench.constants import MODALITY_NUM_CLASSES


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TestClassCounts:
    def test_modality_num_classes(self):
        # Derived from the label maps; audio collapses semisynthetic onto synthetic.
        assert MODALITY_NUM_CLASSES == {"image": 3, "video": 4, "audio": 2}


class TestBinaryReduction:
    """num_classes=2 must reproduce the binary metrics exactly."""

    def test_gorodkin_equals_binary_mcc(self):
        rng = np.random.default_rng(0)
        m = Metrics(num_classes=2)
        for _ in range(2000):
            y = int(rng.random() < 0.5)
            p = _softmax(rng.normal(0, 1.5, 2))
            m.update(y, int(np.argmax(p)), p)
        assert m.calculate_multiclass_mcc() == pytest.approx(
            m.calculate_binary_mcc(), abs=1e-12
        )

    def test_multiclass_brier_is_twice_binary(self):
        rng = np.random.default_rng(1)
        m = Metrics(num_classes=2)
        for _ in range(2000):
            y = int(rng.random() < 0.5)
            p = _softmax(rng.normal(0, 1.5, 2))
            m.update(y, int(np.argmax(p)), p)
        # Sum-over-classes form is exactly 2x the single-probability form, and
        # the baseline scales the same way, so the normalised score is unchanged.
        assert m.calculate_multiclass_brier() == pytest.approx(
            2 * m.calculate_brier(), rel=1e-12
        )
        assert m.multiclass_random_baseline() == pytest.approx(0.5)

    def test_sn34_identical_at_k2(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            m = Metrics(num_classes=2)
            for _ in range(200):
                y = int(rng.random() < 0.5)
                p = _softmax(rng.normal(0, 2, 2))
                m.update(y, int(np.argmax(p)), p, weight=float(rng.random() * 2))
            assert m.compute_sn34_score(multiclass=True) == pytest.approx(
                m.compute_sn34_score(multiclass=False), abs=1e-12
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

    def test_two_class_head_matches_argmax(self):
        rng = np.random.default_rng(3)
        for _ in range(2000):
            p = _softmax(rng.normal(0, 2, 2))
            argmax_pred = int(np.argmax(p))
            m = Metrics(num_classes=2)
            m.update(label=1, pred=argmax_pred, pred_probs=p)
            collapsed = 1.0 if m.true_positives else 0.0
            assert collapsed == float(argmax_pred == 1)

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
        rng = np.random.default_rng(4)
        perfect = Metrics(num_classes=K)
        rand = Metrics(num_classes=K)
        for _ in range(3000):
            y = int(rng.integers(0, K))
            oh = np.zeros(K)
            oh[y] = 1.0
            perfect.update(y, y, oh)
            rand.update(y, int(rng.integers(0, K)), np.ones(K) / K)
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

    def test_out_of_range_pred_is_clipped(self):
        # 4-wide head predicting rendered=3 on a 3-class image run.
        m = Metrics(num_classes=3)
        m.update(label=0, pred=3, pred_probs=np.array([0.1, 0.2, 0.3, 0.4]))
        assert m._clipped_preds == 1
        assert m.confusion.shape == (3, 3)
        assert m.confusion[0, 2] == 1.0


class TestWeighting:
    def test_weight_equals_repetition(self):
        rng = np.random.default_rng(5)
        samples = []
        for _ in range(200):
            y = int(rng.integers(0, 4))
            samples.append((y, _softmax(rng.normal(0, 1.5, 4))))

        weighted = Metrics(num_classes=4)
        repeated = Metrics(num_classes=4)
        for y, p in samples:
            weighted.update(y, int(np.argmax(p)), p, weight=3.0)
            for _ in range(3):
                repeated.update(y, int(np.argmax(p)), p, weight=1.0)

        assert weighted.compute_sn34_score(multiclass=True) == pytest.approx(
            repeated.compute_sn34_score(multiclass=True), abs=1e-12
        )


class TestPerClassRecall:
    def test_recall_reports_each_class(self):
        m = Metrics(num_classes=4)
        for y in (0, 0, 1, 2, 3):
            m.update(y, y, np.eye(4)[y])
        m.update(3, 1, np.eye(4)[1])
        recall = m.per_class_recall()
        assert recall[0] == pytest.approx(1.0)
        assert recall[3] == pytest.approx(0.5)
