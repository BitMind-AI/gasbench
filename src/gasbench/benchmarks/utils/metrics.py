import numpy as np
from typing import Dict, List

from ...logger import get_logger

logger = get_logger(__name__)


class Metrics:

    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: Number of scored classes for this modality (see
                constants.MODALITY_NUM_CLASSES: image=3, video=4, audio=2).
                num_classes=2 makes the multiclass metrics reduce exactly to
                the binary ones.
        """
        self.num_classes = max(2, int(num_classes))
        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.binary_y_true = []
        self.binary_probs = []
        self.binary_weights = []
        # Multiclass accumulators. confusion[true][pred], weighted.
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=float)
        self.mc_sq_error = 0.0   # weighted sum of per-sample multiclass Brier
        self.mc_weight = 0.0
        self._clipped_preds = 0

    def update(
        self,
        label: int,
        pred: int,
        pred_probs: np.ndarray = None,
        weight: float = 1.0,
    ):
        """Update confusion matrices with a new prediction.

        A sample with weight=w contributes to every metric exactly as w copies
        of the same sample would. Default weight=1.0 reproduces unweighted
        behavior.

        Maintains both the binary (real vs not-real) and the full multiclass
        accumulators, so either score can be computed from one pass.
        """
        K = self.num_classes

        # ---- binary (real vs not-real) ----------------------------------
        binary_label = 0 if label == 0 else 1

        # Derive the binary decision from the collapsed not-real mass rather
        # than from argmax over the full head. A K>2 head splits its not-real
        # mass across classes while p[real] stays whole, so argmax only leaves
        # class 0 once p[real] < max(p[1:]) — p_not_real > 0.75 for an even
        # 3-way split versus > 0.5 for a binary head with identical beliefs.
        # Exactly equivalent to argmax when len(pred_probs) <= 2.
        if pred_probs is None or len(pred_probs) == 0:
            p_not_real = None
        elif len(pred_probs) >= 2:
            p_not_real = float(1.0 - pred_probs[0])
        else:
            p_not_real = float(pred_probs[0])

        if p_not_real is None:
            binary_pred = 0 if pred == 0 else 1
        else:
            binary_pred = int(p_not_real > 0.5)

        if binary_label == 1 and binary_pred == 1:
            self.true_positives += weight
        elif binary_label == 0 and binary_pred == 0:
            self.true_negatives += weight
        elif binary_label == 0 and binary_pred == 1:
            self.false_positives += weight
        elif binary_label == 1 and binary_pred == 0:
            self.false_negatives += weight

        if p_not_real is not None:
            self.binary_y_true.append(binary_label)
            self.binary_weights.append(weight)
            self.binary_probs.append(p_not_real)

        # ---- multiclass --------------------------------------------------
        # A head wider than this modality's class count can emit an index that
        # does not exist here (e.g. a 4-wide head predicting rendered=3 on a
        # 3-class image run). See the out-of-range handling below.
        t = int(label)
        if not (0 <= t < K):
            t = min(max(t, 0), K - 1)

        # Project the head's distribution onto this modality's classes. A narrow
        # head is padded with zeros (it assigns no mass to classes it cannot
        # express); a wide one is truncated (mass outside this modality's
        # classes is simply lost, which costs Brier).
        probs = None
        if pred_probs is not None and len(pred_probs) > 0:
            probs = np.zeros(K, dtype=float)
            src = np.asarray(pred_probs, dtype=float).ravel()
            if len(src) == 1:
                # single-logit head: [p_not_real] -> [1-p, p]
                probs[0] = 1.0 - float(src[0])
                probs[1] = float(src[0])
            else:
                n = min(K, len(src))
                probs[:n] = src[:n]

        p = int(pred)
        if not (0 <= p < K):
            # Do NOT clip to K-1: that lands on the diagonal whenever the true
            # label happens to be K-1, scoring a prediction of a class that does
            # not exist in this modality as correct. Fall back to the model's
            # best VALID class — the same projection Brier uses, and what you
            # would do at inference time — or, with no probabilities to fall
            # back on, to a deterministic non-matching class so an invalid
            # prediction can never be counted correct.
            self._clipped_preds += 1
            if probs is not None and probs.sum() > 0:
                p = int(np.argmax(probs))
            else:
                p = (t + 1) % K

        self.confusion[t, p] += weight

        if probs is not None:
            onehot = np.zeros(K, dtype=float)
            onehot[t] = 1.0
            self.mc_sq_error += weight * float(np.sum((probs - onehot) ** 2))
            self.mc_weight += weight

    def calculate_binary_mcc(self) -> float:
        """Calculate Matthews Correlation Coefficient for binary classification."""
        numerator = (self.true_positives * self.true_negatives) - (
            self.false_positives * self.false_negatives
        )
        denominator = np.sqrt(
            (self.true_positives + self.false_positives)
            * (self.true_positives + self.false_negatives)
            * (self.true_negatives + self.false_positives)
            * (self.true_negatives + self.false_negatives)
        )
        return numerator / denominator if denominator > 0 else 0.0

    def calculate_binary_cross_entropy(self) -> float:
        """Calculate binary cross-entropy loss (kept for backward compatibility/logging)."""
        if len(self.binary_y_true) == 0 or len(self.binary_probs) == 0:
            return 0.0

        y_true = np.array(self.binary_y_true)
        y_prob = np.clip(np.array(self.binary_probs), 1e-7, 1 - 1e-7)
        weights = np.array(self.binary_weights)

        losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return float(np.average(losses, weights=weights))

    def calculate_brier(self) -> float:
        """
        Calculate Brier score: mean squared error between predicted probs and true labels.
        
        Brier score directly measures calibration:
        - Perfect calibration: 0.0
        - Random baseline (p=0.5 always): 0.25
        - Worst case (always wrong with p=1.0): 1.0
        
        Unlike CE, Brier penalizes overconfident wrong predictions more severely,
        incentivizing miners to submit calibrated probabilities instead of binary 0/1.
        """
        if len(self.binary_y_true) == 0 or len(self.binary_probs) == 0:
            return 0.25  # random baseline
        
        y_true = np.array(self.binary_y_true)
        y_prob = np.clip(np.array(self.binary_probs), 1e-7, 1 - 1e-7)
        weights = np.array(self.binary_weights)

        return float(np.average((y_prob - y_true) ** 2, weights=weights))

    def calculate_multiclass_mcc(self) -> float:
        """Gorodkin's R_K — the multiclass generalisation of MCC.

            R_K = (N*Tr(C) - sum_k t_k*p_k)
                  / sqrt((N^2 - sum_k p_k^2) * (N^2 - sum_k t_k^2))

        where C is the weighted confusion matrix, t_k the true count of class k
        and p_k the predicted count of class k.

        Reduces exactly to the binary MCC when num_classes == 2. Range is
        [-1, 1] for K=2; for K>2 the attainable minimum is -1/(K-1) rather
        than -1, so the normalised floor in compute_sn34_score is above 0.
        Returns 0.0 for a degenerate matrix (single class present or a model
        that emitted only one class), matching calculate_binary_mcc.
        """
        C = self.confusion
        N = float(C.sum())
        if N <= 0:
            return 0.0
        t = C.sum(axis=1)  # true totals per class
        p = C.sum(axis=0)  # predicted totals per class
        numerator = N * float(np.trace(C)) - float(np.dot(t, p))
        denominator = np.sqrt(
            max(0.0, N * N - float(np.dot(p, p)))
            * max(0.0, N * N - float(np.dot(t, t)))
        )
        return float(numerator / denominator) if denominator > 0 else 0.0

    def calculate_multiclass_brier(self) -> float:
        """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2.

        Range [0, 2]. A uniform guesser (p_k = 1/K) scores (K-1)/K, i.e. 0.5
        for K=2, 0.667 for K=3, 0.75 for K=4 — that is the baseline used for
        normalisation in compute_sn34_score.

        For K=2 this is exactly twice the binary calculate_brier(), and the
        baseline is exactly twice 0.25, so the normalised value is identical.
        """
        if self.mc_weight <= 0:
            K = self.num_classes
            return (K - 1) / K  # random baseline
        return float(self.mc_sq_error / self.mc_weight)

    def multiclass_random_baseline(self) -> float:
        """Brier score of a uniform guesser over num_classes."""
        K = self.num_classes
        return (K - 1) / K

    def per_class_recall(self) -> Dict[int, float]:
        """Recall per true class, for diagnosing which classes a model confuses."""
        out = {}
        for k in range(self.num_classes):
            total = float(self.confusion[k].sum())
            out[k] = float(self.confusion[k, k] / total) if total > 0 else 0.0
        return out

    def compute_sn34_score(
        self,
        alpha: float = 1.2,
        beta: float = 1.8,
        agg: str = "geomean",
        multiclass: bool = False,
    ) -> float:
        """
        SN34 score: combined metric from binary MCC (discrimination) and Brier (calibration).

        - MCC in [-1,1] -> mcc_norm in [0,1] via (mcc+1)/2, then sharpen via alpha.
        - Brier in [0,0.25] -> brier_score in [0,1] via (0.25-brier)/0.25, then sharpen via beta.
        - Geometric mean penalizes imbalance between discrimination (MCC) and calibration (Brier).

        Args:
            alpha: exponent on MCC score (>=1 boosts top-end separation). Default 1.2.
            beta: exponent on Brier score (>=1 boosts calibration emphasis). Default 1.8.
            agg: "geomean" | "harmmean" | "mean" for combining MCC vs Brier.

        Returns:
            float in [0,1].
        """

        if multiclass:
            # Gorodkin R_K in [-1,1] -> [0,1]. Same (x+1)/2 mapping as binary;
            # note the attainable floor is -1/(K-1) for K>2, so a maximally
            # wrong K=4 model lands near 0.33 rather than 0.
            mcc = float(self.calculate_multiclass_mcc())
            brier = self.calculate_multiclass_brier()
            baseline = self.multiclass_random_baseline()
        else:
            mcc = float(self.calculate_binary_mcc())
            brier = self.calculate_brier()
            baseline = 0.25

        mcc_norm = max(0.0, min((mcc + 1.0) / 2.0, 1.0)) ** alpha

        # Normalize Brier against the uniform-guess baseline for this class
        # count: random -> 0, perfect -> 1. Then apply beta exponent.
        brier_score = max(0.0, (baseline - brier) / baseline) ** beta

        if agg == "geomean":
            final = (max(1e-12, mcc_norm * brier_score)) ** 0.5
        elif agg == "harmmean":
            denom = max(1e-12, mcc_norm + brier_score)
            final = 2.0 * mcc_norm * brier_score / denom
        else:
            final = 0.5 * (mcc_norm + brier_score)

        return float(max(0.0, min(final, 1.0)))


def update_generator_stats(
    generator_stats: Dict[str, Dict],
    sample: Dict,
    label: int,
    pred: int
) -> None:
    """Update per-generator fooling statistics for gasstation datasets."""
    generator_hotkey = sample.get("generator_hotkey")
    # Skip if no generator info or not synthetic sample or "unknown" generator
    if generator_hotkey is None or label != 1 or str(generator_hotkey).lower() == "unknown":
        return
    
    stats = generator_stats.get(generator_hotkey) or {
        "fooled_count": 0, 
        "not_fooled_count": 0
    }
    
    generator_uid = sample.get("generator_uid")
    if generator_uid is not None and stats.get("uid") is None:
        try:
            stats["uid"] = int(generator_uid)
            logger.debug(f"📝 Tracking generator: {generator_hotkey[:16]}... (UID: {stats['uid']})")
        except Exception as e:
            logger.warning(f"Failed to parse generator_uid: {generator_uid} - {e}")
            pass
    
    if pred == 0:
        stats["fooled_count"] += 1
    else:
        stats["not_fooled_count"] += 1
    
    generator_stats[generator_hotkey] = stats


def calculate_per_source_accuracy(
    valid_datasets: List,
    per_dataset_results: Dict[str, Dict]
) -> Dict[str, Dict[str, Dict]]:
    """Build per-source prediction distribution organized by media type and dataset.

    For each dataset, returns counts for every media label. Keeping the full
    distribution is required for image/video multiclass heads; otherwise
    predictions above label 1 disappear from per-source reports.
    """
    per_source_accuracy = {}
    
    for dataset_config in valid_datasets:
        results = per_dataset_results.get(dataset_config.name, {})
        media_type = dataset_config.media_type
        
        if media_type not in per_source_accuracy:
            per_source_accuracy[media_type] = {}

        preds = results.get("predictions", {})
        
        per_source_accuracy[media_type][dataset_config.name] = {
            "real": int(preds.get("real", 0)),
            "synthetic": int(preds.get("synthetic", 0)),
            "semisynthetic": int(preds.get("semisynthetic", 0)),
            "rendered": int(preds.get("rendered", 0)),
        }
    
    return per_source_accuracy
