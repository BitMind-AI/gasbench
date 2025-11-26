import numpy as np
from typing import Dict, List

from ...logger import get_logger

logger = get_logger(__name__)


class Metrics:
    
    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.binary_y_true = []
        self.binary_probs = []
    
    def update(
        self, 
        label: int,
        pred: int, 
        pred_probs: np.ndarray = None
    ):
        """Update confusion matrix with new prediction."""
        if label == 1 and pred == 1:
            self.true_positives += 1
        elif label == 0 and pred == 0:
            self.true_negatives += 1
        elif label == 0 and pred == 1:
            self.false_positives += 1
        elif label == 1 and pred == 0:
            self.false_negatives += 1

        if pred_probs is not None:
            self.binary_y_true.append(label)
            if len(pred_probs) == 3:
                binary_prob = 1.0 - pred_probs[0]
                self.binary_probs.append(binary_prob)
            elif len(pred_probs) == 2:
                self.binary_probs.append(pred_probs[1])
            else:
                self.binary_probs.append(pred_probs[0])

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
        """Calculate binary cross-entropy loss."""
        if len(self.binary_y_true) == 0 or len(self.binary_probs) == 0:
            return 0.0

        y_true = np.array(self.binary_y_true)
        y_prob = np.clip(np.array(self.binary_probs), 1e-7, 1 - 1e-7)

        loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return float(loss)
    
    def compute_sn34_score(self, alpha: float = 1.0, beta: float = 1.0,
                        agg: str = "geomean") -> float:
        """
        Combined SN34 score from binary MCC and Cross-Entropy with principled normalization.

        - MCC in [-1,1] -> mcc_score in [0,1] via (mcc+1)/2, then sharpen via alpha.
        - CE -> exp(-CE) = 1/perplexity; rescale so random=0 and perfect=1:
            ce_score = (exp(-CE) - 1/K) / (1 - 1/K), then sharpen via beta.
        - Combine MCC and CE via geometric mean (default) to penalize imbalance.

        Args:
            alpha: exponent on MCC score (>=1 boosts top-end separation).
            beta: exponent on CE score (>=1 boosts top-end separation).
            agg: "geomean" | "harmmean" | "mean" for combining MCC vs CE.

        Returns:
            float in [0,1].
        """
        import math
        
        bin_mcc = float(self.calculate_binary_mcc())
        bin_ce = float(self.calculate_binary_cross_entropy())

        def safe_exp_neg(x):
            x = max(0.0, min(x, 20.0))
            return math.exp(-x)

        mcc_score = max(0.0, min((bin_mcc + 1.0) / 2.0, 1.0))
        mcc_score = mcc_score ** alpha

        def ce_to_score(ce: float, K: int) -> float:
            base = 1.0 / K
            val = safe_exp_neg(ce)
            if K <= 1:
                return 1.0 if ce == 0.0 else 0.0
            num = max(0.0, val - base)
            den = 1.0 - base
            return max(0.0, min(num / den, 1.0))

        ce_score = ce_to_score(bin_ce, K=2) ** beta

        if agg == "geomean":
            final = math.sqrt(max(1e-12, mcc_score * ce_score))
        elif agg == "harmmean":
            denom = max(1e-12, (mcc_score + ce_score))
            final = 2.0 * mcc_score * ce_score / denom
        else:
            final = 0.5 * (mcc_score + ce_score)

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

    For each dataset, returns counts of predictions by binary label:
      { "real": N_real, "synthetic": N_synthetic }
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
        }
    
    return per_source_accuracy

