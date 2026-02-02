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
        """Calculate binary cross-entropy loss (kept for backward compatibility/logging)."""
        if len(self.binary_y_true) == 0 or len(self.binary_probs) == 0:
            return 0.0

        y_true = np.array(self.binary_y_true)
        y_prob = np.clip(np.array(self.binary_probs), 1e-7, 1 - 1e-7)

        loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return float(loss)

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
        
        return float(np.mean((y_prob - y_true) ** 2))

    def compute_sn34_score(
        self,
        alpha: float = 1.2,
        beta: float = 1.8,
        agg: str = "geomean"
    ) -> float:
        """
        Combined SN34 score from binary MCC and Brier Score.

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
        import math

        # Normalize MCC: [-1, 1] -> [0, 1]
        bin_mcc = float(self.calculate_binary_mcc())
        mcc_norm = max(0.0, min((bin_mcc + 1.0) / 2.0, 1.0)) ** alpha

        # Normalize Brier: [0, 0.25] -> [1, 0] (inverted, lower is better)
        # Then apply beta exponent
        brier = self.calculate_brier()
        # random=0.25 -> 0, perfect=0 -> 1
        brier_score = max(0.0, (0.25 - brier) / 0.25) ** beta

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

