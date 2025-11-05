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
        self.multiclass_y_true = []
        self.multiclass_y_pred = []
        self.binary_y_true = []
        self.binary_probs = []
        self.multiclass_probs = []
    
    def update(
        self, 
        true_binary: int,
        pred_binary: int, 
        true_multiclass: int, 
        pred_multiclass: int,
        pred_probs: np.ndarray = None
    ):
        """Update confusion matrix with new prediction."""
        if true_binary == 1 and pred_binary == 1:
            self.true_positives += 1
        elif true_binary == 0 and pred_binary == 0:
            self.true_negatives += 1
        elif true_binary == 0 and pred_binary == 1:
            self.false_positives += 1
        elif true_binary == 1 and pred_binary == 0:
            self.false_negatives += 1
        
        self.multiclass_y_true.append(true_multiclass)
        self.multiclass_y_pred.append(pred_multiclass)

        if pred_probs is not None:
            self.binary_y_true.append(true_binary)
            if len(pred_probs) == 3:
                binary_prob = 1.0 - pred_probs[0]
                self.binary_probs.append(binary_prob)
                self.multiclass_probs.append(pred_probs.copy())
            elif len(pred_probs) == 2:
                self.binary_probs.append(pred_probs[1])
                self.multiclass_probs.append(pred_probs.copy())
            else:
                self.binary_probs.append(pred_probs[0])
                binary_as_multiclass = np.array([1 - pred_probs[0], pred_probs[0]])
                self.multiclass_probs.append(binary_as_multiclass)

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

    def calculate_multiclass_mcc(self) -> float:
        """Calculate Matthews Correlation Coefficient for multiclass classification."""
        if len(self.multiclass_y_true) == 0:
            return 0.0

        try:
            from sklearn.metrics import matthews_corrcoef
            return matthews_corrcoef(self.multiclass_y_true, self.multiclass_y_pred)
        except ImportError:
            logger.warning("sklearn not available, multiclass MCC set to 0")
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate multiclass MCC: {e}")
            return 0.0

    def calculate_binary_cross_entropy(self) -> float:
        """Calculate binary cross-entropy loss."""
        if len(self.binary_y_true) == 0 or len(self.binary_probs) == 0:
            return 0.0

        y_true = np.array(self.binary_y_true)
        y_prob = np.clip(np.array(self.binary_probs), 1e-15, 1 - 1e-15)

        loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return float(loss)

    def calculate_multiclass_cross_entropy(self) -> float:
        """Calculate multiclass cross-entropy loss."""
        if len(self.multiclass_y_true) == 0 or len(self.multiclass_probs) == 0:
            return 0.0
        
        y_true = np.array(self.multiclass_y_true)
        y_probs = np.array(self.multiclass_probs)
        y_probs = np.clip(y_probs, 1e-15, 1 - 1e-15)
        
        n_samples = len(y_true)
        loss = -np.sum(np.log(y_probs[np.arange(n_samples), y_true])) / n_samples
        return float(loss)
    
    def compute_sn34_score(self) -> float:
        """
        Compute combined SN34 benchmark score from MCC and cross-entropy metrics.

        Normalizes both metrics to [0, 1] where higher is better, then combines:
        - MCC: (mcc + 1) / 2  (maps [-1, 1] to [0, 1])
        - CE: exp(-ce)  (maps [0, ∞] to [1, 0])

        Returns:
            float: Combined score in [0, 1], higher is better
        """
        binary_mcc = self.calculate_binary_mcc()
        multiclass_mcc = self.calculate_multiclass_mcc()
        binary_ce = self.calculate_binary_cross_entropy()
        multiclass_ce = self.calculate_multiclass_cross_entropy()

        binary_mcc_norm = (binary_mcc + 1.0) / 2.0
        multiclass_mcc_norm = (multiclass_mcc + 1.0) / 2.0
        mcc_score = (binary_mcc_norm + multiclass_mcc_norm) / 2.0

        binary_ce_norm = np.exp(-binary_ce)
        multiclass_ce_norm = np.exp(-multiclass_ce)
        ce_score = (binary_ce_norm + multiclass_ce_norm) / 2.0

        sn34_score = (mcc_score + ce_score) / 2.0

        return float(sn34_score)


def multiclass_to_binary(label: int) -> int:
    """Convert multiclass label (0=real, 1=synthetic, 2=semisynthetic) to binary (0=real, 1=AI)."""
    return 0 if label == 0 else 1


def update_generator_stats(
    generator_stats: Dict[str, Dict],
    sample: Dict,
    true_binary: int,
    pred_binary: int
) -> None:
    """Update per-generator fooling statistics for gasstation datasets."""
    generator_hotkey = sample.get("generator_hotkey")
    # Skip if no generator info or not synthetic sample or "unknown" generator
    if generator_hotkey is None or true_binary != 1 or str(generator_hotkey).lower() == "unknown":
        return
    
    stats = generator_stats.get(generator_hotkey) or {
        "fooled_count": 0, 
        "not_fooled_count": 0
    }
    
    generator_uid = sample.get("generator_uid")
    if generator_uid is not None and stats.get("uid") is None:
        try:
            stats["uid"] = int(generator_uid)
            # Log first time we see this generator
            logger.debug(f"📝 Tracking generator: {generator_hotkey[:16]}... (UID: {stats['uid']})")
        except Exception as e:
            logger.warning(f"Failed to parse generator_uid: {generator_uid} - {e}")
            pass
    
    if pred_binary == 0:
        stats["fooled_count"] += 1
    else:
        stats["not_fooled_count"] += 1
    
    generator_stats[generator_hotkey] = stats


def calculate_per_source_accuracy(
    valid_datasets: List,
    per_dataset_results: Dict[str, Dict]
) -> Dict[str, Dict[str, Dict]]:
    """Build per-source accuracy structure organized by media type and dataset."""
    per_source_accuracy = {}
    
    for dataset_config in valid_datasets:
        results = per_dataset_results.get(
            dataset_config.name, 
            {"correct": 0, "total": 0, "accuracy": 0.0}
        )
        media_type = dataset_config.media_type
        
        if media_type not in per_source_accuracy:
            per_source_accuracy[media_type] = {}
        
        per_source_accuracy[media_type][dataset_config.name] = {
            "correct": results.get("correct", 0),
            "incorrect": results.get("total", 0) - results.get("correct", 0),
        }
    
    return per_source_accuracy

