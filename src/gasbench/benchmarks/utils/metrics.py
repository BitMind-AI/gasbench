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
        y_prob = np.clip(np.array(self.binary_probs), 1e-7, 1 - 1e-7)

        loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return float(loss)

    def calculate_multiclass_cross_entropy(self) -> float:
        """Calculate multiclass cross-entropy loss."""
        if len(self.multiclass_y_true) == 0 or len(self.multiclass_probs) == 0:
            return 0.0
        
        y_true = np.array(self.multiclass_y_true)
        y_probs = np.array(self.multiclass_probs)
        y_probs = np.clip(y_probs, 1e-7, 1 - 1e-7)
        
        n_samples = len(y_true)
        loss = -np.sum(np.log(y_probs[np.arange(n_samples), y_true])) / n_samples
        return float(loss)
    
    def compute_sn34_score(self, alpha: float = 1.0, beta: float = 1.0,
                        agg: str = "geomean") -> float:
        """
        Combined SN34 score from MCC and Cross-Entropy with principled normalization.

        - MCC in [-1,1] -> mcc_score in [0,1] via (mcc+1)/2, then sharpen via alpha.
        - CE -> exp(-CE) = 1/perplexity; rescale so random=0 and perfect=1:
            ce_score = (exp(-CE) - 1/K) / (1 - 1/K), then sharpen via beta.
        - Aggregate binary + multiclass with sample-count weights.
        - Combine MCC and CE via geometric mean (default) to penalize imbalance.

        Args:
            alpha: exponent on MCC score (>=1 boosts top-end separation).
            beta: exponent on CE score (>=1 boosts top-end separation).
            agg: "geomean" | "harmmean" | "mean" for combining MCC vs CE.

        Returns:
            float in [0,1].
        """
        # --- metrics ---
        bin_mcc = float(self.calculate_binary_mcc())           # [-1, 1]
        mul_mcc = float(self.calculate_multiclass_mcc())       # [-1, 1]
        bin_ce  = float(self.calculate_binary_cross_entropy()) # >= 0
        mul_ce  = float(self.calculate_multiclass_cross_entropy()) # >= 0

        # --- sample counts for micro-weighting ---
        n_bin = int(getattr(self, "n_binary_samples", 1))
        n_mul = int(getattr(self, "n_multiclass_samples", 1))
        w_bin = n_bin / max(n_bin + n_mul, 1)
        w_mul = 1.0 - w_bin

        # --- safe clamps ---
        import math
        def safe_exp_neg(x):  # clamp CE to avoid under/overflow
            x = max(0.0, min(x, 20.0))
            return math.exp(-x)

        # --- MCC normalization to [0,1] ---
        bin_mcc_s = max(0.0, min((bin_mcc + 1.0) / 2.0, 1.0))
        mul_mcc_s = max(0.0, min((mul_mcc + 1.0) / 2.0, 1.0))
        mcc_score = w_bin * bin_mcc_s + w_mul * mul_mcc_s
        mcc_score = mcc_score ** alpha

        # --- CE -> random=0, perfect=1 via K-aware rescale ---
        # binary: K=2; your multiclass: K=3 (0,1,2)
        def ce_to_score(ce: float, K: int) -> float:
            base = 1.0 / K                # value for random predictor
            val  = safe_exp_neg(ce)       # 1/perplexity
            if K <= 1:                    # degenerate safeguard
                return 1.0 if ce == 0.0 else 0.0
            num = max(0.0, val - base)
            den = 1.0 - base
            return max(0.0, min(num / den, 1.0))

        bin_ce_s = ce_to_score(bin_ce, K=2)
        mul_ce_s = ce_to_score(mul_ce, K=3)
        ce_score = (w_bin * bin_ce_s + w_mul * mul_ce_s) ** beta

        # --- combine MCC vs CE ---
        if agg == "geomean":
            final = math.sqrt(max(1e-12, mcc_score * ce_score))
        elif agg == "harmmean":
            denom = max(1e-12, (mcc_score + ce_score))
            final = 2.0 * mcc_score * ce_score / denom
        else:  # arithmetic mean
            final = 0.5 * (mcc_score + ce_score)

        return float(max(0.0, min(final, 1.0)))



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
    """Build per-source prediction distribution organized by media type and dataset.

    For each dataset, returns counts of predictions by label:
      { "real": N_real, "synthetic": N_synth, "semisynthetic": N_semi }
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
        }
    
    return per_source_accuracy

