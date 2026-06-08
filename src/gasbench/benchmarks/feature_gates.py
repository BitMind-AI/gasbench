"""Feature-space generalization gates for synthetic media detection.

These gates are computed post-hoc from parquet data containing per-sample
embeddings captured from the model's classifier input. They measure whether
the model has learned a general "syntheticness" concept vs. pattern-matched
against specific generator artifacts in the public corpus.

All gates are deterministic — no LLM required. Input is a parquet DataFrame
with columns: status, dataset_name, label, predicted, embedding.

Gate thresholds and coefficient values are intentionally coarse (0.3 / 0.5 /
0.7 / 1.0) to prevent optimising for the coefficient itself.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

logger = logging.getLogger(__name__)

# Fixed random seed for reproducibility across runs.
_RGS_SEED = 42

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_generalization_coefficient(
    df: pd.DataFrame,
    *,
    probe_accuracy_threshold: float = 0.70,
    ges_variance_threshold: float = 0.70,
) -> Dict:
    """Compute the generalization coefficient and per-gate diagnostics.

    Walks a strict decision tree of gates. Stops at the first failure. The
    returned coefficient is one of: 0.3, 0.5, 0.7, 1.0.

    Args:
        df: Parquet DataFrame with columns status, dataset_name, label,
            predicted, embedding.
        probe_accuracy_threshold: Minimum balanced accuracy a linear probe
            trained on public-corpus embeddings must achieve on holdout
            embeddings for Gate 2 to pass (default 0.70).
        ges_variance_threshold: Minimum variance explained by the top-2
            singular components of the generator-mean embedding matrix for
            Gate 3 to pass (default 0.70). Values below ~0.40 indicate
            per-generator memorization.

    Returns:
        Dict with keys: coefficient, gates (per-gate diagnostics).
    """
    gates: Dict[str, Dict] = {}

    # ---- Filter to valid, embedded rows ----
    ok = df[(df["status"] == "ok")].copy()
    if ok.empty:
        return _no_data_result("No rows with status='ok'")

    # Split public vs holdout by dataset_name convention
    is_holdout = ok["dataset_name"].str.contains("-holdout-", na=False)
    df_public = ok[~is_holdout]
    df_holdout = ok[is_holdout]

    # ---- Gate 1: Embeddings available ----
    g1 = _gate_embeddings_available(df_holdout)
    gates["embeddings_available"] = g1
    if not g1["passed"]:
        return {"coefficient": 0.3, "gates": gates}

    # Build embedding matrices
    X_pub, y_pub = _build_embedding_matrix(df_public)
    X_hold, y_hold = _build_embedding_matrix(df_holdout)

    if X_pub is None or X_hold is None:
        return {"coefficient": 0.3, "gates": {
            **gates,
            "rgs": {"passed": False, "error": "Could not build embedding matrices"},
        }}

    # ---- Gate 2: Linear Probe Accuracy ----
    g2 = _gate_probe_accuracy(
        X_pub, y_pub, X_hold, y_hold,
        threshold=probe_accuracy_threshold,
    )
    gates["probe_accuracy"] = g2
    if not g2["passed"]:
        coef = 0.5  # Gate 1 already passed (coverage ≈ 100%)
        return {"coefficient": coef, "gates": gates}

    # ---- Gate 3: GES (Generator Entanglement via SVD) ----
    g3 = _gate_ges_svd(df_holdout, top_k_variance_threshold=ges_variance_threshold)
    gates["ges"] = g3
    if not g3["passed"]:
        return {"coefficient": 0.7, "gates": gates}

    # ---- All gates passed ----
    return {"coefficient": 1.0, "gates": gates}


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def _gate_embeddings_available(df_holdout: pd.DataFrame) -> Dict:
    """Check whether embeddings are present.

    A compliant model with self.classifier produces an embedding for every
    inference call, so coverage is effectively binary: ~100% or ~0%.
    """
    num_ok = len(df_holdout)
    num_with = int(df_holdout["embedding"].notna().sum())
    coverage = num_with / num_ok if num_ok > 0 else 0.0
    return {
        "passed": coverage >= 0.999,  # effectively 100%
        "coverage": float(coverage),
        "num_holdout_samples": num_ok,
        "num_with_embeddings": num_with,
    }


def _build_embedding_matrix(
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Stack embeddings into (n, d) matrix. Returns (None, None) on failure."""
    embedded = df[df["embedding"].notna()]
    if embedded.empty:
        return None, None

    emb_list = embedded["embedding"].tolist()
    try:
        X = np.stack(emb_list).astype(np.float32)
    except (ValueError, TypeError):
        return None, None

    y = (
        embedded["label"]
        .astype(int)
        .map({0: 0, 1: 1, 2: 1})  # semisynthetic → synthetic
        .values
    )

    return X, y


def _gate_probe_accuracy(
    X_pub: np.ndarray,
    y_pub: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    threshold: float,
    num_repeats: int = 1,
) -> Dict:
    """Linear probe feature-quality gate.

    Trains a fixed-hyperparameter logistic regression on public-corpus
    embeddings, then evaluates its balanced accuracy on holdout embeddings.
    A high score means the model's features are linearly separable by
    real/synthetic in a way that transfers across data distributions —
    i.e. the features are genuinely useful, not overfitted to specific
    generators.

    Design decisions:
    - Data: subsample the larger of public/holdout to match the smaller
      (stratified) so the probe has no unfair sample-count advantage.
    - Hyperparams: fixed (C=1.0, L2, max_iter=2000). This is a measurement
      instrument, not a model to tune.
    - Metric: balanced accuracy to handle class imbalance.
    """
    n_pub = len(y_pub)
    n_hold = len(y_hold)

    if n_pub == 0 or n_hold == 0:
        return {"passed": False, "error": "Empty public or holdout data"}

    target_n = min(n_pub, n_hold)
    probe_accs = []

    for repeat in range(num_repeats):
        seed = _RGS_SEED + repeat

        X_pub_sub, y_pub_sub = _subsample_stratified(
            X_pub, y_pub, target_n, random_state=seed
        )
        X_hold_sub, y_hold_sub = _subsample_stratified(
            X_hold, y_hold, target_n, random_state=seed + 1000
        )

        scaler = StandardScaler()
        X_pub_s = scaler.fit_transform(X_pub_sub)
        X_hold_s = scaler.transform(X_hold_sub)

        probe = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
        probe.fit(X_pub_s, y_pub_sub)

        y_pred = probe.predict(X_hold_s)
        probe_accs.append(float(balanced_accuracy_score(y_hold_sub, y_pred)))

    probe_bal_acc = float(np.mean(probe_accs))
    probe_std = float(np.std(probe_accs)) if num_repeats > 1 else 0.0

    return {
        "passed": probe_bal_acc >= threshold,
        "value": probe_bal_acc,
        "threshold": threshold,
        "probe_std": probe_std,
        "num_public_original": n_pub,
        "num_holdout_original": n_hold,
        "training_size": target_n,
        "num_repeats": num_repeats,
        "embedding_dim": int(X_pub.shape[1]),
    }


def _gate_ges_svd(
    df_holdout: pd.DataFrame,
    top_k_variance_threshold: float = 0.70,
    min_generators: int = 2,
) -> Dict:
    """Generator Entanglement Score via SVD on generator-mean embeddings.

    Builds a matrix where each row is the mean penultimate embedding for one
    holdout generator (unit-normalised). SVD decomposes this matrix to
    measure how many orthogonal directions span the generator subspace.

    - Top-2 components explain > 70% of variance → all generators point
      in roughly the same direction → unified "synthetic" concept.
    - Top-5 components explain < 40% of variance → generators occupy
      many orthogonal directions → per-generator detector circuits.

    This is a simpler version of the method in S³ (2606.01843), which
    trains a linear probe for forgery method classification and SVDs the
    weight matrix to extract the dominant shortcut subspace.

    Reference:
      Suppressing Forgery-Specific Shortcuts (S³), arXiv 2606.01843
      — Section 3.2: "Shortcut Subspace Extraction via SVD"
    """
    embedded = df_holdout[df_holdout["embedding"].notna()]
    if embedded.empty:
        return {"passed": False, "error": "No holdout embeddings"}

    emb_list = embedded["embedding"].tolist()
    try:
        all_emb = np.stack(emb_list).astype(np.float32)
    except (ValueError, TypeError):
        return {"passed": False, "error": "Could not stack embeddings"}

    # Unit-normalise each embedding for direction-only comparison
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    all_emb = all_emb / norms

    # Per-generator mean embeddings — group by generator_family if present
    group_key = "generator_family" if "generator_family" in embedded.columns and embedded["generator_family"].notna().any() else "dataset_name"
    generator_means: Dict[str, np.ndarray] = {}
    for key, grp in embedded.groupby(group_key):
        key = str(key) if group_key == "generator_family" else key
        grp_emb = np.stack(grp["embedding"].tolist()).astype(np.float32)
        grp_norms = np.linalg.norm(grp_emb, axis=1, keepdims=True)
        grp_norms = np.where(grp_norms == 0, 1.0, grp_norms)
        grp_emb = grp_emb / grp_norms
        generator_means[key] = grp_emb.mean(axis=0)

    n = len(generator_means)
    if n < min_generators:
        return {
            "passed": True,
            "value": 1.0,
            "top_k_variance_threshold": top_k_variance_threshold,
            "note": f"Only {n} holdout generator(s), GES undefined — gate skipped",
            "num_generators": n,
        }

    # SVD on the generator-mean matrix (n_generators × embedding_dim)
    means_matrix = np.stack(list(generator_means.values()))
    _, S, _ = np.linalg.svd(means_matrix, full_matrices=False)

    # Variance explained by top-2 singular components
    total_var = float(np.sum(S ** 2))
    top2_var = float(np.sum(S[:2] ** 2)) if len(S) >= 2 else total_var
    top2_ratio = top2_var / total_var if total_var > 0 else 0.0

    return {
        "passed": top2_ratio >= top_k_variance_threshold,
        "value": top2_ratio,
        "top_k_variance_threshold": top_k_variance_threshold,
        "singular_values": S.tolist(),
        "num_generators": int(n),
        "generators": list(generator_means.keys()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subsample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    random_state: int = _RGS_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample X, y to exactly n samples, stratified by label."""
    if len(y) <= n:
        return X, y
    return resample(
        X, y,
        replace=False,
        n_samples=n,
        stratify=y,
        random_state=random_state,
    )


def _no_data_result(reason: str) -> Dict:
    return {
        "coefficient": 0.0,
        "gates": {
            "embeddings_available": {"passed": False, "error": reason},
        },
    }


def embedding_stats(df: pd.DataFrame) -> Dict:
    """Quick stats about embeddings in a parquet file. Useful for debugging."""
    ok = df[df["status"] == "ok"]
    emb_ok = ok[ok["embedding"].notna()]

    if emb_ok.empty:
        return {"embeddings_available": False, "total_ok": len(ok)}

    dims = emb_ok["embedding"].apply(lambda e: len(e) if isinstance(e, list) else 0)
    return {
        "embeddings_available": True,
        "total_ok": len(ok),
        "with_embeddings": len(emb_ok),
        "coverage": float(len(emb_ok) / len(ok)) if len(ok) > 0 else 0.0,
        "embedding_dim": int(dims.mode().iloc[0]),
    }
