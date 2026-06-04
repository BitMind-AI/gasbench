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
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_generalization_coefficient(
    df: pd.DataFrame,
    *,
    rgs_threshold: float = 0.85,
    ges_threshold: float = 0.5,
    min_embedding_coverage: float = 0.8,
) -> Dict:
    """Compute the generalization coefficient and per-gate diagnostics.

    Walks a strict decision tree of gates. Stops at the first failure. The
    returned coefficient is one of: 0.3, 0.5, 0.7, 1.0.

    Args:
        df: Parquet DataFrame with columns status, dataset_name, label,
            predicted, embedding.
        rgs_threshold: Minimum RGS for Gate 2 to pass.
        ges_threshold: Minimum GES for Gate 3 to pass.
        min_embedding_coverage: Minimum fraction of holdout samples that must
            have non-null embeddings for Gate 1 to pass.

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
    g1 = _gate_embeddings_available(df_holdout, min_embedding_coverage)
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

    # ---- Gate 2: RGS (Linear Probe Transfer Gap) ----
    g2 = _gate_rgs(X_pub, y_pub, X_hold, y_hold, rgs_threshold)
    gates["rgs"] = g2
    if not g2["passed"]:
        coef = 0.5 if g1["coverage"] >= min_embedding_coverage else 0.3
        return {"coefficient": coef, "gates": gates}

    # ---- Gate 3: GES (Generator Entanglement Score) ----
    g3 = _gate_ges(df_holdout, ges_threshold)
    gates["ges"] = g3
    if not g3["passed"]:
        return {"coefficient": 0.7, "gates": gates}

    # ---- All gates passed ----
    return {"coefficient": 1.0, "gates": gates}


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def _gate_embeddings_available(
    df_holdout: pd.DataFrame, min_coverage: float
) -> Dict:
    coverage = df_holdout["embedding"].notna().mean() if len(df_holdout) > 0 else 0.0
    return {
        "passed": coverage >= min_coverage,
        "coverage": float(coverage),
        "threshold": min_coverage,
        "num_holdout_samples": len(df_holdout),
        "num_with_embeddings": int(df_holdout["embedding"].notna().sum()),
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


def _gate_rgs(
    X_pub: np.ndarray,
    y_pub: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    threshold: float,
) -> Dict:
    """Linear Probe Transfer Gap: how well does a linear probe trained on
    public-corpus embeddings generalise to holdout embeddings?"""
    try:
        scaler = StandardScaler()
        X_pub_s = scaler.fit_transform(X_pub)
        X_hold_s = scaler.transform(X_hold)

        probe = LogisticRegression(max_iter=2000, random_state=42)
        probe.fit(X_pub_s, y_pub)

        probe_acc = float(probe.score(X_hold_s, y_hold))

        return {
            "passed": probe_acc >= threshold,
            "value": probe_acc,
            "threshold": threshold,
            "num_public": len(y_pub),
            "num_holdout": len(y_hold),
            "embedding_dim": int(X_pub.shape[1]),
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def _gate_ges(df_holdout: pd.DataFrame, threshold: float) -> Dict:
    """Generator Entanglement Score: mean pairwise cosine similarity between
    per-generator mean embeddings. High GES = all generators point in the
    same "synthetic" direction. Low GES = per-generator detector circuits."""
    embedded = df_holdout[df_holdout["embedding"].notna()]
    if embedded.empty:
        return {"passed": False, "error": "No holdout embeddings"}

    emb_list = embedded["embedding"].tolist()
    try:
        all_emb = np.stack(emb_list).astype(np.float32)
    except (ValueError, TypeError):
        return {"passed": False, "error": "Could not stack embeddings"}

    # Normalise to unit length for cosine similarity
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    all_emb = all_emb / norms

    # Per-generator mean embeddings
    generator_means: Dict[str, np.ndarray] = {}
    for ds_name, grp in embedded.groupby("dataset_name"):
        grp_emb = np.stack(grp["embedding"].tolist()).astype(np.float32)
        grp_norms = np.linalg.norm(grp_emb, axis=1, keepdims=True)
        grp_norms = np.where(grp_norms == 0, 1.0, grp_norms)
        grp_emb = grp_emb / grp_norms
        generator_means[ds_name] = grp_emb.mean(axis=0)

    means_matrix = np.stack(list(generator_means.values()))
    sim_matrix = means_matrix @ means_matrix.T

    # Upper triangle (excluding diagonal)
    n = len(generator_means)
    if n < 2:
        # Single generator — can't measure entanglement, skip gate
        return {
            "passed": True,
            "value": 1.0,
            "threshold": threshold,
            "note": "Only 1 holdout generator, GES undefined — gate skipped",
            "num_generators": 1,
        }

    triu = sim_matrix[np.triu_indices(n, k=1)]
    ges = float(triu.mean())

    return {
        "passed": ges >= threshold,
        "value": ges,
        "threshold": threshold,
        "num_generators": int(n),
        "generators": list(generator_means.keys()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
