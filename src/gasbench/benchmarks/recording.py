import uuid
import time
import json
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .utils import Metrics


class BenchmarkRunRecorder:
    def __init__(
        self,
        run_id: Optional[str] = None,
        mode: Optional[str] = None,
        modality: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
        model_input_name: Optional[str] = None,
        model_name: Optional[str] = None,
        augment_level: Optional[int] = 0,
        crop_prob: float = 0.0,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.run_started_at = int(time.time())
        self.mode = mode
        self.modality = modality
        self.target_height = int(target_size[0]) if target_size else None
        self.target_width = int(target_size[1]) if target_size else None
        self.model_input_name = model_input_name
        self.model_name = model_name
        self.augment_level = augment_level
        self.crop_prob = crop_prob
        self.rows: List[Dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.rows)

    def add_ok(
        self,
        *,
        dataset_name: str,
        sample_index: int,
        sample: Dict[str, Any],
        label: int,
        predicted: int,
        probs: Any,
        inference_time_ms: float,
        batch_inference_time_ms: float,
        batch_id: int,
        batch_size: int,
        sample_seed: Optional[int],
    ):
        # Normalize probabilities to a list of floats for parquet friendliness
        try:
            probs_list = [
                float(x)
                for x in (probs.tolist() if hasattr(probs, "tolist") else list(probs))
            ]
        except Exception:
            probs_list = []
        row = {
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
            "mode": self.mode,
            "modality": self.modality,
            "model_name": self.model_name,
            "input_name": self.model_input_name,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "augment_level": self.augment_level,
            "crop_prob": self.crop_prob,
            "dataset_name": dataset_name,
            "iteration_index": int(sample_index),
            "media_type": sample.get("media_type"),
            "status": "ok",
            "label": int(label),
            "predicted": int(predicted),
            "probs": probs_list,
            "correct": bool(predicted == label),
            "inference_time_ms": float(inference_time_ms),
            "batch_inference_time_ms": float(batch_inference_time_ms),
            "batch_id": int(batch_id),
            "batch_size": int(batch_size),
            "sample_seed": None if sample_seed is None else int(sample_seed),
            "skip_reason": None,
            "error_message": None,
        }

        row.update(
            {
                "source_kind": sample.get("source_kind"),
                "dataset_path": sample.get("dataset_path"),
                "hf_resolved_revision": sample.get("hf_resolved_revision"),
                "archive_filename": sample.get("archive_filename"),
                "path_in_archive": sample.get("member_path"),
                "source_file": sample.get("source_file"),
                "iso_week": sample.get("iso_week"),
                "cache_relpath": sample.get("cache_relpath"),
                "generator_hotkey": sample.get("generator_hotkey"),
                "generator_uid": sample.get("generator_uid"),
                "generator_name": sample.get("generator_name"),
                "generator_model": sample.get("model_name"),
            }
        )

        # Derived convenience identifiers
        row["sample_id"] = build_sample_id(row)
        row["sample_compound_id"] = build_compound_id(row)
        row["sample_display_uri"] = build_display_uri(row)

        self.rows.append(row)

    def add_skip(
        self,
        *,
        dataset_name: str,
        sample_index: int,
        sample: Dict[str, Any],
        reason: str,
    ):
        row = {
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
            "mode": self.mode,
            "modality": self.modality,
            "model_name": self.model_name,
            "input_name": self.model_input_name,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "augment_level": self.augment_level,
            "crop_prob": self.crop_prob,
            "dataset_name": dataset_name,
            "iteration_index": int(sample_index),
            "media_type": sample.get("media_type"),
            "status": "skipped",
            "label": None,
            "predicted": None,
            "probs": None,
            "correct": None,
            "inference_time_ms": None,
            "batch_inference_time_ms": None,
            "batch_id": None,
            "batch_size": None,
            "sample_seed": None,
            "skip_reason": reason,
            "error_message": None,
        }
        row.update(
            {
                "source_kind": sample.get("source_kind"),
                "dataset_path": sample.get("dataset_path"),
                "hf_resolved_revision": sample.get("hf_resolved_revision"),
                "archive_filename": sample.get("archive_filename"),
                "path_in_archive": sample.get("member_path"),
                "source_file": sample.get("source_file"),
                "iso_week": sample.get("iso_week"),
                "cache_relpath": sample.get("cache_relpath"),
                "generator_hotkey": sample.get("generator_hotkey"),
                "generator_uid": sample.get("generator_uid"),
                "generator_name": sample.get("generator_name"),
                "generator_model": sample.get("model_name"),
            }
        )
        row["sample_id"] = build_sample_id(row)
        row["sample_compound_id"] = build_compound_id(row)
        row["sample_display_uri"] = build_display_uri(row)
        self.rows.append(row)

    def add_error(
        self,
        *,
        dataset_name: str,
        sample_index: int,
        sample: Dict[str, Any],
        error_message: str,
    ):
        row = {
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
            "mode": self.mode,
            "modality": self.modality,
            "model_name": self.model_name,
            "input_name": self.model_input_name,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "augment_level": self.augment_level,
            "crop_prob": self.crop_prob,
            "dataset_name": dataset_name,
            "iteration_index": int(sample_index),
            "media_type": sample.get("media_type"),
            "status": "error",
            "label": None,
            "predicted": None,
            "probs": None,
            "correct": None,
            "inference_time_ms": None,
            "batch_inference_time_ms": None,
            "batch_id": None,
            "batch_size": None,
            "sample_seed": None,
            "skip_reason": None,
            "error_message": error_message[:300],
        }
        row.update(
            {
                "source_kind": sample.get("source_kind"),
                "dataset_path": sample.get("dataset_path"),
                "hf_resolved_revision": sample.get("hf_resolved_revision"),
                "archive_filename": sample.get("archive_filename"),
                "path_in_archive": sample.get("member_path"),
                "source_file": sample.get("source_file"),
                "iso_week": sample.get("iso_week"),
                "cache_relpath": sample.get("cache_relpath"),
                "generator_hotkey": sample.get("generator_hotkey"),
                "generator_uid": sample.get("generator_uid"),
                "generator_name": sample.get("generator_name"),
                "generator_model": sample.get("model_name"),
            }
        )
        row["sample_id"] = build_sample_id(row)
        row["sample_compound_id"] = build_compound_id(row)
        row["sample_display_uri"] = build_display_uri(row)
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(self.rows)
        return df

    def write_parquet(self, path: str) -> str:
        df = self.to_dataframe()
        if df.empty:
            return path
        try:
            dirpath = os.path.dirname(path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        except Exception:
            pass
        df.to_parquet(path, index=False)
        return path


def build_compound_id(row: Dict[str, Any]) -> Optional[str]:
    try:
        source = (row.get("source_kind") or "").lower()
        if source in ("hf", "huggingface"):
            repo = row.get("dataset_path")
            rev = row.get("hf_resolved_revision")
            archive_filename = row.get("archive_filename")
            path_in_archive = row.get("path_in_archive")
            source_file = row.get("source_file")
            
            if path_in_archive and archive_filename:
                return f"hf://{repo}@{rev or 'main'}::{archive_filename}#{path_in_archive}"
            if archive_filename:
                return f"hf://{repo}@{rev or 'main'}::{archive_filename}"
            if source_file:
                return f"hf://{repo}@{rev or 'main'}::{source_file}"
            return f"hf://{repo}@{rev or 'main'}"
        if source == "r2":
            bucket = row.get("r2_bucket")
            key = row.get("r2_key")
            version = row.get("r2_version_id") or row.get("r2_etag")
            path_in_archive = row.get("path_in_archive")
            base = f"r2://{bucket}/{key}" if bucket and key else None
            if not base:
                return None
            if version:
                base = f"{base}@{version}"
            if path_in_archive:
                base = f"{base}#{path_in_archive}"
            return base
    except Exception:
        return None
    return None


def build_display_uri(row: Dict[str, Any]) -> Optional[str]:
    try:
        source = (row.get("source_kind") or "").lower()
        if source in ("hf", "huggingface"):
            repo = row.get("dataset_path")
            archive_filename = row.get("archive_filename")
            path_in_archive = row.get("path_in_archive")
            source_file = row.get("source_file")
            
            if path_in_archive and archive_filename:
                return f"{repo}::{archive_filename}#{path_in_archive}"
            if archive_filename:
                return f"{repo}::{archive_filename}"
            if source_file:
                return f"{repo}::{source_file}"
            return f"{repo}"
        if source == "r2":
            bucket = row.get("r2_bucket")
            key = row.get("r2_key")
            path_in_archive = row.get("path_in_archive")
            base = f"{bucket}/{key}" if bucket and key else None
            if not base:
                return None
            if path_in_archive:
                base = f"{base}#{path_in_archive}"
            return base
    except Exception:
        return None
    return None


def build_sample_id(row: Dict[str, Any]) -> str:
    source_kind = row.get("source_kind") or ""
    dataset_path = row.get("dataset_path") or ""
    archive_filename = row.get("archive_filename") or ""
    path_in_archive = row.get("path_in_archive") or ""
    source_file = row.get("source_file") or ""
    
    id_string = f"{source_kind}::{dataset_path}::{archive_filename}::{path_in_archive}::{source_file}"
    return hashlib.sha256(id_string.encode()).hexdigest()[:16]


def compute_metrics_from_df(
    df: pd.DataFrame, holdout_weight: float = 1.0
) -> Dict[str, Any]:
    """Compute benchmark metrics from a DataFrame of results.

    Args:
        df: DataFrame containing benchmark results with 'status', 'correct',
            'dataset_name', etc.
        holdout_weight: Weight multiplier for holdout dataset samples when
            computing the benchmark_score. Holdout datasets are identified by
            having '-holdout-' in their name. Regular datasets get weight=1.0,
            holdout datasets get weight=holdout_weight. Default is 1.0 (equal
            weighting).

    Returns:
        Dict with benchmark_score, timing metrics, and other computed metrics.
    """
    result: Dict[str, Any] = {}
    if df is None or df.empty:
        return {
            "benchmark_score": 0.0,
            "avg_inference_time_ms": 0.0,
            "p95_inference_time_ms": 0.0,
            "binary_mcc": 0.0,
            "binary_cross_entropy": 0.0,
            "sn34_score": 0.0,
        }

    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        return {
            "benchmark_score": 0.0,
            "avg_inference_time_ms": 0.0,
            "p95_inference_time_ms": 0.0,
            "binary_mcc": 0.0,
            "binary_cross_entropy": 0.0,
            "sn34_score": 0.0,
        }

    # Compute weighted accuracy based on holdout_weight
    # Holdout datasets are identified by having '-holdout-' in the name
    if holdout_weight != 1.0 and "dataset_name" in ok_df.columns:
        # Assign weights: holdout datasets get holdout_weight, others get 1.0
        is_holdout = ok_df["dataset_name"].str.contains("-holdout-", na=False)
        weights = np.where(is_holdout, holdout_weight, 1.0)
        
        # Weighted accuracy: sum(correct * weight) / sum(weight)
        correct_arr = ok_df["correct"].astype(float).values
        weighted_correct = (correct_arr * weights).sum()
        total_weight = weights.sum()
        accuracy = float(weighted_correct / total_weight) if total_weight > 0 else 0.0
    else:
        # Standard unweighted accuracy
        accuracy = float(ok_df["correct"].mean()) if "correct" in ok_df else 0.0

    times = ok_df.get("inference_time_ms")
    avg_time = float(times.mean()) if times is not None else 0.0
    p95_time = float(np.percentile(times, 95)) if times is not None else 0.0

    metrics = Metrics()
    for _, r in ok_df.iterrows():
        try:
            probs = [
                float(x)
                for x in (
                    r["probs"].tolist()
                    if hasattr(r["probs"], "tolist")
                    else list(r["probs"])
                )
            ]
        except Exception:
            probs = []
        label = int(r["label"])
        pred = int(r["predicted"])
        metrics.update(
            label,
            pred,
            probs,
        )

    result.update(
        {
            "benchmark_score": accuracy,
            "avg_inference_time_ms": avg_time,
            "p95_inference_time_ms": p95_time,
            "binary_mcc": metrics.calculate_binary_mcc(),
            "binary_cross_entropy": metrics.calculate_binary_cross_entropy(),
            "binary_brier": metrics.calculate_brier(),
            "sn34_score": metrics.compute_sn34_score(),
        }
    )
    return result


def compute_per_dataset_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    ok_df = df[df["status"] == "ok"]
    if ok_df.empty:
        return {}

    per_ds: Dict[str, Any] = {}
    grouped = ok_df.groupby("dataset_name", dropna=False)
    for ds, g in grouped:
        total = int(len(g))
        correct = int(g["correct"].sum())
        accuracy = (correct / total) if total > 0 else 0.0
        counts = g["predicted"].value_counts().to_dict()
        predictions = {
            "real": int(counts.get(0, 0)),
            "synthetic": int(counts.get(1, 0)),
        }
        per_ds[ds] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "predictions": predictions,
        }
    return per_ds


def compute_generator_stats_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    ok_df = df[df["status"] == "ok"].copy()
    key = None
    for candidate in ["generator_hotkey", "generator_uid"]:
        if candidate in ok_df.columns and ok_df[candidate].notna().any():
            key = candidate
            break
    if not key:
        return {}

    stats: Dict[str, Any] = {}
    grouped = ok_df.groupby(key)
    for k, g in grouped:
        total = int(len(g))
        correct = int(g["correct"].sum())
        stats[str(k)] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }
    return stats


def summarize_dataset_from_tracker(
    tracker: BenchmarkRunRecorder,
    dataset_name: str,
    include_skipped: bool = False,
) -> Dict[str, Any]:
    """
    Compute minimal per-dataset summary from the tracker's dataframe.
    Returns a dict with accuracy, correct, total, and optionally skipped.
    """
    df_now = tracker.to_dataframe()
    if df_now.empty:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            **({"skipped": 0} if include_skipped else {}),
        }

    ds_ok = df_now[
        (df_now.get("status") == "ok") & (df_now.get("dataset_name") == dataset_name)
    ]
    total_ds = int(len(ds_ok))
    correct_ds = int(ds_ok["correct"].sum()) if total_ds > 0 else 0
    summary: Dict[str, Any] = {
        "accuracy": (correct_ds / total_ds) if total_ds > 0 else 0.0,
        "correct": correct_ds,
        "total": total_ds,
    }
    if include_skipped:
        ds_skipped = int(
            (
                (df_now.get("status") == "skipped")
                & (df_now.get("dataset_name") == dataset_name)
            ).sum()
        )
        summary["skipped"] = ds_skipped
    return summary


def log_dataset_summary(
    logger,
    tracker: BenchmarkRunRecorder,
    dataset_name: str,
    include_skipped: bool = False,
):
    """
    Log a minimal per-dataset summary using the shared format.
    """
    s = summarize_dataset_from_tracker(
        tracker, dataset_name, include_skipped=include_skipped
    )
    if include_skipped:
        logger.info(
            f"Dataset {dataset_name}: {s['accuracy']:.2%} accuracy ({s['correct']}/{s['total']}), skipped: {s.get('skipped', 0)}"
        )
    else:
        logger.info(
            f"Dataset {dataset_name}: {s['accuracy']:.2%} accuracy ({s['correct']}/{s['total']})"
        )


def compute_generator_fool_rates_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute fool_rate per generator_hotkey:
    fool_rate = fraction of generator-tagged samples predicted as 0 (real).
    """
    if df is None or df.empty:
        return {}
    ok_df = df[(df.get("status") == "ok")].copy()
    if (
        "generator_hotkey" not in ok_df.columns
        or ok_df["generator_hotkey"].isna().all()
    ):
        return {}
    result: Dict[str, float] = {}
    grouped = ok_df.groupby("generator_hotkey")
    for hotkey, g in grouped:
        total = len(g)
        if total <= 0:
            continue
        fooled = int((g["predicted"] == 0).sum())
        result[str(hotkey)] = fooled / total
    return result


def write_cache_policy_from_df(df: pd.DataFrame, policy_path: str) -> Dict[str, float]:
    """
    Write a cache policy JSON with generator_priorities equal to fool_rate per generator.
    Returns the mapping written.
    """
    priorities = compute_generator_fool_rates_from_df(df)
    policy = {"generator_priorities": priorities}
    try:
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    except Exception:
        pass
    with open(policy_path, "w") as f:
        json.dump(policy, f, indent=2)
    return priorities
