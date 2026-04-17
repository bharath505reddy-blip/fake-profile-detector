"""
drift.py — Model drift detection using Population Stability Index (PSI).

Monitors feature distribution drift between training data and recent predictions.
Tracks prediction confidence trends over time.
Stores drift log in models/drift_log.json and optionally in SQLite.

Usage:
    from drift import check_and_log_drift, get_drift_summary, get_all_platforms_drift_summary
    DriftLog = init_drift(db)  # call once from app.py

PSI thresholds:
    < 0.1  — No significant change (OK)
    0.1–0.2 — Moderate change, monitor (WARN)
    > 0.2  — Significant drift, retrain recommended (ALERT)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_LOG_PATH = Path(__file__).parent / "models" / "drift_log.json"
PSI_THRESHOLD_WARN = 0.1
PSI_THRESHOLD_ALERT = 0.2
PREDICTION_COUNT_CHECK = 100   # check drift after every N predictions
MAX_RECORDS_PER_PLATFORM = 50  # keep last 50 drift checks per platform


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI < 0.1  → no significant change
    PSI 0.1–0.2 → moderate change, monitor
    PSI > 0.2  → significant drift, retrain recommended

    Args:
        expected: Array of values from training/reference distribution.
        actual: Array of values from current/prediction distribution.
        buckets: Number of equal-width bins.

    Returns:
        PSI float value. Returns 0.0 if computation fails.
    """
    try:
        expected = np.array(expected, dtype=float)
        actual = np.array(actual, dtype=float)

        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        combined = np.concatenate([expected, actual])
        min_val, max_val = combined.min(), combined.max()

        if min_val == max_val:
            return 0.0  # Constant feature — no drift possible

        bins = np.linspace(min_val, max_val, buckets + 1)
        bins[-1] += 1e-9  # Include right edge

        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)

        exp_pct = exp_counts / max(len(expected), 1)
        act_pct = act_counts / max(len(actual), 1)

        # Add epsilon to avoid log(0)
        eps = 1e-9
        psi = float(np.sum((act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps))))
        return round(max(psi, 0.0), 6)

    except Exception as exc:
        logger.debug("compute_psi error: %s", exc)
        return 0.0


def compute_feature_drift(
    training_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute PSI for each numeric feature column shared between training and prediction DataFrames.

    Args:
        training_df: DataFrame of training/reference data.
        prediction_df: DataFrame of recent predictions.

    Returns:
        Dict: feature_name -> PSI value. Skips non-numeric/constant/missing columns.
    """
    psi_dict: Dict[str, float] = {}

    # Find shared numeric columns (exclude label and prediction columns)
    skip_cols = {"label", "prediction", "confidence", "anomaly", "anomaly_score",
                 "is_fake", "fake", "target", "is_bot", "bot"}

    train_num = set(training_df.select_dtypes(include=np.number).columns) - skip_cols
    pred_num = set(prediction_df.select_dtypes(include=np.number).columns) - skip_cols
    shared = train_num & pred_num

    for col in sorted(shared):
        try:
            exp = training_df[col].dropna().values
            act = prediction_df[col].dropna().values
            if len(exp) >= 10 and len(act) >= 5:
                psi_dict[col] = compute_psi(exp, act)
        except Exception as exc:
            logger.debug("PSI for column %s failed: %s", col, exc)

    return psi_dict


def get_drift_level(psi: float) -> str:
    """
    Return drift severity level based on PSI thresholds.

    Returns:
        'ok' if PSI < 0.1
        'warn' if 0.1 <= PSI < 0.2
        'alert' if PSI >= 0.2
    """
    if psi >= PSI_THRESHOLD_ALERT:
        return "alert"
    if psi >= PSI_THRESHOLD_WARN:
        return "warn"
    return "ok"


# ---------------------------------------------------------------------------
# Drift log (JSON file + SQLite)
# ---------------------------------------------------------------------------

def _load_raw_log() -> dict:
    """Load raw drift log JSON. Returns empty dict on error."""
    try:
        if DRIFT_LOG_PATH.exists():
            return json.loads(DRIFT_LOG_PATH.read_text())
    except Exception as exc:
        logger.warning("Failed to read drift log: %s", exc)
    return {}


def _save_raw_log(data: dict) -> None:
    """Save drift log JSON to disk."""
    try:
        DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_LOG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.error("Failed to write drift log: %s", exc)


def log_drift_check(
    platform: str,
    feature_psi: Dict[str, float],
    avg_confidence: float,
    prediction_count: int,
) -> None:
    """
    Append a drift check record to drift_log.json.

    Record format:
        timestamp, platform, feature_psi, avg_confidence,
        prediction_count, max_psi, drift_level

    Keeps the last MAX_RECORDS_PER_PLATFORM records per platform.
    """
    max_psi = max(feature_psi.values()) if feature_psi else 0.0
    drift_level = get_drift_level(max_psi)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "feature_psi": feature_psi,
        "avg_confidence": round(float(avg_confidence), 3),
        "prediction_count": prediction_count,
        "max_psi": round(max_psi, 6),
        "drift_level": drift_level,
        "retrain_recommended": drift_level == "alert",
    }

    data = _load_raw_log()
    records = data.get(platform, [])
    records.append(record)
    records = records[-MAX_RECORDS_PER_PLATFORM:]
    data[platform] = records
    _save_raw_log(data)

    logger.info(
        "Drift check logged: platform=%s max_psi=%.4f level=%s confidence=%.1f",
        platform, max_psi, drift_level, avg_confidence
    )


def load_drift_log(platform: Optional[str] = None) -> dict:
    """
    Load drift log data.

    Args:
        platform: If provided, return only that platform's records.
                  If None, return all platforms.

    Returns:
        Dict: platform -> list of drift records.
    """
    data = _load_raw_log()
    if platform:
        return {platform: data.get(platform, [])}
    return data


def get_drift_summary(platform: str) -> dict:
    """
    Get the latest drift summary for a platform.

    Returns:
        Dict with:
          platform, latest_check, max_psi, drift_level,
          drifted_features (list), confidence_trend (list),
          retrain_recommended (bool), prediction_count (int)
        Returns empty dict if no data for the platform.
    """
    data = _load_raw_log()
    records = data.get(platform, [])
    if not records:
        return {}

    latest = records[-1]
    feature_psi = latest.get("feature_psi", {})
    max_psi = latest.get("max_psi", 0.0)

    drifted_features = [
        f for f, psi in feature_psi.items()
        if psi >= PSI_THRESHOLD_WARN
    ]

    confidence_trend = [
        r.get("avg_confidence", 0.0)
        for r in records[-10:]
    ]

    return {
        "platform": platform,
        "latest_check": latest.get("timestamp"),
        "max_psi": max_psi,
        "drift_level": get_drift_level(max_psi),
        "drifted_features": drifted_features,
        "confidence_trend": confidence_trend,
        "retrain_recommended": max_psi >= PSI_THRESHOLD_ALERT,
        "prediction_count": latest.get("prediction_count", 0),
        "feature_psi": feature_psi,
    }


def check_and_log_drift(
    platform: str,
    training_csv_path: Path,
    recent_predictions_df: pd.DataFrame,
    avg_confidence: float,
) -> dict:
    """
    Compute drift between training data and recent predictions, then log and return summary.

    Args:
        platform: Platform name.
        training_csv_path: Path to the training CSV used to build the reference distribution.
        recent_predictions_df: DataFrame of recent prediction inputs (feature values).
        avg_confidence: Average model confidence over recent predictions.

    Returns:
        Drift summary dict (from get_drift_summary).
    """
    try:
        if not Path(training_csv_path).exists():
            logger.warning("Training CSV not found for drift check: %s", training_csv_path)
            return {}

        training_df = pd.read_csv(training_csv_path)

        feature_psi = compute_feature_drift(training_df, recent_predictions_df)

        log_drift_check(
            platform=platform,
            feature_psi=feature_psi,
            avg_confidence=avg_confidence,
            prediction_count=len(recent_predictions_df),
        )

        return get_drift_summary(platform)

    except Exception as exc:
        logger.error("check_and_log_drift error for platform=%s: %s", platform, exc)
        return {}


def get_all_platforms_drift_summary() -> List[dict]:
    """
    Return drift summaries for all platforms that have drift data.
    Sorted by max_psi descending (most drifted first).

    Returns:
        List of drift summary dicts.
    """
    data = _load_raw_log()
    summaries = []
    for platform in data.keys():
        summary = get_drift_summary(platform)
        if summary:
            summaries.append(summary)
    summaries.sort(key=lambda s: s.get("max_psi", 0.0), reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# SQLAlchemy model (optional DB persistence)
# ---------------------------------------------------------------------------

def init_drift(db):
    """
    Initialize the DriftLog SQLAlchemy model using the provided db instance.

    Call once from app.py after db is created:
        DriftLog = init_drift(db)
        with app.app_context():
            db.create_all()

    Returns:
        The DriftLog model class.
    """

    class DriftLog(db.Model):
        """Persists drift check results in SQLite for historical querying."""

        __tablename__ = "drift_log_db"

        id = db.Column(db.Integer, primary_key=True)
        platform = db.Column(db.String(50), nullable=False, index=True)
        timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
        max_psi = db.Column(db.Float, nullable=True)
        drift_level = db.Column(db.String(10), nullable=True)  # ok/warn/alert
        feature_psi_json = db.Column(db.Text, nullable=True)
        avg_confidence = db.Column(db.Float, nullable=True)
        prediction_count = db.Column(db.Integer, nullable=True)
        retrain_recommended = db.Column(db.Boolean, default=False)

        def to_dict(self) -> dict:
            """Serialize to plain dict."""
            return {
                "id": self.id,
                "platform": self.platform,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "max_psi": self.max_psi,
                "drift_level": self.drift_level,
                "feature_psi": json.loads(self.feature_psi_json or "{}"),
                "avg_confidence": self.avg_confidence,
                "prediction_count": self.prediction_count,
                "retrain_recommended": self.retrain_recommended,
            }

    return DriftLog
