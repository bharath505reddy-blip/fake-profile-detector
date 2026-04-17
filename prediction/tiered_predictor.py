"""
prediction/tiered_predictor.py — Two-stage tiered prediction for single profiles.

Stage 1: Quick scan using base features → return if confidence >= 0.80.
Stage 2: Deep analysis (live enrichment + photo + username + bio + behavioral +
          cross-platform) → re-predict with enriched feature set.

Confidence calibration uses isotonic regression stored alongside the model.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Confidence labels
_CONFIDENCE_LABELS = [
    (0.90, "Very High Confidence"),
    (0.80, "High Confidence"),
    (0.65, "Moderate Confidence"),
    (0.50, "Low Confidence"),
    (0.40, "Uncertain"),
    (0.00, "Uncertain"),
]

QUICK_THRESHOLD = 0.85   # If confidence ≥ this, skip Stage 2
LOW_CONFIDENCE_THRESHOLD = 0.40  # Below this, still return quick result

# How much Stage 2 confidence can DROP vs Stage 1 before we fall back
MAX_CONFIDENCE_DEGRADATION = 0.15  # 15 percentage points


def confidence_label(confidence_pct: float) -> str:
    """Map a confidence percentage (0-100) to a human-readable label."""
    conf_frac = confidence_pct / 100.0
    for threshold, label in _CONFIDENCE_LABELS:
        if conf_frac >= threshold:
            return label
    return "Uncertain"


def _run_model(
    df: pd.DataFrame,
    model_path: Path,
    feature_builder: Callable,
) -> Tuple[str, float]:
    """
    Run the saved model on a single-row DataFrame.
    Returns (prediction_label, confidence_pct).
    """
    from ML.persistent_common import predict_with_saved_model
    from pathlib import Path
    import os

    charts_dir = Path(__file__).parent.parent / "static" / "charts"
    tag = os.urandom(6).hex()

    df_pred, _, _, _ = predict_with_saved_model(
        df=df,
        model_path=model_path,
        charts_dir=charts_dir,
        tag=tag,
        title="",
        feature_builder=feature_builder,
        skip_charts=True,
        include_shap=False,
        include_anomaly=False,
    )

    prediction = str(df_pred["prediction"].iloc[0])
    raw_conf = df_pred["confidence"].iloc[0] if "confidence" in df_pred.columns else 50.0
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 50.0

    return prediction, confidence


def _run_model_with_shap(
    df: pd.DataFrame,
    model_path: Path,
    feature_builder: Callable,
    tag: str,
    charts_dir: Path,
) -> Tuple[str, float, str, list]:
    """
    Run model with SHAP chart generation.
    Returns (prediction, confidence, shap_chart_path, feature_importance_list).
    """
    from ML.persistent_common import predict_with_saved_model, SHAP_AVAILABLE
    import os

    df_pred, _, chart_files, warnings = predict_with_saved_model(
        df=df,
        model_path=model_path,
        charts_dir=charts_dir,
        tag=tag,
        title="",
        feature_builder=feature_builder,
        skip_charts=False,
        include_shap=SHAP_AVAILABLE,
        include_anomaly=False,
    )

    prediction = str(df_pred["prediction"].iloc[0])
    raw_conf = df_pred["confidence"].iloc[0] if "confidence" in df_pred.columns else 50.0
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 50.0

    shap_chart = chart_files.get("shap_single", "")
    return prediction, confidence, shap_chart, warnings


def should_escalate_to_stage2(
    confidence: float,
    profile_data: dict,
    enable_deep_analysis: bool,
) -> bool:
    """
    Determine whether Stage 2 deep analysis is worth running.

    Only escalate when:
      - Confidence is in the uncertain zone (40–85%)
      - We have a username to enrich
      - Deep analysis is enabled

    Skip escalation when:
      - Already high confidence (no value added)
      - Very low confidence AND no enrichable data (waste of time)
      - No username (live enrichment won't work)
    """
    if not enable_deep_analysis:
        return False

    conf_frac = confidence / 100.0

    # Already confident — skip
    if conf_frac >= QUICK_THRESHOLD:
        return False

    # Check for enrichable identity data
    has_username = bool(
        profile_data.get("username")
        or profile_data.get("name")
        or profile_data.get("profile_url")
    )

    # Very low confidence with no enrichable data: Stage 2 won't help
    if conf_frac < LOW_CONFIDENCE_THRESHOLD and not has_username:
        return False

    # Sweet spot: uncertain AND we can enrich
    if LOW_CONFIDENCE_THRESHOLD <= conf_frac < QUICK_THRESHOLD:
        return True

    # Borderline low: only escalate if we have a username
    return has_username


def _calculate_feature_completeness(features: dict) -> float:
    """Calculate what fraction of present features have non-zero/non-None values."""
    try:
        from features.missing_value_handler import calculate_feature_completeness
        return calculate_feature_completeness(features, "")
    except Exception:
        pass
    if not features:
        return 0.0
    total = len(features)
    present = sum(
        1 for v in features.values()
        if v is not None and v != "" and not (isinstance(v, float) and v != v)
        and v != 0
    )
    return present / max(total, 1)


def merge_stage_predictions(
    stage1_label: str,
    stage1_confidence: float,
    stage1_features: dict,
    stage2_label: str,
    stage2_confidence: float,
    stage2_features: dict,
    platform: str = "",
) -> tuple:
    """
    Merge Stage 1 and Stage 2 predictions by feature completeness weighting.

    If Stage 2 has poor data coverage (< 70% of Stage 1), trust Stage 1 more.
    Prevents Stage 2 from degrading confidence when enrichment data is sparse.

    Returns:
        (merged_label, merged_confidence, source)
        source: 'stage1_fallback', 'stage2_dominant', 'weighted_merge'
    """
    s1_completeness = _calculate_feature_completeness(stage1_features)
    s2_completeness = _calculate_feature_completeness(stage2_features)

    logger.debug(
        "Merge stages: s1_completeness=%.2f s2_completeness=%.2f "
        "s1_conf=%.1f%% s2_conf=%.1f%%",
        s1_completeness, s2_completeness, stage1_confidence, stage2_confidence,
    )

    # If Stage 2 significantly degraded confidence, fall back to Stage 1
    confidence_drop = stage1_confidence - stage2_confidence
    if confidence_drop > MAX_CONFIDENCE_DEGRADATION * 100:
        logger.warning(
            "[%s] Stage 2 reduced confidence by %.1f pp (s1=%.1f%% → s2=%.1f%%). "
            "Falling back to Stage 1 — likely due to missing enrichment data.",
            platform, confidence_drop, stage1_confidence, stage2_confidence,
        )
        return stage1_label, stage1_confidence, "stage1_fallback"

    # If Stage 2 data coverage is much worse, down-weight it
    if s2_completeness < s1_completeness * 0.70:
        logger.info(
            "[%s] Stage 2 feature completeness (%.2f) << Stage 1 (%.2f). "
            "Trusting Stage 1 prediction.",
            platform, s2_completeness, s1_completeness,
        )
        return stage1_label, stage1_confidence, "stage1_fallback"

    # Stage 2 is more complete: use it with optional confidence weighting
    if s2_completeness >= s1_completeness:
        return stage2_label, stage2_confidence, "stage2_dominant"

    # Weighted blend of confidences
    total = s1_completeness + s2_completeness
    w1 = s1_completeness / total
    w2 = s2_completeness / total
    merged_confidence = stage1_confidence * w1 + stage2_confidence * w2

    # Pick label from stage with more features
    label = stage2_label if s2_completeness > s1_completeness else stage1_label
    return label, round(merged_confidence, 1), "weighted_merge"


def _merge_enrichment_into_df(df: pd.DataFrame, enrichment: dict) -> pd.DataFrame:
    """
    Merge enriched features dict into a single-row DataFrame.
    New feature columns are added; existing columns are overwritten only if they
    were empty/zero in the original.
    High-value signals (account_age_days, public_gists, star_received_total, etc.)
    from live enrichment always override default-zero values.
    """
    df = df.copy()
    features = enrichment.get("features", {})
    raw_data = enrichment.get("raw_data", {})

    # These fields from live data should always override user-entered zeros
    _high_priority_fields = {
        "account_age_days", "public_gists", "star_received_total", "fork_ratio",
        "readme_presence_ratio", "unique_languages", "repo_diversity_score",
        "activity_recency", "commit_regularity", "repos_per_day",
        "followers", "following", "public_repos",
    }

    for key, value in features.items():
        if value is None:
            continue
        try:
            numeric_val = float(value)
            # Override if: column missing, value is 0 in original, or it's a high-priority field
            should_override = (
                key not in df.columns
                or float(df[key].iloc[0]) == 0
                or key in _high_priority_fields
            )
            if should_override and not (key in _high_priority_fields and numeric_val == 0):
                df[key] = numeric_val
        except (TypeError, ValueError):
            if isinstance(value, str) and key not in df.columns:
                df[key] = value

    # Also pull from raw_data for key fields not in features
    for key in ("account_age_days", "public_gists"):
        if key not in features and key in raw_data:
            try:
                df[key] = float(raw_data[key])
            except (TypeError, ValueError):
                pass

    return df


def _build_progress_steps(stage: int) -> List[dict]:
    """Return progress step descriptors for the UI."""
    steps = [
        {"label": "Quick scan inconclusive — running deep analysis...", "done": False},
        {"label": "Fetching live profile data...", "done": False},
        {"label": "Analyzing profile photo...", "done": False},
        {"label": "Checking cross-platform identity...", "done": False},
        {"label": "Computing behavioral fingerprint...", "done": False},
        {"label": "Running enhanced prediction...", "done": False},
    ]
    for i in range(min(stage + 1, len(steps))):
        steps[i]["done"] = True
    return steps


def predict_single_profile(
    profile_data: Dict[str, Any],
    platform: str,
    model_path: Path,
    feature_builder: Callable,
    push_progress: Optional[Callable] = None,
    task_id: Optional[str] = None,
    enable_deep_analysis: bool = True,
) -> Dict:
    """
    Tiered single-profile prediction.

    Stage 1: Quick prediction using available form data.
    Stage 2 (if confidence 0.40–0.80): Deep enrichment + re-predict.

    Args:
        profile_data: Dict of profile fields from form or API.
        platform: Platform name (e.g., "github").
        model_path: Path to the saved model .pkl file.
        feature_builder: Platform-specific feature builder function.
        push_progress: Optional SSE progress push function(task_id, pct, msg).
        task_id: SSE task ID for streaming progress.
        enable_deep_analysis: Set False to force quick mode only.

    Returns:
        {
            label, confidence, confidence_label, analysis_depth,
            features_used, shap_chart, warnings,
            enrichment_summary (dict),
            progress_steps (list),
            cross_platform (dict),
            username_features (dict),
            bio_features (dict),
            photo_features (dict),
            behavioral_features (dict),
        }
    """
    def _push(pct: int, msg: str, **kwargs):
        if push_progress and task_id:
            push_progress(task_id, pct, msg, **kwargs)

    t0 = time.time()
    charts_dir = Path(__file__).parent.parent / "static" / "charts"
    import os
    tag = os.urandom(6).hex()

    # ---- Stage 1: Quick prediction ----
    df = pd.DataFrame([profile_data])
    _push(10, "Running quick scan...")

    try:
        label, confidence = _run_model(df, model_path, feature_builder)
    except Exception as exc:
        logger.warning("Stage 1 prediction failed for [%s]: %s", platform, exc)
        return {
            "label": "Unknown",
            "confidence": 50.0,
            "confidence_label": "Uncertain",
            "analysis_depth": "quick",
            "features_used": [],
            "shap_chart": "",
            "warnings": [str(exc)],
            "enrichment_summary": {},
            "progress_steps": _build_progress_steps(0),
            "error": str(exc),
        }

    conf_frac = confidence / 100.0

    # If high confidence OR deep analysis disabled → return quick result
    if not should_escalate_to_stage2(confidence, profile_data, enable_deep_analysis):
        _push(90, "High-confidence result. Generating SHAP explanation...")
        try:
            label, confidence, shap_chart, warnings = _run_model_with_shap(
                df, model_path, feature_builder, tag, charts_dir
            )
        except Exception:
            shap_chart, warnings = "", []

        # Apply signal-based confidence adjustment for quick scan too
        try:
            from features.confidence_explainer import (
                adjust_confidence_for_display,
                generate_confidence_explanation,
            )
            confidence = adjust_confidence_for_display(confidence, label, dict(df.iloc[0]))
            explanation = generate_confidence_explanation(
                label, confidence, dict(df.iloc[0]), platform=platform
            )
        except Exception:
            explanation = {}

        _push(100, "Complete.", done=True)
        return {
            "label": label,
            "confidence": confidence,
            "confidence_label": confidence_label(confidence),
            "analysis_depth": "quick",
            "features_used": list(df.columns),
            "shap_chart": shap_chart,
            "warnings": warnings,
            "enrichment_summary": {"stage": 1, "reason": "High confidence on quick scan"},
            "progress_steps": _build_progress_steps(0),
            "confidence_explanation": explanation,
        }

    # ---- Stage 2: Deep analysis ----
    _push(15, f"Quick scan inconclusive ({confidence:.0f}% confidence). Running deep analysis...")
    logger.info("[%s] Stage 2 triggered (quick confidence=%.1f%%)", platform, confidence)

    enrichment_summary: Dict = {"stage": 2, "quick_confidence": confidence}
    extra_features: Dict = {}
    username = profile_data.get("username") or profile_data.get("name") or ""
    avatar_url = profile_data.get("avatar_url", "")
    bio_text = profile_data.get("bio") or profile_data.get("about") or profile_data.get("headline") or ""

    # 2a: Live API enrichment
    _push(25, "Fetching live profile data...")
    enrichment = {}
    if username:
        try:
            from live_enrichment import enrich_profile
            enrichment = enrich_profile(platform, username, timeout_seconds=10.0)
            extra_features.update(enrichment.get("features", {}))
            enrichment_summary["live_completeness"] = enrichment.get("data_completeness_score", 0)
            # Use avatar URL from enrichment if not already provided
            if not avatar_url and enrichment.get("raw_data", {}).get("avatar_url"):
                avatar_url = enrichment["raw_data"]["avatar_url"]
            if not bio_text and enrichment.get("features", {}).get("bio"):
                bio_text = enrichment["features"]["bio"]
        except Exception as exc:
            logger.debug("Stage 2 live enrichment failed: %s", exc)
            enrichment_summary["live_error"] = str(exc)

    # 2b: Photo analysis
    _push(40, "Analyzing profile photo...")
    photo_features: Dict = {}
    if avatar_url:
        try:
            from features.photo_analysis import analyze_avatar
            photo_features = analyze_avatar(avatar_url, platform=platform,
                                            profile_identifier=username)
            extra_features.update({k: v for k, v in photo_features.items()
                                    if v is not None and isinstance(v, (int, float))})
            enrichment_summary["photo_analyzed"] = True
        except Exception as exc:
            logger.debug("Photo analysis failed: %s", exc)
            enrichment_summary["photo_error"] = str(exc)

    # 2c: Username forensics
    username_features: Dict = {}
    if username:
        try:
            from features.username_forensics import analyze_username
            username_features = analyze_username(username)
            extra_features.update({k: v for k, v in username_features.items()
                                    if v is not None and isinstance(v, (int, float))})
        except Exception as exc:
            logger.debug("Username forensics failed: %s", exc)

    # 2d: Bio forensics
    bio_features: Dict = {}
    if bio_text:
        try:
            from features.bio_forensics import analyze_bio
            bio_features = analyze_bio(bio_text, platform=platform)
            extra_features.update({k: v for k, v in bio_features.items()
                                    if v is not None and isinstance(v, (int, float))})
        except Exception as exc:
            logger.debug("Bio forensics failed: %s", exc)

    # 2e: Behavioral fingerprint
    _push(55, "Computing behavioral fingerprint...")
    behavioral_features: Dict = {}
    posts = enrichment.get("posts") or enrichment.get("comments") or []
    timestamps = enrichment.get("timestamps", [])
    if posts or timestamps:
        try:
            from features.behavioral import compute_behavioral_fingerprint
            behavioral_features = compute_behavioral_fingerprint(posts, timestamps, platform)
            extra_features.update({k: v for k, v in behavioral_features.items()
                                    if v is not None and isinstance(v, (int, float))})
            enrichment_summary["behavioral_posts"] = behavioral_features.get("behav_post_count", 0)
        except Exception as exc:
            logger.debug("Behavioral fingerprint failed: %s", exc)

    # 2f: Cross-platform check
    _push(70, "Checking cross-platform identity...")
    cross_platform_result: Dict = {}
    if username:
        try:
            from features.cross_platform import cross_platform_check
            cross_platform_result = cross_platform_check(username, platform, bio_text)
            enrichment_summary["platforms_found"] = cross_platform_result.get("platforms_found_count", 0)
            # Add numeric cross-platform features
            for k in ("cross_platform_presence_ratio", "bio_similarity_score",
                      "follower_consistency_score", "creation_date_consistency",
                      "username_similarity_score", "overall_consistency_score"):
                v = cross_platform_result.get(k)
                if v is not None:
                    extra_features[f"xp_{k}"] = v
        except Exception as exc:
            logger.debug("Cross-platform check failed: %s", exc)

    # Build enriched DataFrame
    df_enriched = df.copy()
    for k, v in extra_features.items():
        try:
            df_enriched[k] = float(v)
        except (TypeError, ValueError):
            pass

    # 2g: Re-predict with enriched features
    _push(85, "Running enhanced prediction...")
    stage1_label = label
    stage1_confidence = confidence
    try:
        label2, confidence2, shap_chart, warnings = _run_model_with_shap(
            df_enriched, model_path, feature_builder, tag, charts_dir
        )
        # Apply confidence adjustment based on agreeing signals
        try:
            from features.confidence_explainer import adjust_confidence_for_display
            all_features = {**profile_data, **extra_features}
            confidence2 = adjust_confidence_for_display(confidence2, label2, all_features)
        except Exception:
            pass

        # Smart merge: prefer Stage 2 only if it has better data coverage
        merged_label, merged_confidence, merge_source = merge_stage_predictions(
            stage1_label=label,
            stage1_confidence=confidence,
            stage1_features=dict(df.iloc[0]),
            stage2_label=label2,
            stage2_confidence=confidence2,
            stage2_features=dict(df_enriched.iloc[0]),
            platform=platform,
        )
        label, confidence = merged_label, merged_confidence
        enrichment_summary["merge_source"] = merge_source
        shap_chart_final = shap_chart
    except Exception as exc:
        logger.warning("Stage 2 re-prediction failed: %s", exc)
        shap_chart_final, warnings = "", [str(exc)]

    elapsed = time.time() - t0
    enrichment_summary["total_time_s"] = round(elapsed, 2)
    enrichment_summary["extra_features_added"] = len(extra_features)

    # Generate human-readable confidence explanation
    all_features = {**profile_data, **extra_features}
    explanation: Dict = {}
    try:
        from features.confidence_explainer import (
            generate_confidence_explanation,
            adjust_confidence_for_display,
        )
        confidence = adjust_confidence_for_display(confidence, label, all_features)
        explanation = generate_confidence_explanation(
            label, confidence, all_features,
            enrichment_summary=enrichment_summary,
            platform=platform,
        )
    except Exception as exc:
        logger.debug("Confidence explanation failed: %s", exc)

    # Count missing features for logging
    count_missing = sum(
        1 for v in all_features.values()
        if v is None or v == "" or (isinstance(v, float) and v != v)
    )
    total_features = len(df_enriched.columns)
    completeness_ratio = 1.0 - (count_missing / max(total_features, 1))

    _push(100, f"Deep analysis complete in {elapsed:.1f}s.", done=True)
    logger.info(
        "[%s] Tiered prediction complete: depth=deep label=%s conf=%.1f%% "
        "time=%.1fs features_used=%d features_missing=%d completeness=%.2f",
        platform, label, confidence, elapsed, total_features, count_missing, completeness_ratio,
    )

    return {
        "label": label,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "analysis_depth": "deep",
        "features_used": list(df_enriched.columns),
        "shap_chart": shap_chart_final,
        "warnings": warnings,
        "enrichment_summary": enrichment_summary,
        "progress_steps": _build_progress_steps(5),
        "cross_platform": cross_platform_result,
        "username_features": username_features,
        "bio_features": bio_features,
        "photo_features": photo_features,
        "behavioral_features": behavioral_features,
        "extra_features_count": len(extra_features),
        "confidence_explanation": explanation,
        "feature_completeness": round(completeness_ratio, 2),
    }
