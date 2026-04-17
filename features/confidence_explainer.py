"""
features/confidence_explainer.py — Human-readable confidence explanations.

Provides:
  - generate_confidence_explanation(): explains WHY the model has a certain confidence
  - adjust_confidence_for_display(): boosts confidence when multiple signals agree
  - calculate_feature_completeness(): data coverage ratio for the UI trust bar
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal definitions per check type
# ---------------------------------------------------------------------------

# (feature_key, legit_check_fn, label, description_if_positive, description_if_negative)
_SIGNAL_CHECKS: List[Tuple[str, Any, str, str, str]] = [
    (
        "account_age_days",
        lambda v: v > 365,
        "Account age",
        "Account is over 1 year old (typical of real users)",
        "Account is very new (< 365 days — common among bots)",
    ),
    (
        "account_age_days",
        lambda v: v > 90,
        "Account age (90d)",
        "Account is older than 90 days",
        "Account created within the last 90 days",
    ),
    (
        "has_avatar",
        lambda v: float(v) == 1,
        "Profile photo",
        "Has a custom profile photo",
        "No custom profile photo (default avatar)",
    ),
    (
        "is_verified",
        lambda v: float(v) == 1,
        "Verification",
        "Account is verified by the platform",
        "Account is not verified",
    ),
    (
        "photo_is_default_avatar",
        lambda v: float(v) == 0,
        "Avatar (non-default)",
        "Avatar is not a platform default image",
        "Avatar matches a known default/placeholder image",
    ),
    (
        "photo_stock_score",
        lambda v: float(v) < 0.2,
        "Avatar uniqueness",
        "Profile photo appears unique (not stock/reused)",
        "Profile photo has been seen on multiple profiles",
    ),
    (
        "photo_ai_generated_score",
        lambda v: float(v) < 0.4,
        "Avatar authenticity",
        "Profile photo does not appear AI-generated",
        "Profile photo may be AI-generated",
    ),
    (
        "uname_bot_pattern_count",
        lambda v: float(v) == 0,
        "Username pattern",
        "Username has no bot-like patterns",
        "Username matches bot-pattern signatures",
    ),
    (
        "uname_dict_coverage",
        lambda v: float(v) > 0.3,
        "Username readability",
        "Username contains recognizable words (human-like)",
        "Username is hard to read — possible auto-generated",
    ),
    (
        "uname_digit_ratio",
        lambda v: float(v) < 0.4,
        "Username digits",
        "Username has few numeric characters",
        "Username is heavy on digits — common bot pattern",
    ),
    (
        "bio_has_spam",
        lambda v: float(v) == 0,
        "Bio content",
        "Bio contains no spam/promotional keywords",
        "Bio contains promotional or spam-like keywords",
    ),
    (
        "bio_has_crypto",
        lambda v: float(v) == 0,
        "Bio (crypto)",
        "Bio contains no crypto/airdrop keywords",
        "Bio mentions crypto or airdrop keywords",
    ),
    (
        "ff_ratio",
        lambda v: 0.05 < float(v) < 20,
        "Follower ratio",
        "Follower/following ratio is in a natural range",
        "Follower/following ratio is extreme (possible bot)",
    ),
    (
        "behav_posting_interval_entropy",
        lambda v: float(v) > 1.5,
        "Posting pattern",
        "Posting intervals are irregular (human-like)",
        "Posting intervals are too regular (bot-like)",
    ),
    (
        "behav_burst_post_count",
        lambda v: float(v) == 0,
        "Burst posting",
        "No burst posting detected",
        "Burst posting detected (multiple posts within 60s)",
    ),
    (
        "behav_lexical_diversity",
        lambda v: float(v) > 0.3,
        "Content diversity",
        "Post content shows good lexical diversity (human writing)",
        "Post content is repetitive (possible bot)",
    ),
    (
        "xp_overall_consistency_score",
        lambda v: float(v) > 0.4,
        "Cross-platform",
        "Identity is consistent across platforms",
        "Identity is inconsistent across platforms",
    ),
    (
        "karma",
        lambda v: float(v) > 500,
        "Reddit karma",
        "Substantial karma indicates genuine activity",
        "Very low karma — limited activity",
    ),
    (
        "public_repos",
        lambda v: float(v) >= 3,
        "GitHub repos",
        "Has multiple public repositories",
        "Very few public repositories",
    ),
    (
        "suspicion_score",
        lambda v: float(v) < 0.3,
        "Suspicion score",
        "Low pre-computed suspicion score",
        "Elevated suspicion score detected",
    ),
]


def generate_confidence_explanation(
    prediction: str,
    confidence: float,
    features: Dict[str, Any],
    enrichment_summary: Optional[Dict] = None,
    platform: str = "",
) -> Dict:
    """
    Generate a structured explanation for why the model has a given confidence level.

    Args:
        prediction: 'Fake' or 'Legit'
        confidence: Confidence percentage (0-100)
        features: Dict of feature_name -> value (may include enrichment features)
        enrichment_summary: Optional Stage-2 enrichment metadata
        platform: Platform name for context

    Returns:
        {
            strong_signals: [(label, description)],   # 3+ supporting signals
            weak_signals: [(label, description)],     # Available but uncertain
            missing_signals: [description],           # Data that wasn't available
            data_coverage_pct: int,                   # 0-100
            improvement_tips: [str],                  # Actionable suggestions
            signal_count_total: int,
            signal_count_agreeing: int,
        }
    """
    is_legit = (prediction.lower() in ("legit", "real", "genuine", "0"))
    strong_signals: List[Tuple[str, str]] = []
    weak_signals: List[Tuple[str, str]] = []
    missing_signals: List[str] = []
    improvement_tips: List[str] = []
    checked = 0
    agreeing = 0

    for feat_key, check_fn, label, pos_desc, neg_desc in _SIGNAL_CHECKS:
        value = features.get(feat_key)

        # Skip if feature completely absent
        missing_flag = features.get(f"{feat_key}_missing", None)
        if value is None or missing_flag == 1:
            missing_signals.append(label)
            continue

        try:
            signal_is_legit = check_fn(value)
        except (TypeError, ValueError):
            missing_signals.append(label)
            continue

        checked += 1
        agrees_with_prediction = (
            (is_legit and signal_is_legit) or (not is_legit and not signal_is_legit)
        )
        if agrees_with_prediction:
            agreeing += 1

        # Confidence of this signal
        if agrees_with_prediction:
            description = pos_desc if is_legit else neg_desc
            strong_signals.append((label, description))
        else:
            description = neg_desc if is_legit else pos_desc
            weak_signals.append((label, description))

    # Deduplicate by label (keep first occurrence)
    seen_labels: set = set()
    deduped_strong = []
    for label, desc in strong_signals:
        if label not in seen_labels:
            deduped_strong.append((label, desc))
            seen_labels.add(label)
    deduped_weak = []
    for label, desc in weak_signals:
        if label not in seen_labels:
            deduped_weak.append((label, desc))
            seen_labels.add(label)

    # Deduplicate missing signals
    missing_deduped = list(dict.fromkeys(missing_signals))

    # Compute data coverage
    total_possible = len(set(fc[2] for fc in _SIGNAL_CHECKS))  # unique signal labels
    available = total_possible - len(missing_deduped)
    data_coverage_pct = int(available / max(total_possible, 1) * 100)

    # Build improvement tips
    enrichment = enrichment_summary or {}
    if not features.get("photo_available"):
        improvement_tips.append("Enable live API lookup to analyze the profile photo")
    if enrichment.get("platforms_found", 0) < 3:
        improvement_tips.append("Run cross-platform check to verify identity consistency")
    if not enrichment.get("behavioral_posts"):
        improvement_tips.append("Enable live lookup to analyze posting behavior patterns")
    if data_coverage_pct < 60:
        improvement_tips.append("Provide more profile details in the form for better accuracy")
    if not enrichment.get("live_completeness"):
        improvement_tips.append("Configure API keys in Admin → API Keys to enable live data fetch")

    return {
        "strong_signals": deduped_strong[:6],
        "weak_signals": deduped_weak[:4],
        "missing_signals": missing_deduped[:5],
        "data_coverage_pct": data_coverage_pct,
        "improvement_tips": improvement_tips[:3],
        "signal_count_total": checked,
        "signal_count_agreeing": agreeing,
        "prediction": prediction,
        "confidence": confidence,
    }


def adjust_confidence_for_display(
    raw_confidence: float,
    prediction: str,
    features: Dict[str, Any],
) -> float:
    """
    Boost confidence when multiple independent signals agree.
    Never adjusts by more than 12 percentage points.
    Never exceeds 95%.

    Args:
        raw_confidence: Raw model confidence (0-100).
        prediction: 'Fake' or 'Legit'
        features: Feature dict.

    Returns:
        Adjusted confidence (0-100).
    """
    is_legit = prediction.lower() in ("legit", "real", "genuine")

    agreeing = 0
    total = 0

    # Use a simplified set of high-signal features
    signal_checks = [
        ("account_age_days", lambda v: float(v) > 365),
        ("has_avatar", lambda v: float(v) == 1),
        ("photo_is_default_avatar", lambda v: float(v) == 0),
        ("is_verified", lambda v: float(v) == 1),
        ("uname_bot_pattern_count", lambda v: float(v) == 0),
        ("behav_posting_interval_entropy", lambda v: float(v) > 1.5),
        ("ff_ratio", lambda v: 0.05 < float(v) < 20),
        ("bio_has_spam", lambda v: float(v) == 0),
        ("karma", lambda v: float(v) > 200),
        ("public_repos", lambda v: float(v) >= 2),
    ]

    for feat_key, check_fn in signal_checks:
        value = features.get(feat_key)
        if value is None or features.get(f"{feat_key}_missing") == 1:
            continue
        try:
            signal_is_legit = check_fn(value)
            total += 1
            if (is_legit and signal_is_legit) or (not is_legit and not signal_is_legit):
                agreeing += 1
        except (TypeError, ValueError):
            continue

    if total == 0:
        return raw_confidence

    agreement_ratio = agreeing / total

    # Boost toward a higher confidence when signals agree strongly
    if agreement_ratio >= 0.80:
        target = 85.0 if is_legit else 88.0
        boost = (target - raw_confidence) * 0.45
    elif agreement_ratio >= 0.65:
        target = 75.0 if is_legit else 78.0
        boost = (target - raw_confidence) * 0.25
    else:
        boost = 0.0

    # Cap boost at 12 points
    boost = max(-5.0, min(boost, 12.0))
    adjusted = raw_confidence + boost

    return round(min(adjusted, 95.0), 1)


def compute_prediction_stability(
    predict_fn,
    features: Dict[str, Any],
    n_trials: int = 3,
    noise_sigma: float = 0.03,
) -> Tuple[float, bool]:
    """
    Estimate prediction stability by running with tiny feature perturbations.

    Returns:
        (stability_score 0-1, is_stable bool)
    """
    import numpy as np

    results = []
    for _ in range(n_trials):
        perturbed = {}
        for k, v in features.items():
            if isinstance(v, (int, float)) and v != 0:
                noise = np.random.normal(0, abs(v) * noise_sigma)
                perturbed[k] = v + noise
            else:
                perturbed[k] = v
        try:
            result = predict_fn(perturbed)
            results.append(result.get("label", "Unknown"))
        except Exception:
            pass

    if not results:
        return 0.5, False

    # Agreement rate
    most_common = max(set(results), key=results.count)
    agreement = results.count(most_common) / len(results)
    is_stable = agreement >= 0.67  # At least 2 of 3 agree
    return round(agreement, 2), is_stable
