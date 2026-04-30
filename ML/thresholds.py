# ML/thresholds.py
"""
Per-platform confidence thresholds for batch fake-profile classification.

When the model predicts "Fake" but its confidence is below the platform
threshold, the prediction is downgraded to "Legit" to reduce false positives
on borderline profiles.

Only applied in batch CSV analysis (predict_with_saved_model).
NOT applied in the tiered single-profile predictor.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Minimum confidence (%) required to keep a "Fake" prediction.
# Below this → downgraded to "Legit" (model isn't confident enough).
FAKE_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "instagram": 60.0,
    "facebook":  60.0,
    "x":         60.0,
    "linkedin":  65.0,
    "github":    62.0,
    "youtube":   60.0,
    "reddit":    58.0,
    "discord":   65.0,
    "tiktok":    60.0,
    "snapchat":  55.0,
    "default":   60.0,
}


def apply_confidence_threshold(
    prediction: str, confidence: float, platform: str
) -> tuple[str, float, str]:
    """
    Apply per-platform confidence threshold to a single prediction.

    If the model says "Fake" but confidence is below the platform threshold,
    downgrade to "Legit" to avoid false positives on borderline profiles.

    Returns:
        (new_prediction, confidence, reason)
        reason is "below_threshold" if downgraded, "model" otherwise.
    """
    threshold = FAKE_CONFIDENCE_THRESHOLDS.get(
        platform.lower(),
        FAKE_CONFIDENCE_THRESHOLDS["default"],
    )

    if prediction.lower() == "fake" and confidence < threshold:
        logger.debug(
            "[%s] Downgraded Fake→Legit: confidence=%.1f%% < threshold=%.1f%%",
            platform, confidence, threshold,
        )
        return "Legit", confidence, "below_threshold"

    return prediction, confidence, "model"
