"""
features/__init__.py — Orchestrator for all feature extraction modules.

Provides a unified extract_all_features() function that combines:
  - Base platform features (from ML/features_<platform>.py)
  - Temporal/behavioral features (temporal.py)
  - Content/linguistic features (content.py)
  - Network/graph features (network.py)
  - Consistency/coherence features (consistency.py)

Each module fails gracefully — if a module errors, its features are omitted
(filled with 0.0 defaults) and the pipeline continues with remaining features.

Usage:
    from features import extract_all_features
    features_dict = extract_all_features(profile_data, platform="instagram")
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Lazy imports of feature modules (wrapped for graceful failure)
try:
    from features.temporal import extract_temporal_features as _extract_temporal
    TEMPORAL_AVAILABLE = True
except Exception as _e:
    TEMPORAL_AVAILABLE = False
    logger.warning("Temporal features unavailable: %s", _e)

try:
    from features.content import extract_content_features as _extract_content
    CONTENT_AVAILABLE = True
except Exception as _e:
    CONTENT_AVAILABLE = False
    logger.warning("Content features unavailable: %s", _e)

try:
    from features.network import extract_network_features as _extract_network
    NETWORK_AVAILABLE = True
except Exception as _e:
    NETWORK_AVAILABLE = False
    logger.warning("Network features unavailable: %s", _e)

try:
    from features.consistency import extract_consistency_features as _extract_consistency
    CONSISTENCY_AVAILABLE = True
except Exception as _e:
    CONSISTENCY_AVAILABLE = False
    logger.warning("Consistency features unavailable: %s", _e)

try:
    from features.username_forensics import analyze_username as _analyze_username
    USERNAME_FORENSICS_AVAILABLE = True
except Exception as _e:
    USERNAME_FORENSICS_AVAILABLE = False
    logger.warning("Username forensics unavailable: %s", _e)

try:
    from features.bio_forensics import analyze_bio as _analyze_bio
    BIO_FORENSICS_AVAILABLE = True
except Exception as _e:
    BIO_FORENSICS_AVAILABLE = False
    logger.warning("Bio forensics unavailable: %s", _e)

try:
    from features.photo_analysis import analyze_avatar as _analyze_avatar
    PHOTO_ANALYSIS_AVAILABLE = True
except Exception as _e:
    PHOTO_ANALYSIS_AVAILABLE = False
    logger.warning("Photo analysis unavailable: %s", _e)

try:
    from features.behavioral import compute_behavioral_fingerprint as _compute_behavioral
    BEHAVIORAL_AVAILABLE = True
except Exception as _e:
    BEHAVIORAL_AVAILABLE = False
    logger.warning("Behavioral fingerprinting unavailable: %s", _e)


def extract_temporal_features(profile_data: dict) -> dict:
    """
    Extract temporal and behavioral pattern features.
    Returns empty dict if module unavailable.
    """
    if not TEMPORAL_AVAILABLE:
        return {}
    try:
        return _extract_temporal(profile_data)
    except Exception as exc:
        logger.warning("Temporal feature extraction failed: %s", exc)
        return {}


def extract_content_features(profile_data: dict) -> dict:
    """
    Extract text and linguistic features from post content.
    Returns empty dict if module unavailable.
    """
    if not CONTENT_AVAILABLE:
        return {}
    try:
        return _extract_content(profile_data)
    except Exception as exc:
        logger.warning("Content feature extraction failed: %s", exc)
        return {}


def extract_network_features(
    profile_data: dict,
    existing_predictions: Optional[Dict[str, str]] = None
) -> dict:
    """
    Extract network and graph topology features.
    existing_predictions: optional dict of username -> "Fake"/"Legit" for second-pass enrichment.
    Returns empty dict if module unavailable.
    """
    if not NETWORK_AVAILABLE:
        return {}
    try:
        return _extract_network(profile_data, existing_predictions)
    except Exception as exc:
        logger.warning("Network feature extraction failed: %s", exc)
        return {}


def extract_consistency_features(profile_data: dict) -> dict:
    """
    Extract profile consistency and coherence features.
    Returns empty dict if module unavailable.
    """
    if not CONSISTENCY_AVAILABLE:
        return {}
    try:
        return _extract_consistency(profile_data)
    except Exception as exc:
        logger.warning("Consistency feature extraction failed: %s", exc)
        return {}


def extract_all_features(
    profile_data: dict,
    platform: str = "",
    existing_predictions: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Extract all available features from a profile data dict.

    Combines temporal, content, network, and consistency features into a single dict.
    If any module fails, its features are skipped and the others continue.

    Args:
        profile_data: Dict with profile fields. See individual modules for expected keys.
            Common keys:
              username, bio, posts (list of str), followers, following,
              post_timestamps (list), account_age_days, has_avatar,
              followers_list, following_list, follower_metadata
        platform: Platform name (e.g., "instagram") — used for logging context.
        existing_predictions: Optional dict mapping username -> "Fake"/"Legit"
            for second-pass fake_neighbor_ratio computation.

    Returns:
        Dict of feature_name -> float with all available features.
        Features from unavailable modules are simply absent (not filled with 0).
    """
    all_features: dict = {}

    # 1. Temporal features
    temporal = extract_temporal_features(profile_data)
    if temporal:
        all_features.update(temporal)
        logger.debug("[%s] Extracted %d temporal features", platform, len(temporal))

    # 2. Content features
    content = extract_content_features(profile_data)
    if content:
        all_features.update(content)
        logger.debug("[%s] Extracted %d content features", platform, len(content))

    # 3. Network features
    network = extract_network_features(profile_data, existing_predictions)
    if network:
        all_features.update(network)
        logger.debug("[%s] Extracted %d network features", platform, len(network))

    # 4. Consistency features
    consistency = extract_consistency_features(profile_data)
    if consistency:
        all_features.update(consistency)
        logger.debug("[%s] Extracted %d consistency features", platform, len(consistency))

    logger.debug(
        "[%s] Total extracted features: %d (temporal=%d content=%d network=%d consistency=%d)",
        platform, len(all_features), len(temporal), len(content), len(network), len(consistency)
    )

    return all_features


def analyze_username(username) -> dict:
    """Extract username forensic features. Returns {} if unavailable."""
    if not USERNAME_FORENSICS_AVAILABLE:
        return {}
    try:
        return _analyze_username(username)
    except Exception as exc:
        logger.warning("Username forensics failed: %s", exc)
        return {}


def analyze_bio(bio_text, platform: str = "") -> dict:
    """Extract bio forensic features. Returns {} if unavailable."""
    if not BIO_FORENSICS_AVAILABLE:
        return {}
    try:
        return _analyze_bio(bio_text, platform)
    except Exception as exc:
        logger.warning("Bio forensics failed: %s", exc)
        return {}


def analyze_avatar(image_source, platform: str = "", profile_identifier: str = "") -> dict:
    """Analyze a profile avatar image. Returns {} if unavailable."""
    if not PHOTO_ANALYSIS_AVAILABLE:
        return {}
    try:
        return _analyze_avatar(image_source, platform=platform,
                               profile_identifier=profile_identifier)
    except Exception as exc:
        logger.warning("Photo analysis failed: %s", exc)
        return {}


def compute_behavioral_fingerprint(posts, timestamps, platform: str = "") -> dict:
    """Compute behavioral fingerprint from post history. Returns {} if unavailable."""
    if not BEHAVIORAL_AVAILABLE:
        return {}
    try:
        return _compute_behavioral(posts, timestamps, platform)
    except Exception as exc:
        logger.warning("Behavioral fingerprint failed: %s", exc)
        return {}


__all__ = [
    "extract_all_features",
    "extract_temporal_features",
    "extract_content_features",
    "extract_network_features",
    "extract_consistency_features",
    "analyze_username",
    "analyze_bio",
    "analyze_avatar",
    "compute_behavioral_fingerprint",
    "TEMPORAL_AVAILABLE",
    "CONTENT_AVAILABLE",
    "NETWORK_AVAILABLE",
    "CONSISTENCY_AVAILABLE",
    "USERNAME_FORENSICS_AVAILABLE",
    "BIO_FORENSICS_AVAILABLE",
    "PHOTO_ANALYSIS_AVAILABLE",
    "BEHAVIORAL_AVAILABLE",
]
