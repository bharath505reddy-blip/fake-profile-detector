"""
features/missing_value_handler.py — Platform-aware missing feature handling.

Replaces silent zero-filling with:
  1. Platform-median defaults for legit profiles (not zeros which look like fakes)
  2. Binary _missing flags so the model can learn "data absent" ≠ "low value"

Usage:
    from features.missing_value_handler import handle_missing_features, load_feature_defaults
    filled = handle_missing_features(features_dict, platform)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULTS_FILE = Path(__file__).parent.parent / "config" / "feature_defaults.json"

# ---------------------------------------------------------------------------
# Hardcoded fallback defaults (median of legit users per platform)
# These are used when the computed defaults file doesn't exist yet.
# Values are empirically reasonable medians for organic/legitimate users.
# ---------------------------------------------------------------------------
_HARDCODED_DEFAULTS: Dict[str, Dict[str, float]] = {
    "instagram": {
        "followers": 150,
        "following": 300,
        "posts": 15,
        "is_verified": 0,
        "bio_length": 20,
        "ff_ratio": 0.5,
        "posts_per_100_followers": 10.0,
        "uname_len": 10,
        "uname_digit_count": 1,
        "bio_len": 20,
        "bio_has_url": 0,
        "bio_has_crypto": 0,
        "bio_has_at": 0,
        "bio_digit_count": 0,
    },
    "facebook": {
        "friends": 350,
        "followers": 200,
        "posts": 40,
        "is_verified": 0,
        "ff_ratio": 1.0,
    },
    "x": {
        "followers": 200,
        "following": 350,
        "tweets": 300,
        "is_verified": 0,
        "ff_ratio": 0.6,
    },
    "linkedin": {
        "connections": 180,
        "followers": 200,
        "headline_len": 50,
        "about_len": 80,
    },
    "github": {
        "followers": 15,
        "following": 30,
        "public_repos": 10,
        "public_gists": 1,
        "account_age_days": 730,
        "bio_len": 30,
        "ff_ratio": 0.5,
    },
    "discord": {
        "account_age_days": 400,
        "has_avatar": 1,
        "suspicion_score": 0.1,
    },
    "youtube": {
        "subscribers": 1500,
        "videos": 30,
        "about_len": 60,
    },
    "tiktok": {
        "followers": 500,
        "following": 300,
        "videos": 20,
        "bio_len": 20,
    },
    "reddit": {
        "karma": 2500,
        "account_age_days": 800,
        "about_len": 30,
    },
    "snapchat": {
        "score": 15000,
        "bio_len": 15,
    },
}

# Features that should ALWAYS get a missing flag (never silently zero-fill)
_IMPORTANT_FEATURES = {
    "followers", "following", "posts", "tweets", "videos", "karma",
    "account_age_days", "public_repos", "public_gists", "connections",
    "subscribers", "score", "friends", "bio", "about", "headline",
    "has_avatar", "is_verified",
}

# Cache for loaded defaults
_loaded_defaults: Optional[Dict] = None


def load_feature_defaults() -> Dict[str, Dict[str, float]]:
    """
    Load computed feature defaults from config/feature_defaults.json.
    Falls back to hardcoded defaults if file doesn't exist.
    """
    global _loaded_defaults
    if _loaded_defaults is not None:
        return _loaded_defaults

    if _DEFAULTS_FILE.exists():
        try:
            data = json.loads(_DEFAULTS_FILE.read_text())
            _loaded_defaults = data
            logger.debug("Loaded feature defaults from %s", _DEFAULTS_FILE)
            return _loaded_defaults
        except Exception as exc:
            logger.warning("Could not load feature_defaults.json: %s — using hardcoded fallback", exc)

    _loaded_defaults = _HARDCODED_DEFAULTS
    return _loaded_defaults


def get_platform_default(platform: str, feature: str) -> float:
    """Return the default value for a specific platform feature."""
    defaults = load_feature_defaults()
    platform_defaults = defaults.get(platform.lower(), {})
    # Try exact match, then try without prefix
    if feature in platform_defaults:
        return float(platform_defaults[feature])
    # Global fallback: 0 for most features, with some sensible overrides
    return 0.0


def handle_missing_features(
    features_dict: Dict,
    platform: str,
    add_missing_flags: bool = True,
) -> Dict:
    """
    Replace None/NaN/empty values with platform-median defaults for legit profiles.
    Optionally adds binary <feature>_missing flags.

    Args:
        features_dict: Dict of feature_name -> value (may contain None/NaN).
        platform: Platform name for looking up medians.
        add_missing_flags: If True, adds {feature}_missing=1 for each filled value.

    Returns:
        New dict with filled values and optional missing flags.
    """
    defaults = load_feature_defaults()
    platform_defaults = defaults.get(platform.lower(), {})
    filled: Dict = dict(features_dict)  # start with a copy
    missing_flags: Dict = {}

    # Process keys that exist in the input (replace missing/None/NaN values)
    for key, value in features_dict.items():
        is_missing = (
            value is None
            or value == ""
            or (isinstance(value, float) and math.isnan(value))
        )

        if is_missing:
            # Use platform-specific median default, fall back to 0
            default_val = platform_defaults.get(key, 0.0)
            filled[key] = default_val
            if add_missing_flags and key in _IMPORTANT_FEATURES:
                missing_flags[f"{key}_missing"] = 1
        else:
            if add_missing_flags and key in _IMPORTANT_FEATURES:
                missing_flags[f"{key}_missing"] = 0

    # Also inject defaults for platform features that are completely absent
    # (not present at all in the input dict)
    for key, default_val in platform_defaults.items():
        if key not in filled:
            filled[key] = default_val
            if add_missing_flags and key in _IMPORTANT_FEATURES:
                missing_flags[f"{key}_missing"] = 1

    return {**filled, **missing_flags}


def calculate_feature_completeness(features_dict: Dict, platform: str) -> float:
    """
    Compute what fraction of important platform features have non-missing values.
    Returns 0.0–1.0.
    """
    platform_fields = set(load_feature_defaults().get(platform.lower(), {}).keys())
    important = _IMPORTANT_FEATURES & platform_fields
    if not important:
        # Fall back to counting non-None values over total
        total = len(features_dict)
        if total == 0:
            return 0.0
        non_missing = sum(
            1 for v in features_dict.values()
            if v is not None and v != "" and not (isinstance(v, float) and math.isnan(v))
        )
        return non_missing / total

    present = 0
    for feat in important:
        val = features_dict.get(feat)
        if val is not None and val != "" and not (isinstance(val, float) and math.isnan(val)):
            present += 1

    return present / len(important)


def compute_and_save_defaults(
    training_dfs: Dict[str, "pd.DataFrame"],
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-platform feature medians from labeled training data (legit only).
    Saves to config/feature_defaults.json.

    Args:
        training_dfs: {platform: DataFrame with 'label' column (0=legit, 1=fake)}
        output_path: Where to save (default: config/feature_defaults.json)

    Returns:
        Computed defaults dict.
    """
    import pandas as pd

    output_path = output_path or _DEFAULTS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    computed: Dict[str, Dict[str, float]] = {}
    for platform, df in training_dfs.items():
        if df is None or len(df) == 0:
            continue
        # Use only legit profiles (label=0)
        label_col = next((c for c in ["label", "is_fake", "fake", "target"] if c in df.columns), None)
        if label_col:
            legit_df = df[df[label_col] == 0]
        else:
            legit_df = df

        if len(legit_df) == 0:
            continue

        num_cols = legit_df.select_dtypes(include=[float, int]).columns
        num_cols = [c for c in num_cols if c not in ("label", "is_fake", "fake", "target")]

        platform_defaults: Dict[str, float] = {}
        for col in num_cols:
            median_val = float(legit_df[col].dropna().median())
            if not math.isnan(median_val):
                platform_defaults[col] = round(median_val, 4)

        computed[platform.lower()] = platform_defaults
        logger.info(
            "Computed %d feature defaults for platform=%s from %d legit profiles",
            len(platform_defaults), platform, len(legit_df),
        )

    # Merge with hardcoded (hardcoded fills gaps for features not in training data)
    merged: Dict[str, Dict[str, float]] = {}
    for platform in set(list(_HARDCODED_DEFAULTS.keys()) + list(computed.keys())):
        base = dict(_HARDCODED_DEFAULTS.get(platform, {}))
        base.update(computed.get(platform, {}))
        merged[platform] = base

    output_path.write_text(json.dumps(merged, indent=2))
    logger.info("Saved feature defaults to %s", output_path)

    # Invalidate cache
    global _loaded_defaults
    _loaded_defaults = merged

    return merged
