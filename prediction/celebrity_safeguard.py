"""
prediction/celebrity_safeguard.py — Rule-based safeguard layer for celebrity misclassification.

Runs AFTER the ML model to catch celebrity/public figure misclassifications
without modifying the model itself. Also catches obvious bots that ML might miss.

All thresholds are configurable via environment variables so they can be
tuned without code changes.
"""
from __future__ import annotations
import json
import logging
import os
import pathlib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable thresholds — override via environment variables
# ---------------------------------------------------------------------------

CELEBRITY_FOLLOWER_THRESHOLD = int(
    os.environ.get("CELEBRITY_FOLLOWER_THRESHOLD", "1000000")
)
CELEBRITY_HIGH_FOLLOWER_THRESHOLD = int(
    os.environ.get("CELEBRITY_HIGH_FOLLOWER_THRESHOLD", "100000")
)
MEGA_FOLLOWER_THRESHOLD = int(
    os.environ.get("MEGA_FOLLOWER_THRESHOLD", "10000000")
)
CELEBRITY_MIN_ACCOUNT_AGE_DAYS = int(
    os.environ.get("CELEBRITY_MIN_ACCOUNT_AGE_DAYS", "730")
)
CELEBRITY_EXTREME_RATIO_AGE_DAYS = int(
    os.environ.get("CELEBRITY_EXTREME_RATIO_AGE_DAYS", "1825")
)
BOT_NEW_ACCOUNT_MAX_DAYS = int(
    os.environ.get("BOT_NEW_ACCOUNT_MAX_DAYS", "7")
)
BOT_MASS_FOLLOWING_MIN = int(
    os.environ.get("BOT_MASS_FOLLOWING_MIN", "1000")
)

_ALLOWLIST_PATH = pathlib.Path(__file__).parent.parent / "data" / "public_figures.json"


def _extract_counts(features: dict) -> tuple:
    """Extract follower/following/posts/age/verified from a features dict."""
    followers = float(features.get("followers", 0) or 0)
    following = float(features.get("following", 0) or 0)
    posts = float(
        features.get("posts", 0)
        or features.get("tweets", 0)
        or features.get("videos", 0)
        or 0
    )
    account_age_days = float(features.get("account_age_days", 0) or 0)
    verified = bool(
        features.get("verified") or features.get("is_verified") or False
    )
    return followers, following, posts, account_age_days, verified


def apply_celebrity_safeguard(
    features: dict,
    ml_prediction: str,
    ml_confidence: float,
) -> tuple:
    """
    Apply rule-based overrides to catch celebrity misclassification.

    Returns:
        (final_prediction, final_confidence, override_applied, override_reason)
    """
    followers, following, posts, account_age_days, verified = _extract_counts(features)
    ratio = followers / max(following, 1)

    # Rule 1: Verified + Very High Followers = Almost Certainly Legit
    # No bot achieves verified status with 1M+ organic followers
    if verified and followers >= CELEBRITY_FOLLOWER_THRESHOLD:
        logger.info(
            "Celebrity safeguard Rule 1: verified=%s followers=%.0f", verified, followers
        )
        return (
            "Legit", 97.0, True,
            "Verified account with 1M+ followers — celebrity/public figure pattern",
        )

    # Rule 2: Verified + High Followers + Mature Account
    if (verified
            and followers >= CELEBRITY_HIGH_FOLLOWER_THRESHOLD
            and account_age_days >= CELEBRITY_MIN_ACCOUNT_AGE_DAYS
            and ml_prediction == "Fake"
            and ml_confidence > 60.0):
        logger.info("Celebrity safeguard Rule 2 fired")
        return (
            "Legit", 92.0, True,
            "Verified account with 100K+ followers and 2+ year history",
        )

    # Rule 3: Mega Follower Count with Posting History
    if followers >= MEGA_FOLLOWER_THRESHOLD and posts >= 100:
        logger.info("Celebrity safeguard Rule 3 fired: %.0f followers", followers)
        return (
            "Legit", 95.0, True,
            "Mega-scale account (10M+ followers) with substantial post history",
        )

    # Rule 4: Extreme Ratio + Mature Account + Posts (unverified celebrities exist)
    if (ratio >= 1_000
            and account_age_days >= CELEBRITY_EXTREME_RATIO_AGE_DAYS
            and posts >= 50
            and ml_prediction == "Fake"):
        logger.info("Celebrity safeguard Rule 4 fired: ratio=%.0f age=%.0f", ratio, account_age_days)
        return (
            "Legit", 88.0, True,
            "Extreme follower ratio on mature account with post history — celebrity pattern",
        )

    # Rule 5: High Followers + Selective Following + Mature Account
    if (followers >= CELEBRITY_FOLLOWER_THRESHOLD
            and following <= 500
            and account_age_days >= 1095
            and ml_prediction == "Fake"):
        logger.info("Celebrity safeguard Rule 5 fired")
        return (
            "Legit", 90.0, True,
            "High follower count with selective following on mature account",
        )

    return ml_prediction, ml_confidence, False, None


def apply_bot_safeguard(
    features: dict,
    ml_prediction: str,
    ml_confidence: float,
) -> tuple:
    """
    Catch obvious bots that the ML model might classify as Legit.

    Returns:
        (final_prediction, final_confidence, override_applied, override_reason)
    """
    followers, following, posts, account_age_days, _ = _extract_counts(features)

    # Rule: Brand new account mass-following with no content
    if (account_age_days <= BOT_NEW_ACCOUNT_MAX_DAYS
            and following >= BOT_MASS_FOLLOWING_MIN
            and followers <= 50
            and posts == 0
            and ml_prediction == "Legit"):
        logger.info("Bot safeguard fired: new account mass-following")
        return (
            "Fake", 95.0, True,
            "Brand new account mass-following with zero posts and minimal followers",
        )

    # Rule: Extremely high following + near-zero followers + no content
    if (following >= 5_000
            and followers <= 100
            and posts == 0
            and ml_prediction == "Legit"):
        logger.info("Bot safeguard fired: mass-following no organic return")
        return (
            "Fake", 93.0, True,
            "Mass-following behavior with no organic following back",
        )

    return ml_prediction, ml_confidence, False, None


def check_public_figure_allowlist(username: str, platform: str) -> tuple:
    """
    Check if username is on the known public figures allowlist.

    Returns:
        (is_known_figure: bool, confidence: float | None)
    """
    if not _ALLOWLIST_PATH.exists():
        return False, None
    try:
        with open(_ALLOWLIST_PATH) as f:
            allowlist = json.load(f)
        platform_list = allowlist.get(platform, [])
        if username.lower() in [u.lower() for u in platform_list]:
            return True, 99.0
    except Exception as exc:
        logger.warning("Failed to read public figures allowlist: %s", exc)
    return False, None


def load_allowlist() -> dict:
    """Load and return the full allowlist dict, or empty dict if missing."""
    if not _ALLOWLIST_PATH.exists():
        return {}
    try:
        with open(_ALLOWLIST_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_allowlist(allowlist: dict) -> None:
    """Persist the allowlist dict to disk."""
    _ALLOWLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_ALLOWLIST_PATH, "w") as f:
        json.dump(allowlist, f, indent=2)
