"""
features/celebrity_detector.py — Celebrity/public figure feature extractor.

Computes features that distinguish celebrities from bots, even when both
have extreme follower/following ratios. Integrated into platform feature builders.
"""
from __future__ import annotations
import math
import os

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

CELEBRITY_THRESHOLDS = {
    "celebrity_scale_followers":     int(os.environ.get("CELEB_SCALE_FOLLOWERS",      "500000")),
    "verified_celebrity_followers":  int(os.environ.get("CELEB_VERIFIED_FOLLOWERS",   "100000")),
    "mega_account_followers":        int(os.environ.get("CELEB_MEGA_FOLLOWERS",       "10000000")),
    "celebrity_following_cap":       int(os.environ.get("CELEB_FOLLOWING_CAP",        "1000")),
    "celebrity_min_age_days":        int(os.environ.get("CELEB_MIN_AGE_DAYS",         "1825")),
    "maturity_high_age":             int(os.environ.get("CELEB_MATURITY_HIGH_AGE",    "1825")),
    "maturity_high_followers":       int(os.environ.get("CELEB_MATURITY_HIGH_FOLLOW", "100000")),
    "maturity_mid_age":              int(os.environ.get("CELEB_MATURITY_MID_AGE",     "730")),
    "maturity_mid_followers":        int(os.environ.get("CELEB_MATURITY_MID_FOLLOW",  "10000")),
    "maturity_low_age":              int(os.environ.get("CELEB_MATURITY_LOW_AGE",     "365")),
    "maturity_low_followers":        int(os.environ.get("CELEB_MATURITY_LOW_FOLLOW",  "1000")),
}


def compute_celebrity_features(profile_data: dict) -> dict:
    """
    Extract features that distinguish celebrities from bots,
    even when both have extreme follower/following ratios.

    Handles missing values gracefully — returns 0 for unavailable fields.
    All output keys are prefixed with 'celeb_' to avoid column collisions.
    """
    T = CELEBRITY_THRESHOLDS

    followers = float(profile_data.get("followers", 0) or 0)
    following = float(profile_data.get("following", 0) or 0)
    posts = float(
        profile_data.get("posts", 0)
        or profile_data.get("tweets", 0)
        or profile_data.get("videos", 0)
        or 0
    )
    account_age_days = float(profile_data.get("account_age_days", 0) or 0)
    is_verified = int(
        bool(profile_data.get("verified") or profile_data.get("is_verified") or 0)
    )

    features: dict = {}

    # 1. log10(followers) — log scale prevents extreme values from dominating
    features["celeb_log_followers"] = math.log10(max(followers, 1))

    # 2. 7-bucket follower tier (0=micro … 6=mega-celebrity)
    if followers < 100:
        features["celeb_followers_tier"] = 0
    elif followers < 1_000:
        features["celeb_followers_tier"] = 1
    elif followers < 10_000:
        features["celeb_followers_tier"] = 2
    elif followers < 100_000:
        features["celeb_followers_tier"] = 3
    elif followers < 1_000_000:
        features["celeb_followers_tier"] = 4
    elif followers < 10_000_000:
        features["celeb_followers_tier"] = 5
    else:
        features["celeb_followers_tier"] = 6

    # 3. Hard celebrity-scale flag — bots never organically reach 500K+ followers
    features["celeb_is_celebrity_scale"] = (
        1 if followers >= T["celebrity_scale_followers"] else 0
    )

    # 4. posts_per_follower — celebrities have very low posts/follower ratio
    features["celeb_posts_per_follower"] = posts / max(followers, 1)

    # 5. follower_per_day — real accounts accumulate followers gradually
    if account_age_days > 0:
        features["celeb_follower_per_day"] = followers / account_age_days
    else:
        features["celeb_follower_per_day"] = 0.0

    # 6. posts_per_year — celebrities post consistently over years; bots burst briefly
    if account_age_days > 0:
        features["celeb_posts_per_year"] = (posts / account_age_days) * 365
    else:
        features["celeb_posts_per_year"] = 0.0

    # 7. Following caps — celebrities follow very few people; bots mass-follow
    features["celeb_following_capped_1000"] = 1 if following <= 1_000 else 0
    features["celeb_following_capped_5000"] = 1 if following <= 5_000 else 0

    # 8. Normalized ratio — elite ratio scaled by log(followers) to account for tier
    ratio = followers / max(following, 1)
    log_followers = math.log10(max(followers, 1))
    features["celeb_normalized_ratio"] = ratio / max(log_followers, 1)

    # 9. Verified + scale combination signals
    features["celeb_verified_celebrity"] = (
        1 if (is_verified and followers >= T["verified_celebrity_followers"]) else 0
    )
    features["celeb_verified_micro"] = (
        1 if (is_verified and followers < T["verified_celebrity_followers"]) else 0
    )

    # 10. Account maturity score — old account + high followers = legitimate
    if account_age_days > 0:
        if (account_age_days >= T["maturity_high_age"]
                and followers >= T["maturity_high_followers"]):
            features["celeb_account_maturity_score"] = 1.0
        elif (account_age_days >= T["maturity_mid_age"]
              and followers >= T["maturity_mid_followers"]):
            features["celeb_account_maturity_score"] = 0.7
        elif (account_age_days >= T["maturity_low_age"]
              and followers >= T["maturity_low_followers"]):
            features["celeb_account_maturity_score"] = 0.4
        else:
            features["celeb_account_maturity_score"] = 0.1
    else:
        features["celeb_account_maturity_score"] = 0.1

    return features
