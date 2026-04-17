"""
Instagram feature builder.

Supports two input formats:
  1. Raw profile data (username, followers, following, posts, bio, is_verified, …)
  2. Pre-computed dataset format (profile pic, nums/length username, #followers, …)
     — e.g. the public Instagram Fake Profile dataset from Kaggle/GitHub.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


# Columns present in the public pre-computed dataset
_PRECOMPUTED_COLS = {
    "profile pic", "nums/length username", "fullname words",
    "nums/length fullname", "name==username", "description length",
    "external url", "#posts", "#followers", "#follows",
}


def _is_precomputed_format(df: pd.DataFrame) -> bool:
    """Detect whether the dataframe uses the pre-computed feature format."""
    return len(_PRECOMPUTED_COLS & set(df.columns)) >= 4


def _build_from_precomputed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix directly from pre-computed columns.
    Handles the public Instagram Fake Profile dataset format.
    """
    X = pd.DataFrame()

    # Core counts (directly available)
    X["followers"] = safe_num(df, "#followers", 0)
    X["following"] = safe_num(df, "#follows",   0)
    X["posts"]     = safe_num(df, "#posts",     0)

    # Ratios
    X["ff_ratio"] = X["followers"] / X["following"].replace({0: 1})
    X["posts_per_100_followers"] = X["posts"] / X["followers"].replace({0: 1}) * 100

    # Log-scale — critical for celebrity detection
    X["log_followers"] = np.log1p(X["followers"])
    X["log_following"] = np.log1p(X["following"])
    X["log_posts"]     = np.log1p(X["posts"])

    # Follower tier
    X["followers_tier"] = pd.cut(
        X["followers"],
        bins=[-1, 999, 99_999, 999_999, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Pre-computed username signals
    X["uname_digit_ratio"]  = safe_num(df, "nums/length username", 0)
    X["fullname_word_count"] = safe_num(df, "fullname words",      0)
    X["fullname_digit_ratio"] = safe_num(df, "nums/length fullname", 0)
    X["name_eq_username"]   = safe_num(df, "name==username",       0)

    # Bio / description
    X["bio_len"]      = safe_num(df, "description length", 0)
    X["bio_has_url"]  = safe_num(df, "external url",       0)
    X["bio_is_empty"] = (X["bio_len"] == 0).astype(int)

    # Profile completeness
    X["has_profile_pic"] = safe_num(df, "profile pic", 1)
    X["is_private"]      = safe_num(df, "private",     0)

    # Verification not available in this format — default 0
    X["is_verified"] = safe_num(df, "is_verified", 0)
    X["is_celebrity"] = (
        (X["is_verified"] == 1) & (X["followers"] > 100_000)
    ).astype(int)

    return X


def build_instagram_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    # --- Pre-computed format (public dataset) ---
    if _is_precomputed_format(df):
        X = _build_from_precomputed(df)
        return X, df

    # --- Raw profile format (manual input / CSV upload / live lookup) ---
    for col in ("username", "bio"):
        if col not in df.columns:
            df[col] = ""

    df["followers"]  = safe_num(df, "followers",  0)
    df["following"]  = safe_num(df, "following",  0)
    df["posts"]      = safe_num(df, "posts",      0)
    df["is_verified"] = safe_num(df, "is_verified", 0)
    if "has_profile_pic" in df.columns:
        df["has_profile_pic"] = safe_num(df, "has_profile_pic", 1.0)
    elif "profile_pic" in df.columns:
        df["has_profile_pic"] = safe_num(df, "profile_pic", 1.0)
    else:
        df["has_profile_pic"] = 1.0

    # Ratio features
    df["ff_ratio"] = df["followers"] / df["following"].replace({0: 1})
    df["posts_per_100_followers"] = (
        df["posts"] / df["followers"].replace({0: 1}) * 100
    )

    # Log-scale transforms — essential for celebrity-range accounts (10M+ followers)
    df["log_followers"] = np.log1p(df["followers"])
    df["log_following"] = np.log1p(df["following"])
    df["log_posts"]     = np.log1p(df["posts"])

    # Follower tier: 0=micro(<1k), 1=mid(1k-100k), 2=macro(100k-1M), 3=mega(1M+)
    df["followers_tier"] = pd.cut(
        df["followers"],
        bins=[-1, 999, 99_999, 999_999, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Celebrity signal: verified + large following
    df["is_celebrity"] = (
        (df["is_verified"] == 1) & (df["followers"] > 100_000)
    ).astype(int)

    # Text features
    add_text_features(df, "username", "uname")
    add_text_features(df, "bio", "bio")

    # Username-specific signals
    df["uname_underscore"]   = df["username"].astype(str).str.count(r"_")
    df["uname_pure_numeric"] = df["username"].astype(str).str.fullmatch(r"\d+").astype(int)

    feature_cols = [
        # Core numeric
        "followers", "following", "posts",
        # Ratio signals
        "ff_ratio", "posts_per_100_followers",
        # Verification & profile completeness (critical for celebrities)
        "is_verified", "has_profile_pic", "is_celebrity",
        # Log-scale (handles 10M+ follower range)
        "log_followers", "log_following", "log_posts",
        # Tier bucket
        "followers_tier",
        # Username signals
        "uname_len", "uname_digit_count", "uname_underscore", "uname_pure_numeric",
        # Bio signals
        "bio_len", "bio_word_count", "bio_has_url", "bio_has_crypto",
        "bio_has_at", "bio_emoji_count", "bio_caps_ratio",
        "bio_has_spam", "bio_is_empty",
    ]

    X = df[feature_cols]
    return X, df
