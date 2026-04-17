from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_x_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    for col in ("username", "bio"):
        if col not in df.columns:
            df[col] = ""

    # Core numeric signals
    df["followers"]    = safe_num(df, "followers",    0)
    df["following"]    = safe_num(df, "following",    0)
    df["tweets"]       = safe_num(df, "tweets",
                                  safe_num(df, "statuses_count", 0))
    df["listed_count"] = safe_num(df, "listed_count", 0)
    df["is_verified"] = safe_num(df, "is_verified", 0)
    if "has_profile_pic" in df.columns:
        df["has_profile_pic"] = safe_num(df, "has_profile_pic", 1.0)
    elif "default_profile_image" in df.columns:
        df["has_profile_pic"] = 1.0 - safe_num(df, "default_profile_image", 0.0)
    else:
        df["has_profile_pic"] = 1.0

    # Ratio features
    df["ff_ratio"] = df["followers"] / df["following"].replace({0: 1})
    df["listed_per_follower"] = df["listed_count"] / df["followers"].replace({0: 1})
    df["tweet_to_follower"]   = df["tweets"] / df["followers"].replace({0: 1})

    # Log-scale transforms — essential for accounts with millions of followers
    df["log_followers"] = np.log1p(df["followers"])
    df["log_tweets"]    = np.log1p(df["tweets"])

    # Follower tier
    df["followers_tier"] = pd.cut(
        df["followers"],
        bins=[-1, 999, 99_999, 999_999, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Celebrity signal
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
        "followers", "following", "tweets", "listed_count",
        # Ratio signals
        "ff_ratio", "listed_per_follower", "tweet_to_follower",
        # Verification & profile completeness
        "is_verified", "has_profile_pic", "is_celebrity",
        # Log-scale
        "log_followers", "log_tweets",
        # Tier
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
