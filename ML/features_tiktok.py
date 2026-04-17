from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_tiktok_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    for col in ("username", "bio"):
        if col not in df.columns:
            df[col] = ""

    df["followers"]   = safe_num(df, "followers",  0)
    df["following"]   = safe_num(df, "following",  0)
    df["videos"]      = safe_num(df, "videos",
                                 safe_num(df, "posts", 0))
    df["likes"]       = safe_num(df, "likes",      0)
    df["is_verified"] = safe_num(df, "is_verified", 0)

    # Ratio features
    df["ff_ratio"]       = df["followers"] / df["following"].replace({0: 1})
    df["likes_per_video"] = df["likes"] / df["videos"].replace({0: 1})
    df["likes_per_follower"] = df["likes"] / df["followers"].replace({0: 1})

    # Log-scale
    df["log_followers"] = np.log1p(df["followers"])
    df["log_likes"]     = np.log1p(df["likes"])

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

    # Username-specific
    df["uname_pure_numeric"] = df["username"].astype(str).str.fullmatch(r"\d+").astype(int)
    df["uname_underscore"]   = df["username"].astype(str).str.count(r"_")

    feature_cols = [
        "followers", "following", "videos", "likes",
        "ff_ratio", "likes_per_video", "likes_per_follower",
        "is_verified", "is_celebrity",
        "log_followers", "log_likes",
        "followers_tier",
        "uname_len", "uname_digit_count", "uname_underscore", "uname_pure_numeric",
        "bio_len", "bio_word_count", "bio_has_url", "bio_has_crypto",
        "bio_has_spam", "bio_emoji_count", "bio_is_empty",
    ]

    X = df[feature_cols]
    return X, df
