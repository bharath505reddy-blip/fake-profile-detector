from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features

try:
    from features.celebrity_detector import compute_celebrity_features
    _CELEB_AVAILABLE = True
except Exception:
    _CELEB_AVAILABLE = False


def build_facebook_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    if "name" not in df.columns:
        df["name"] = df["username"] if "username" in df.columns else ""
    if "bio" not in df.columns:
        df["bio"] = df["about"] if "about" in df.columns else ""

    df["friends"]       = safe_num(df, "friends",    0)
    df["followers"]     = safe_num(df, "followers",  0)
    df["posts"]         = safe_num(df, "posts",      0)
    df["is_verified"] = safe_num(df, "is_verified", 0)
    if "has_profile_pic" in df.columns:
        df["has_profile_pic"] = safe_num(df, "has_profile_pic", 1.0)
    elif "profile_pic" in df.columns:
        df["has_profile_pic"] = safe_num(df, "profile_pic", 1.0)
    else:
        df["has_profile_pic"] = 1.0

    # Ratio features
    df["friends_followers_ratio"] = df["friends"] / df["followers"].replace({0: 1})
    df["posts_per_friend"]        = df["posts"] / df["friends"].replace({0: 1})

    # Log-scale
    df["log_friends"]    = np.log1p(df["friends"])
    df["log_followers"]  = np.log1p(df["followers"])
    df["log_posts"]      = np.log1p(df["posts"])

    # Celebrity/public figure signal
    df["is_celebrity"] = (
        (df["is_verified"] == 1) & (df["followers"] > 10_000)
    ).astype(int)

    # Text features
    add_text_features(df, "name", "name")

    # Bio/about section
    bio_col = "bio" if "bio" in df.columns else "about"
    if bio_col not in df.columns:
        df[bio_col] = ""
    add_text_features(df, bio_col, "bio")

    # Name-specific signals
    df["name_word_count"] = df["name"].astype(str).str.split().str.len().fillna(0).astype(int)
    df["name_all_caps"]   = (df["name"].astype(str) == df["name"].astype(str).str.upper()).astype(int)

    feature_cols = [
        "friends", "followers", "posts",
        "friends_followers_ratio", "posts_per_friend",
        "is_verified", "has_profile_pic", "is_celebrity",
        "log_friends", "log_followers", "log_posts",
        "name_len", "name_digit_count", "name_has_url", "name_word_count", "name_all_caps",
        "bio_len", "bio_word_count", "bio_has_url", "bio_has_crypto",
        "bio_has_spam", "bio_is_empty",
    ]

    X = df[feature_cols].copy()

    if _CELEB_AVAILABLE:
        try:
            celeb_rows = [compute_celebrity_features(row.to_dict())
                          for _, row in df.iterrows()]
            celeb_df = pd.DataFrame(celeb_rows, index=df.index)
            for col in celeb_df.columns:
                X[col] = celeb_df[col].values
        except Exception:
            pass

    return X, df
