from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_snapchat_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    if "username" not in df.columns:
        df["username"] = df["name"] if "name" in df.columns else ""
    if "bio" not in df.columns:
        df["bio"] = ""

    df["score"]        = safe_num(df, "score",         0)
    df["friends_count"] = safe_num(df, "friends_count", 0)
    df["streaks"]      = safe_num(df, "streaks",       0)

    # Ratio features
    df["score_per_friend"] = df["score"] / df["friends_count"].replace({0: 1})

    # Log-scale
    df["log_score"]   = np.log1p(df["score"])
    df["log_friends"] = np.log1p(df["friends_count"])

    # Score tier: 0=<1k, 1=1k-100k, 2=100k+
    df["score_tier"] = pd.cut(
        df["score"],
        bins=[-1, 999, 99_999, float("inf")],
        labels=[0, 1, 2],
    ).astype(float)

    df["has_streaks"] = (df["streaks"] > 0).astype(int)

    # Text features
    add_text_features(df, "username", "uname")
    add_text_features(df, "bio", "bio")

    # Username-specific
    df["uname_pure_numeric"] = df["username"].astype(str).str.fullmatch(r"\d+").astype(int)
    df["uname_underscore"]   = df["username"].astype(str).str.count(r"_")

    feature_cols = [
        "score", "friends_count", "streaks",
        "score_per_friend",
        "log_score", "log_friends",
        "score_tier", "has_streaks",
        "uname_len", "uname_digit_count", "uname_underscore", "uname_pure_numeric",
        "bio_len", "bio_word_count", "bio_has_url",
        "bio_has_crypto", "bio_has_spam", "bio_is_empty",
    ]

    X = df[feature_cols]
    return X, df
