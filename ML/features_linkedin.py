from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_linkedin_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    for col in ("headline", "about"):
        if col not in df.columns:
            df[col] = ""
    if "name" not in df.columns:
        df["name"] = df["username"] if "username" in df.columns else ""

    df["connections"]   = safe_num(df, "connections",   0)
    df["followers"]     = safe_num(df, "followers",     0)
    df["endorsements"]  = safe_num(df, "endorsements",  0)
    if "has_photo" in df.columns:
        df["has_photo"] = safe_num(df, "has_photo", 1.0)
    elif "profile_pic" in df.columns:
        df["has_photo"] = safe_num(df, "profile_pic", 1.0)
    else:
        df["has_photo"] = 1.0

    # Ratio features
    df["conn_follow_ratio"] = df["connections"] / df["followers"].replace({0: 1})
    df["endorse_per_conn"]  = df["endorsements"] / df["connections"].replace({0: 1})

    # Log-scale
    df["log_connections"] = np.log1p(df["connections"])
    df["log_endorsements"] = np.log1p(df["endorsements"])

    # Connection tier: 0=<50, 1=50-500, 2=500+ (LinkedIn's "500+ connections")
    df["conn_tier"] = pd.cut(
        df["connections"],
        bins=[-1, 49, 499, float("inf")],
        labels=[0, 1, 2],
    ).astype(float)

    # Text features
    add_text_features(df, "name", "name")
    add_text_features(df, "headline", "headline")
    add_text_features(df, "about", "about")

    # Name-specific
    df["name_word_count"] = df["name"].astype(str).str.split().str.len().fillna(0).astype(int)

    # Headline-specific signals (job title pattern detection)
    df["headline_has_pipe"] = df["headline"].astype(str).str.contains(r"\|", regex=False).astype(int)
    df["headline_has_at"]   = df["headline"].astype(str).str.contains(r"@", regex=False).astype(int)

    feature_cols = [
        "connections", "followers", "endorsements",
        "conn_follow_ratio", "endorse_per_conn",
        "has_photo",
        "log_connections", "log_endorsements",
        "conn_tier",
        "name_len", "name_digit_count", "name_word_count",
        "headline_len", "headline_has_url", "headline_has_crypto",
        "headline_word_count", "headline_has_pipe", "headline_has_at", "headline_has_spam",
        "about_len", "about_word_count", "about_has_url", "about_has_crypto",
        "about_has_spam", "about_is_empty",
    ]

    X = df[feature_cols]
    return X, df
