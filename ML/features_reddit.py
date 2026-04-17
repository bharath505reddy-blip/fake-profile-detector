from typing import Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_reddit_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    if "username" not in df.columns:
        df["username"] = df["name"] if "name" in df.columns else ""
    if "about" not in df.columns:
        df["about"] = ""

    # Karma signals
    df["karma"]         = safe_num(df, "karma",         0)
    df["comment_karma"] = safe_num(df, "comment_karma", 0)
    df["post_karma"]    = safe_num(df, "post_karma",    0)

    # If karma columns are available, combine; otherwise use karma directly
    has_split = df["comment_karma"].sum() > 0 or df["post_karma"].sum() > 0
    if has_split:
        df["total_karma"] = df["comment_karma"] + df["post_karma"]
    else:
        df["total_karma"] = df["karma"]
    df["karma_ratio"] = df["comment_karma"] / (df["total_karma"].replace({0: 1}))

    # Account age (compute from created_at if present)
    if "created_at" in df.columns or "created_utc" in df.columns:
        col_name = "created_at" if "created_at" in df.columns else "created_utc"
        created = pd.to_datetime(df[col_name], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        df["account_age_days"] = ((now - created).dt.total_seconds() / 86400).fillna(0)
    else:
        df["account_age_days"] = safe_num(df, "account_age_days", 0)

    df["karma_per_day"] = df["total_karma"] / df["account_age_days"].replace({0: 1})

    # Log-scale
    df["log_karma"]       = np.log1p(df["total_karma"])
    df["log_account_age"] = np.log1p(df["account_age_days"])

    # Text features
    add_text_features(df, "username", "uname")
    add_text_features(df, "about", "about")

    # Username-specific
    df["uname_pure_numeric"] = df["username"].astype(str).str.fullmatch(r"\d+").astype(int)
    df["uname_underscore"]   = df["username"].astype(str).str.count(r"_")

    feature_cols = [
        "total_karma", "comment_karma", "post_karma", "karma_ratio",
        "account_age_days", "karma_per_day",
        "log_karma", "log_account_age",
        "uname_len", "uname_digit_count", "uname_underscore", "uname_pure_numeric",
        "about_len", "about_word_count", "about_has_url",
        "about_has_crypto", "about_has_spam", "about_is_empty",
    ]

    X = df[feature_cols]
    return X, df
