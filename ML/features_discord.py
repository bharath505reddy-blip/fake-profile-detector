from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_discord_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    if "username" not in df.columns:
        df["username"] = ""

    # Parse timestamps
    for ts_col in ("created_at_utc", "joined_at_utc"):
        if ts_col not in df.columns:
            df[ts_col] = pd.NaT
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # Time between account creation and server join
    gap = (df["joined_at_utc"] - df["created_at_utc"]).dt.total_seconds()
    df["join_gap_seconds"] = pd.to_numeric(gap, errors="coerce").fillna(0)

    # Log of join gap — very small values indicate raid bots
    df["log_join_gap"] = np.log1p(df["join_gap_seconds"].clip(lower=0))

    # Account age
    df["account_age_days"] = safe_num(df, "account_age_days", 0)
    df["log_account_age"]  = np.log1p(df["account_age_days"])

    df["has_avatar"]      = safe_num(df, "has_avatar",      0)
    df["suspicion_score"] = safe_num(df, "suspicion_score", 0)

    # Indicators
    df["very_new_account"]  = (df["account_age_days"] < 7).astype(int)
    df["instant_join"]      = (df["join_gap_seconds"] < 60).astype(int)

    # Text features (username only for Discord)
    add_text_features(df, "username", "uname")

    # Username-specific
    df["uname_pure_numeric"] = df["username"].astype(str).str.fullmatch(r"\d+").astype(int)
    df["uname_underscore"]   = df["username"].astype(str).str.count(r"_")

    feature_cols = [
        "account_age_days", "log_account_age",
        "has_avatar",
        "suspicion_score",
        "join_gap_seconds", "log_join_gap",
        "very_new_account", "instant_join",
        "uname_len", "uname_digit_count",
        "uname_underscore", "uname_pure_numeric",
    ]

    X = df[feature_cols]
    return X, df
