from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features

try:
    from features.celebrity_detector import compute_celebrity_features
    _CELEB_AVAILABLE = True
except Exception:
    _CELEB_AVAILABLE = False


def build_youtube_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    if "channel_name" not in df.columns:
        df["channel_name"] = df["username"] if "username" in df.columns else ""
    if "about" not in df.columns:
        df["about"] = ""

    df["subscribers"] = safe_num(df, "subscribers", 0)
    df["videos"]      = safe_num(df, "videos",      0)
    df["avg_views"]   = safe_num(df, "avg_views",   0)
    df["is_verified"] = safe_num(df, "is_verified",  0)

    # Ratio features
    df["subs_per_video"]     = df["subscribers"] / df["videos"].replace({0: 1})
    df["views_per_sub"]      = df["avg_views"] / df["subscribers"].replace({0: 1})
    df["views_per_video"]    = df["avg_views"]  # already per-video if present

    # Log-scale
    df["log_subscribers"] = np.log1p(df["subscribers"])
    df["log_videos"]      = np.log1p(df["videos"])

    # Subscriber tier: 0=<1k, 1=1k-100k, 2=100k-1M, 3=1M+
    df["subs_tier"] = pd.cut(
        df["subscribers"],
        bins=[-1, 999, 99_999, 999_999, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Celebrity / monetization signal
    df["is_celebrity"] = (
        (df["is_verified"].astype(int) == 1) & (df["subscribers"] > 100_000)
    ).astype(int)

    # Text features
    add_text_features(df, "channel_name", "cname")
    add_text_features(df, "about", "about")

    feature_cols = [
        "subscribers", "videos", "avg_views",
        "subs_per_video", "views_per_sub",
        "is_verified", "is_celebrity",
        "log_subscribers", "log_videos",
        "subs_tier",
        "cname_len", "cname_digit_count",
        "about_len", "about_word_count", "about_has_url",
        "about_has_crypto", "about_has_spam", "about_is_empty",
    ]

    X = df[feature_cols].copy()

    if _CELEB_AVAILABLE:
        try:
            # YouTube uses subscribers as the followers analogue
            celeb_rows = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict.setdefault("followers", row_dict.get("subscribers", 0))
                celeb_rows.append(compute_celebrity_features(row_dict))
            celeb_df = pd.DataFrame(celeb_rows, index=df.index)
            for col in celeb_df.columns:
                X[col] = celeb_df[col].values
        except Exception:
            pass

    return X, df
