from typing import Tuple
import numpy as np
import pandas as pd
from .persistent_common import normalize_columns, safe_num, add_text_features


def build_github_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)

    for col in ("username", "bio"):
        if col not in df.columns:
            df[col] = ""

    df["followers"]    = safe_num(df, "followers",    0)
    df["following"]    = safe_num(df, "following",    0)
    df["public_repos"] = safe_num(df, "public_repos",
                                  safe_num(df, "repos", 0))
    df["public_gists"] = safe_num(df, "public_gists", 0)

    # Account age — critical signal (new accounts are highly suspicious)
    # Use 0 as default; callers should populate via live enrichment when possible
    df["account_age_days"] = safe_num(df, "account_age_days", 0)

    # ---- Ratio features ----
    df["ff_ratio"]           = df["followers"] / df["following"].replace({0: 1})
    df["repos_per_follower"] = df["public_repos"] / df["followers"].replace({0: 1})
    df["gists_per_repo"]     = df["public_gists"] / df["public_repos"].replace({0: 1})

    # Engagement signal from enrichment (stars received / repos)
    df["star_received_total"] = safe_num(df, "star_received_total", 0)
    df["stars_per_repo"] = df["star_received_total"] / df["public_repos"].replace({0: 1})

    # Fork ratio from enrichment
    df["fork_ratio"]              = safe_num(df, "fork_ratio", 0)
    df["readme_presence_ratio"]   = safe_num(df, "readme_presence_ratio", 0)
    df["unique_languages"]        = safe_num(df, "unique_languages", 0)
    df["repo_diversity_score"]    = safe_num(df, "repo_diversity_score", 0)
    df["activity_recency"]        = safe_num(df, "activity_recency", 9999)
    df["commit_regularity"]       = safe_num(df, "commit_regularity", 0)

    # Activity recency clipped — convert 9999 (no events) to a high but bounded value
    df["activity_recency_capped"] = df["activity_recency"].clip(upper=365)

    # ---- Profile completeness score (0-4) ----
    has_bio      = (df["bio"].astype(str).str.strip().str.len() > 0).astype(int)
    has_repos    = (df["public_repos"] > 0).astype(int)
    has_gists    = (df["public_gists"] > 0).astype(int)
    has_activity = (df["activity_recency"] < 365).astype(int)
    df["profile_completeness"] = has_bio + has_repos + has_gists + has_activity
    df["has_bio"] = has_bio

    # ---- Account age derived features ----
    df["is_new_account"]  = (df["account_age_days"] < 30).astype(int)
    df["is_young_account"] = (df["account_age_days"] < 90).astype(int)
    df["repos_per_day"]   = df["public_repos"] / df["account_age_days"].replace({0: 1})
    # Cap at a sane maximum to prevent outliers
    df["repos_per_day"]   = df["repos_per_day"].clip(upper=10)

    # ---- Log-scale features ----
    df["log_followers"]        = np.log1p(df["followers"])
    df["log_repos"]            = np.log1p(df["public_repos"])
    df["log_following"]        = np.log1p(df["following"])
    df["log_account_age"]      = np.log1p(df["account_age_days"])

    # ---- Activity signal: repos + gists combined ----
    df["total_public_work"] = df["public_repos"] + df["public_gists"]

    # ---- Text features ----
    add_text_features(df, "username", "uname")
    add_text_features(df, "bio", "bio")

    # ---- Username patterns ----
    df["uname_hyphen"]       = df["username"].astype(str).str.count(r"-")
    df["uname_underscore"]   = df["username"].astype(str).str.count(r"_")
    df["uname_pure_numeric"] = df["username"].astype(str).str.fullmatch(r"\d+").astype(int)

    feature_cols = [
        # Core account metrics
        "followers", "following", "public_repos", "public_gists",
        # Ratios
        "ff_ratio", "repos_per_follower", "gists_per_repo",
        # Log-scale
        "log_followers", "log_repos", "log_following", "log_account_age",
        # Activity
        "total_public_work", "has_bio",
        # Account age signals (most discriminative)
        "account_age_days", "is_new_account", "is_young_account", "repos_per_day",
        # Repository quality signals
        "star_received_total", "stars_per_repo", "fork_ratio",
        "readme_presence_ratio", "unique_languages", "repo_diversity_score",
        # Activity recency
        "activity_recency_capped", "commit_regularity",
        # Profile completeness
        "profile_completeness",
        # Username features
        "uname_len", "uname_digit_count", "uname_hyphen",
        "uname_underscore", "uname_pure_numeric",
        # Bio features
        "bio_len", "bio_word_count", "bio_has_url",
        "bio_has_crypto", "bio_has_spam", "bio_is_empty",
    ]

    X = df[feature_cols]
    return X, df
