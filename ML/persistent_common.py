# ML/persistent_common.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import logging

from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, IsolationForest,
    StackingClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost unavailable — using RF+SVM ensemble only. "
                   "On macOS, run: brew install libomp")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    logger.warning("shap not available — SHAP explainability disabled.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    logger.warning("lightgbm unavailable — stacking will use RF+XGB+SVM only.")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn unavailable — class balancing disabled.")


LABEL_COLUMNS = ["label", "is_fake", "fake", "target", "is_bot", "bot"]
RF_ESTIMATORS = 300

# ---------------------------------------------------------------------------
# Log-transform preprocessor — fixes celebrity outlier clipping
# ---------------------------------------------------------------------------

# Features with power-law distributions that need log1p before scaling.
# Raw followers/following for celebrity accounts (10M+) get clipped by
# RobustScaler when trained on regular users. Log-transforming first
# preserves the information at the high end of the distribution.
LOG_TRANSFORM_FEATURES = {
    "followers", "following", "posts", "tweets", "subscribers",
    "total_views", "snap_score", "post_karma", "comment_karma",
    "total_likes", "total_stars", "listed_count", "friends",
    "videos", "likes",
}


class LogRobustScaler(BaseEstimator, TransformerMixin):
    """
    Two-phase preprocessing: log1p-transform power-law columns, then RobustScaler.

    Avoids clipping celebrity-tier accounts (10M+ followers) that RobustScaler
    would treat as outliers when trained on regular-user distributions.
    Skips columns already log-transformed by feature builders (log_* prefix).
    """

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.log_col_indices_ = [
                i for i, col in enumerate(X.columns)
                if col.lower() in LOG_TRANSFORM_FEATURES
                and not col.lower().startswith("log_")
            ]
        else:
            self.log_col_indices_ = []
        self.scaler_ = RobustScaler(with_centering=False)
        self.scaler_.fit(self._apply_log(X))
        return self

    def transform(self, X):
        return self.scaler_.transform(self._apply_log(X))

    def _apply_log(self, X):
        if hasattr(X, "values"):
            arr = X.values.copy().astype(float)
        else:
            arr = np.array(X, dtype=float).copy()
        for idx in self.log_col_indices_:
            arr[:, idx] = np.log1p(np.maximum(arr[:, idx], 0))
        return arr


def build_preprocessor(feature_names: list) -> "Pipeline":
    """
    Build a ColumnTransformer-based preprocessing pipeline when feature names
    are known ahead of time. Applies log1p + RobustScaler to power-law features
    and plain RobustScaler to everything else.

    Use LogRobustScaler when feature names are unknown at model-build time.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer

    log_cols = [f for f in feature_names
                if f.lower() in LOG_TRANSFORM_FEATURES
                and not f.lower().startswith("log_")]
    other_cols = [f for f in feature_names if f not in log_cols]

    log_pipe = Pipeline([
        ("log",    FunctionTransformer(lambda x: np.log1p(np.maximum(x, 0)), validate=True)),
        ("scaler", RobustScaler()),
    ])

    transformers = [("log_transform", log_pipe, log_cols)] if log_cols else []
    if other_cols:
        transformers.append(("standard_scale", RobustScaler(), other_cols))

    return ColumnTransformer(transformers, remainder="passthrough")

# Feature group weights — applied as column multipliers after RobustScaler.
# Higher weight = feature group has more influence on the model.
FEATURE_GROUP_WEIGHTS = {
    "base_metadata":      1.0,   # followers, following, posts, ff_ratio, account_age_days …
    "username_forensics": 0.8,   # uname_* columns
    "bio_forensics":      0.9,   # bio_* columns
    "photo_analysis":     1.2,   # has_avatar, avatar_*, photo_* columns
    "behavioral":         1.5,   # behavioral_*, timing_*, engagement_*, hhi_* columns
    "cross_platform":     1.3,   # cross_platform_*, xp_* columns
    "temporal":           1.1,   # temporal_*, post_freq_* columns
}

def _get_feature_group_weight(col: str) -> float:
    """Return the group weight multiplier for a feature column name."""
    col_l = col.lower()
    if any(col_l.startswith(p) for p in ("behavioral_", "timing_", "engagement_", "hhi_", "topic_")):
        return FEATURE_GROUP_WEIGHTS["behavioral"]
    if any(col_l.startswith(p) for p in ("cross_platform_", "xp_")):
        return FEATURE_GROUP_WEIGHTS["cross_platform"]
    if any(col_l.startswith(p) for p in ("has_avatar", "avatar_", "photo_")):
        return FEATURE_GROUP_WEIGHTS["photo_analysis"]
    if col_l.startswith("bio_"):
        return FEATURE_GROUP_WEIGHTS["bio_forensics"]
    if any(col_l.startswith(p) for p in ("uname_", "username_")):
        return FEATURE_GROUP_WEIGHTS["username_forensics"]
    if any(col_l.startswith(p) for p in ("temporal_", "post_freq_")):
        return FEATURE_GROUP_WEIGHTS["temporal"]
    return FEATURE_GROUP_WEIGHTS["base_metadata"]


def apply_feature_group_weights(X: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of X with each column multiplied by its feature-group weight.
    Applied after RobustScaler so all features are on comparable scales first.
    """
    X_w = X.copy()
    for col in X_w.columns:
        w = _get_feature_group_weight(col)
        if w != 1.0:
            X_w[col] = X_w[col] * w
    return X_w

# Optional Optuna for hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# HistGradientBoosting — sklearn native, fast, handles missing values
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HISTGB_AVAILABLE = True
except ImportError:
    HISTGB_AVAILABLE = False

# Human-readable descriptions for common feature names (used for tooltips)
FEATURE_DESCRIPTIONS = {
    "followers": "Number of followers the account has",
    "following": "Number of accounts this user follows",
    "posts": "Total number of posts/tweets made",
    "ff_ratio": "Followers-to-following ratio — low values suggest bot-like behavior",
    "posts_per_100_followers": "Post activity relative to follower count",
    "uname_len": "Length of the username",
    "uname_digit_count": "Number of digits in the username — many digits suggest bot",
    "bio_len": "Length of the bio text",
    "bio_has_url": "Whether the bio contains a URL (1=yes)",
    "bio_has_crypto": "Whether the bio mentions crypto/airdrop keywords",
    "bio_has_at": "Whether the bio contains @mentions",
    "bio_digit_count": "Number of digits in the bio",
    "karma": "Reddit karma score",
    "account_age_days": "Age of the account in days — very new accounts are suspicious",
    "has_avatar": "Whether the profile has a custom avatar (1=yes)",
    "suspicion_score": "Pre-computed suspicion score from Discord analysis",
    "subscribers": "YouTube subscriber count",
    "videos": "Number of uploaded videos",
    "public_repos": "Number of public GitHub repositories",
    "connections": "LinkedIn connection count",
    "score": "Snapchat score (total snaps sent/received)",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_label_col(df: pd.DataFrame) -> Optional[str]:
    for c in LABEL_COLUMNS:
        if c in df.columns:
            return c
    return None


def coerce_label_binary(s: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "fake", "bot", "spam"}
    falsey = {"false", "0", "no", "n", "legit", "real", "human", "genuine"}
    def to_bin(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        t = str(v).strip().lower()
        if t in truthy:
            return 1
        if t in falsey:
            return 0
        return np.nan
    return s.map(to_bin)


def safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _count_emoji(text: str) -> int:
    """Count Unicode emoji characters using standard code-point ranges."""
    count = 0
    for char in str(text):
        cp = ord(char)
        if (0x1F600 <= cp <= 0x1F64F or  # Emoticons
                0x1F300 <= cp <= 0x1F5FF or  # Misc symbols & pictographs
                0x1F680 <= cp <= 0x1F6FF or  # Transport & map
                0x1F1E0 <= cp <= 0x1F1FF or  # Flags
                0x2600 <= cp <= 0x26FF or    # Misc symbols
                0x2700 <= cp <= 0x27BF or    # Dingbats
                0x1F900 <= cp <= 0x1F9FF):   # Supplemental symbols
            count += 1
    return count


def add_text_features(df: pd.DataFrame, col: str, prefix: str) -> None:
    """Comprehensive text feature extraction for a single text column."""
    s = df[col].fillna("").astype(str)

    # Length and word count
    df[f"{prefix}_len"]        = s.str.len()
    df[f"{prefix}_word_count"] = s.str.split().str.len().fillna(0).astype(int)

    # Character-level signals
    df[f"{prefix}_digit_count"] = s.str.count(r"\d")
    df[f"{prefix}_caps_ratio"]  = s.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df[f"{prefix}_special_ratio"] = s.apply(
        lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / max(len(x), 1)
    )

    # Content signals
    df[f"{prefix}_has_url"]    = s.str.contains(r"https?://|www\.", regex=True).astype(int)
    df[f"{prefix}_has_at"]     = s.str.contains(r"@", regex=False).astype(int)
    df[f"{prefix}_has_crypto"] = s.str.contains(
        r"\b(?:btc|eth|crypto|airdrop|nft|coin|token|forex|defi)\b",
        case=False, regex=True,
    ).astype(int)
    df[f"{prefix}_has_spam"]   = s.str.contains(
        r"\b(?:free|win|click here|earn|make money|passive income|"
        r"gain followers|follow back|promo|dm me|link in bio.*earn|"
        r"investment advisor|99% win rate|get rich)\b",
        case=False, regex=True,
    ).astype(int)
    df[f"{prefix}_is_empty"]   = (s.str.strip().str.len() == 0).astype(int)

    # Emoji count (bot bios often have 0 or excessive emojis)
    df[f"{prefix}_emoji_count"] = s.apply(_count_emoji)


def default_model(tuned_params: Optional[dict] = None) -> Pipeline:
    """
    Build a high-accuracy ensemble (RF + XGBoost + HistGB + SVM) with soft voting.
    tuned_params: optional dict from optuna_tune() to override defaults.
    """
    p = tuned_params or {}

    rf = RandomForestClassifier(
        n_estimators=p.get("rf_n_estimators", RF_ESTIMATORS),
        max_depth=p.get("rf_max_depth", 20),
        min_samples_leaf=p.get("rf_min_samples_leaf", 2),
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    svm = CalibratedClassifierCV(
        SVC(
            kernel="rbf",
            C=p.get("svm_C", 5.0),
            gamma=p.get("svm_gamma", "scale"),
            probability=False,
            random_state=42,
        ),
        cv=3,
    )
    estimators = [("rf", rf), ("svm", svm)]

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=p.get("xgb_n_estimators", RF_ESTIMATORS),
            learning_rate=p.get("xgb_learning_rate", 0.05),
            max_depth=p.get("xgb_max_depth", 6),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=p.get("xgb_reg_alpha", 0.1),
            reg_lambda=p.get("xgb_reg_lambda", 1.0),
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        estimators.append(("xgb", xgb))

    if HISTGB_AVAILABLE:
        hgb = HistGradientBoostingClassifier(
            max_iter=p.get("hgb_max_iter", 200),
            learning_rate=p.get("hgb_lr", 0.05),
            max_depth=p.get("hgb_depth", 6),
            l2_regularization=p.get("hgb_l2", 0.1),
            random_state=42,
        )
        estimators.append(("hgb", hgb))

    if LIGHTGBM_AVAILABLE:
        lgbm = LGBMClassifier(
            n_estimators=p.get("lgbm_n_estimators", RF_ESTIMATORS),
            learning_rate=p.get("lgbm_learning_rate", 0.05),
            num_leaves=p.get("lgbm_num_leaves", 31),
            reg_alpha=p.get("lgbm_reg_alpha", 0.1),
            reg_lambda=p.get("lgbm_reg_lambda", 0.5),
            min_child_weight=p.get("lgbm_min_child_weight", 5),
            feature_fraction=p.get("lgbm_feature_fraction", 0.8),
            random_state=42,
            verbose=-1,
        )
        estimators.append(("lgbm", lgbm))

    ensemble = VotingClassifier(estimators=estimators, voting="soft")

    return Pipeline([
        ("imputer",      SimpleImputer(strategy="median")),
        ("preprocessor", LogRobustScaler()),
        ("ensemble",     ensemble),
    ])


def optuna_tune(X: pd.DataFrame, y: pd.Series,
                n_trials: int = 40, timeout: int = 120) -> dict:
    """
    Use Optuna to tune RF + XGB hyperparameters.
    Returns best params dict to pass into default_model().
    Falls back to {} if Optuna is unavailable or tuning fails.
    """
    if not OPTUNA_AVAILABLE:
        return {}
    try:
        from sklearn.model_selection import cross_val_score

        def objective(trial):
            rf = RandomForestClassifier(
                n_estimators=trial.suggest_int("rf_n_estimators", 100, 500),
                max_depth=trial.suggest_int("rf_max_depth", 5, 25),
                min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 5),
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            scores = cross_val_score(
                rf, X.fillna(0), y, cv=3, scoring="f1", n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        logger.info("Optuna best F1=%.4f params=%s", study.best_value, study.best_params)
        return study.best_params
    except Exception as exc:
        logger.warning("Optuna tuning failed: %s — using defaults", exc)
        return {}


def stacking_model(tuned_params: Optional[dict] = None) -> Pipeline:
    """
    Stacking ensemble: RF + XGB + HistGB + SVM + LightGBM → LogisticRegression meta.
    More powerful than voting but slower to train.
    """
    p = tuned_params or {}

    rf = RandomForestClassifier(
        n_estimators=p.get("rf_n_estimators", RF_ESTIMATORS),
        max_depth=p.get("rf_max_depth", 20),
        min_samples_leaf=p.get("rf_min_samples_leaf", 2),
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=5.0, gamma="scale", probability=False, random_state=42),
        cv=3,
    )
    estimators = [("rf", rf), ("svm", svm)]

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=p.get("xgb_n_estimators", RF_ESTIMATORS),
            learning_rate=p.get("xgb_learning_rate", 0.05),
            max_depth=p.get("xgb_max_depth", 6),
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss",
            verbosity=0,
        )
        estimators.append(("xgb", xgb))

    if HISTGB_AVAILABLE:
        hgb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6, random_state=42,
        )
        estimators.append(("hgb", hgb))

    if LIGHTGBM_AVAILABLE:
        lgbm = LGBMClassifier(
            n_estimators=p.get("lgbm_n_estimators", RF_ESTIMATORS),
            learning_rate=p.get("lgbm_learning_rate", 0.05),
            num_leaves=p.get("lgbm_num_leaves", 31),
            reg_alpha=p.get("lgbm_reg_alpha", 0.1),
            reg_lambda=p.get("lgbm_reg_lambda", 0.5),
            min_child_weight=p.get("lgbm_min_child_weight", 5),
            feature_fraction=p.get("lgbm_feature_fraction", 0.8),
            random_state=42,
            verbose=-1,
        )
        estimators.append(("lgbm", lgbm))

    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    stacking = StackingClassifier(
        estimators=estimators, final_estimator=meta, cv=3, n_jobs=-1, passthrough=False
    )

    return Pipeline([
        ("imputer",      SimpleImputer(strategy="median")),
        ("preprocessor", LogRobustScaler()),
        ("ensemble",     stacking),
    ])


def save_charts(charts_dir: Path, tag: str, title: str, fake_count: int, legit_count: int) -> Dict[str, str]:
    bar_name = f"bar_{tag}.png"
    pie_name = f"pie_{tag}.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Fake", "Legit"], [fake_count, legit_count],
                  color=["#dc3545", "#198754"], edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Profile Type")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, [fake_count, legit_count]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(charts_dir / bar_name, dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    total = fake_count + legit_count
    wedge_sizes = [fake_count or 1e-9, legit_count or 1e-9]
    wedges, texts, autotexts = ax.pie(
        wedge_sizes,
        labels=["Fake", "Legit"],
        colors=["#dc3545", "#198754"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title(f"Distribution ({total} profiles)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(charts_dir / pie_name, dpi=120)
    plt.close()

    return {"bar": f"charts/{bar_name}", "pie": f"charts/{pie_name}"}


def save_feature_importance_chart(
    charts_dir: Path, feature_names: List[str], importances, platform: str
) -> str:
    """Save a horizontal feature-importance bar chart."""
    pairs = sorted(zip(importances, feature_names), reverse=True)
    top_pairs = pairs[:20]  # Show top 20
    imp_vals = [p[0] for p in top_pairs]
    feat_names = [p[1] for p in top_pairs]

    fig_height = max(4, len(feat_names) * 0.45)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    colors = ["#dc3545" if v > np.mean(imp_vals) else "#0d6efd" for v in imp_vals[::-1]]
    bars = ax.barh(feat_names[::-1], imp_vals[::-1], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"{platform.capitalize()} — Top Feature Importances (Random Forest)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, imp_vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    name = f"{platform}_importance.png"
    plt.savefig(charts_dir / name, bbox_inches="tight", dpi=120)
    plt.close()
    return f"charts/{name}"


def save_shap_chart(
    charts_dir: Path, shap_values: np.ndarray, feature_names: List[str],
    platform: str, tag: str = "shap"
) -> Optional[str]:
    """Save a SHAP summary bar chart showing mean |SHAP| per feature."""
    if not SHAP_AVAILABLE:
        return None
    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        pairs = sorted(zip(mean_abs_shap, feature_names), reverse=True)[:15]
        vals = [p[0] for p in pairs]
        names = [p[1] for p in pairs]

        fig_height = max(4, len(names) * 0.45)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        cmap = plt.cm.RdBu_r
        norm_vals = [(v - min(vals)) / (max(vals) - min(vals) + 1e-9) for v in vals[::-1]]
        colors = [cmap(nv) for nv in norm_vals]
        ax.barh(names[::-1], vals[::-1], color=colors, edgecolor="white")
        ax.set_xlabel("Mean |SHAP value| — Impact on model output")
        ax.set_title(f"{platform.capitalize()} — SHAP Feature Explanation", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        name = f"{tag}_{platform}_shap.png"
        plt.savefig(charts_dir / name, bbox_inches="tight", dpi=120)
        plt.close()
        return f"charts/{name}"
    except Exception as exc:
        logger.warning("SHAP chart failed: %s", exc)
        return None


def save_shap_single_chart(
    charts_dir: Path, shap_row: np.ndarray, feature_names: List[str],
    base_value: float, platform: str, tag: str
) -> Optional[str]:
    """Save a waterfall-style SHAP chart for a single prediction."""
    if not SHAP_AVAILABLE:
        return None
    try:
        pairs = sorted(zip(shap_row, feature_names), key=lambda x: abs(x[0]), reverse=True)[:12]
        vals = [p[0] for p in pairs]
        names = [p[1] for p in pairs]

        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5)))
        colors = ["#dc3545" if v > 0 else "#198754" for v in vals[::-1]]
        ax.barh(names[::-1], vals[::-1], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (red = pushes toward Fake, green = pushes toward Legit)")
        ax.set_title(f"{platform.capitalize()} — Why this prediction? (SHAP)", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        name = f"{tag}_{platform}_shap_single.png"
        plt.savefig(charts_dir / name, bbox_inches="tight", dpi=120)
        plt.close()
        return f"charts/{name}"
    except Exception as exc:
        logger.warning("SHAP single chart failed: %s", exc)
        return None


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame
) -> Tuple[Optional[np.ndarray], float]:
    """Compute SHAP values using the RF component of the ensemble pipeline."""
    if not SHAP_AVAILABLE:
        return None, 0.0
    try:
        # Transform features through all steps except the final ensemble
        preprocessing = pipeline[:-1]
        X_transformed = preprocessing.transform(X)

        # Extract RF estimator from VotingClassifier (estimators_ is a flat list)
        ensemble = pipeline.named_steps["ensemble"]
        rf = next((e for e in ensemble.estimators_
                   if isinstance(e, RandomForestClassifier)), None)
        if rf is None:
            return None, 0.0

        explainer = shap.TreeExplainer(rf)
        raw = explainer.shap_values(X_transformed)
        # For binary classification, shap_values returns [class0, class1] or single array
        if isinstance(raw, list) and len(raw) == 2:
            shap_vals = raw[1]  # class 1 = Fake
        else:
            shap_vals = raw
        base_value = float(explainer.expected_value[1]) if isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else float(explainer.expected_value)
        return shap_vals, base_value
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None, 0.0


def detect_anomalies(X: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
    """Use IsolationForest to compute anomaly scores (-1=anomaly, 1=normal)."""
    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        # anomaly_score: -1 = anomaly (more fake), 1 = normal
        scores = iso.fit_predict(X.fillna(0))
        # decision_function gives raw scores (lower = more anomalous)
        raw_scores = iso.decision_function(X.fillna(0))
        return scores, raw_scores
    except Exception as exc:
        logger.warning("Anomaly detection failed: %s", exc)
        return np.ones(len(X)), np.zeros(len(X))


def save_dataset_stats_charts(
    charts_dir: Path, X: pd.DataFrame, y: pd.Series, platform: str
) -> Dict[str, str]:
    """Generate dataset statistics charts for the stats view."""
    charts = {}
    tag = f"stats_{platform}"

    # Class distribution
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = y.value_counts().sort_index()
        labels = ["Legit" if k == 0 else "Fake" for k in counts.index]
        colors = ["#198754" if k == 0 else "#dc3545" for k in counts.index]
        ax.bar(labels, counts.values, color=colors, edgecolor="white")
        ax.set_title("Class Distribution", fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        name = f"{tag}_class_dist.png"
        plt.savefig(charts_dir / name, dpi=100)
        plt.close()
        charts["class_dist"] = f"charts/{name}"
    except Exception:
        pass

    # Feature correlation heatmap (top 10 features)
    try:
        top_feats = X.columns[:min(10, len(X.columns))]
        corr = X[top_feats].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(top_feats)))
        ax.set_yticks(range(len(top_feats)))
        ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(top_feats, fontsize=8)
        ax.set_title("Feature Correlation Matrix", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(len(top_feats)):
            for j in range(len(top_feats)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        plt.tight_layout()
        name = f"{tag}_corr.png"
        plt.savefig(charts_dir / name, bbox_inches="tight", dpi=100)
        plt.close()
        charts["correlation"] = f"charts/{name}"
    except Exception:
        pass

    # Feature distributions (box plots for top 8 numeric features)
    try:
        num_feats = X.select_dtypes(include=np.number).columns[:8]
        if len(num_feats) > 0:
            fig, axes = plt.subplots(2, 4, figsize=(14, 6))
            axes = axes.flatten()
            for i, feat in enumerate(num_feats):
                if i >= 8:
                    break
                vals_legit = X[feat][y == 0].dropna()
                vals_fake = X[feat][y == 1].dropna()
                axes[i].hist(vals_legit, bins=20, alpha=0.6, color="#198754", label="Legit", density=True)
                axes[i].hist(vals_fake, bins=20, alpha=0.6, color="#dc3545", label="Fake", density=True)
                axes[i].set_title(feat, fontsize=8)
                axes[i].tick_params(labelsize=7)
                axes[i].spines["top"].set_visible(False)
                axes[i].spines["right"].set_visible(False)
            for j in range(len(num_feats), 8):
                axes[j].set_visible(False)
            handles = [
                plt.Rectangle((0, 0), 1, 1, fc="#198754", alpha=0.6),
                plt.Rectangle((0, 0), 1, 1, fc="#dc3545", alpha=0.6),
            ]
            fig.legend(handles, ["Legit", "Fake"], loc="upper right", fontsize=9)
            fig.suptitle(f"{platform.capitalize()} — Feature Distributions by Class", fontweight="bold")
            plt.tight_layout()
            name = f"{tag}_distributions.png"
            plt.savefig(charts_dir / name, bbox_inches="tight", dpi=100)
            plt.close()
            charts["distributions"] = f"charts/{name}"
    except Exception:
        pass

    return charts


def train_and_save(
    labeled_csv_paths: List[Path],
    model_path: Path,
    feature_builder,
    min_rows: int = 10,
    charts_dir: Optional[Path] = None,
    platform: str = "",
    include_stats: bool = False,
    use_stacking: bool = False,
    balancing_strategy: str = "none",
    use_temporal_cv: bool = False,
    use_feature_selection: bool = False,
    use_optuna: bool = False,
    optuna_trials: int = 40,
) -> Tuple[bool, str, dict, int]:
    """
    Train an ensemble on the supplied labeled CSVs, persist it, and return
    (ok, message, metrics_dict, rows_used).

    New parameters:
      use_stacking         — Use StackingClassifier instead of VotingClassifier.
      balancing_strategy   — "none" | "smote" | "adasyn" | "class_weight"
      use_temporal_cv      — Use TimeSeriesSplit instead of StratifiedKFold.
      use_feature_selection — Apply RFE + correlation filter before training.
    """
    dfs = []
    for p in labeled_csv_paths:
        df = pd.read_csv(p)
        df = normalize_columns(df)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    label_col = find_label_col(full)
    if not label_col:
        return False, f"No label column found. Add one of: {', '.join(LABEL_COLUMNS)}", {}, 0

    y = coerce_label_binary(full[label_col])
    keep = y.notna()
    y = y[keep].astype(int)

    X, _ = feature_builder(full.loc[keep])

    if len(X) < min_rows:
        return False, f"Not enough labeled rows (need >= {min_rows}).", {}, 0
    if y.nunique() < 2:
        return False, "Only one class present (need both Fake and Legit).", {}, 0

    # --- Feature selection ---
    dropped_features: List[str] = []
    if use_feature_selection:
        try:
            # Correlation filter: drop one of each highly correlated pair
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            if corr_drop:
                logger.info("[%s] Dropping %d corr features: %s", platform, len(corr_drop), corr_drop)
                dropped_features.extend(corr_drop)
                X = X.drop(columns=corr_drop, errors="ignore")

            # RFE: keep top features by RF importance (drop below 1% threshold)
            if len(X.columns) > 3:
                rf_sel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf_sel.fit(X.fillna(0), y)
                importances = rf_sel.feature_importances_
                mean_imp = importances.mean()
                threshold = max(mean_imp * 0.5, 0.01)
                low_imp = [col for col, imp in zip(X.columns, importances) if imp < threshold]
                if low_imp:
                    logger.info("[%s] Dropping %d low-importance features: %s", platform, len(low_imp), low_imp)
                    dropped_features.extend(low_imp)
                    X = X.drop(columns=low_imp, errors="ignore")
        except Exception as exc:
            logger.warning("Feature selection failed: %s — continuing with all features", exc)

    # --- Class distribution before balancing ---
    class_dist_before = y.value_counts().to_dict()
    logger.info("[%s] Class distribution before balancing: %s", platform, class_dist_before)

    # --- Class balancing ---
    X_train, y_train = X, y
    if balancing_strategy in ("smote", "adasyn") and IMBLEARN_AVAILABLE:
        try:
            sampler = SMOTE(random_state=42) if balancing_strategy == "smote" else ADASYN(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X.fillna(0), y)
            X_train = pd.DataFrame(X_resampled, columns=X.columns)
            y_train = pd.Series(y_resampled)
            class_dist_after = pd.Series(y_train).value_counts().to_dict()
            logger.info("[%s] Class distribution after %s: %s", platform, balancing_strategy.upper(), class_dist_after)
        except Exception as exc:
            logger.warning("Balancing strategy %s failed: %s — using original data", balancing_strategy, exc)
            X_train, y_train = X, y
    elif balancing_strategy in ("smote", "adasyn"):
        logger.warning("imbalanced-learn not available — skipping %s balancing", balancing_strategy)

    # --- Cross-validation metrics ---
    metrics: dict = {}
    n_splits = min(5, int(min(y_train.value_counts())))
    if n_splits >= 2:
        try:
            if use_temporal_cv:
                cv = TimeSeriesSplit(n_splits=n_splits)
                logger.info("[%s] Using TimeSeriesSplit (temporal CV)", platform)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            cv_model = stacking_model() if use_stacking else default_model()
            cv_scores = cross_validate(
                cv_model, X_train, y_train, cv=cv,
                scoring=["accuracy", "precision", "recall", "f1"],
            )
            metrics = {
                "accuracy":  round(float(cv_scores["test_accuracy"].mean()), 3),
                "precision": round(float(cv_scores["test_precision"].mean()), 3),
                "recall":    round(float(cv_scores["test_recall"].mean()), 3),
                "f1":        round(float(cv_scores["test_f1"].mean()), 3),
                "cv_folds":  n_splits,
                "cv_type":   "temporal" if use_temporal_cv else "stratified",
                "balancing": balancing_strategy,
                "use_stacking": use_stacking,
            }
            if dropped_features:
                metrics["dropped_features"] = dropped_features
        except Exception as exc:
            logger.warning("Cross-validation failed: %s", exc)

    # --- Optional Optuna hyperparameter tuning ---
    tuned_params: dict = {}
    if use_optuna and OPTUNA_AVAILABLE and not use_stacking:
        logger.info("[%s] Running Optuna tuning (%d trials) ...", platform, optuna_trials)
        tuned_params = optuna_tune(X_train, y_train, n_trials=optuna_trials, timeout=180)
        if tuned_params:
            metrics["optuna_params"] = tuned_params

    # --- Apply feature group weights (after cross-val, before final fit) ---
    X_train_w = apply_feature_group_weights(X_train)
    logger.info("[%s] Applied feature group weights to %d columns", platform, len(X_train_w.columns))

    # --- Train on full dataset ---
    if use_stacking:
        model = stacking_model(tuned_params)
    elif balancing_strategy == "class_weight":
        # class_weight handled inside default_model (balanced is always on RF now)
        model = default_model(tuned_params)
    else:
        model = default_model(tuned_params)

    model.fit(X_train_w, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # --- Save feature names alongside the model ---
    # This JSON file lets predict_with_saved_model align new feature columns
    # to exactly what the model was trained on, preventing sklearn ValueError
    # when new features (e.g. celeb_*) are added after a model was trained.
    try:
        features_json_path = model_path.with_suffix(".features.json")
        with open(features_json_path, "w") as _fj:
            json.dump(list(X_train_w.columns), _fj)
        logger.info("[%s] Feature names saved: %s (%d features)",
                    platform, features_json_path.name, len(X_train_w.columns))
    except Exception as _fj_exc:
        logger.warning("[%s] Could not save feature names JSON: %s", platform, _fj_exc)

    # --- Fit and save confidence calibrator ---
    try:
        from ML.confidence_calibrator import RobustConfidenceCalibrator
        raw_probas = model.predict_proba(X_train_w)[:, 1].tolist()
        calibrator = RobustConfidenceCalibrator()
        calibrator.fit(raw_probas, y_train.tolist(), verbose=True)
        calibrator.save(model_path)
        logger.info("[%s] Confidence calibrator saved (method=%s)", platform, calibrator.method)
    except Exception as _cal_exc:
        logger.warning("[%s] Calibrator fit/save failed: %s", platform, _cal_exc)

    # --- Compute and save feature defaults for missing-value imputation ---
    try:
        from features.missing_value_handler import compute_and_save_defaults
        df_with_label = X_train_w.copy()
        df_with_label["label"] = y_train.values
        compute_and_save_defaults({platform: df_with_label})
        logger.info("[%s] Feature defaults saved", platform)
    except Exception as _def_exc:
        logger.warning("[%s] Feature defaults save failed: %s", platform, _def_exc)

    # --- Feature importance chart (from RF in ensemble) ---
    if charts_dir and platform:
        try:
            ensemble = model.named_steps["ensemble"]
            # VotingClassifier: estimators_ is a flat list of fitted estimators
            rf = next(
                (e for e in ensemble.estimators_ if isinstance(e, RandomForestClassifier)),
                None,
            )
            if rf is not None:
                save_feature_importance_chart(
                    charts_dir, list(X_train_w.columns), rf.feature_importances_, platform
                )
        except Exception as exc:
            logger.warning("Feature importance chart failed: %s", exc)

        if include_stats:
            try:
                save_dataset_stats_charts(charts_dir, X, y, platform)
            except Exception as exc:
                logger.warning("Stats charts failed: %s", exc)

    # Build ensemble type string for display
    parts: List[str] = ["RF", "SVM"]
    if XGBOOST_AVAILABLE: parts.append("XGB")
    if HISTGB_AVAILABLE:  parts.append("HGB")
    if LIGHTGBM_AVAILABLE: parts.append("LGBM")
    if use_stacking:
        ensemble_type = "+".join(parts) + "→LR (stacking)"
    else:
        ensemble_type = "+".join(parts) + " (voting)"
    metrics["ensemble_type"] = ensemble_type
    metrics["use_optuna"] = use_optuna and OPTUNA_AVAILABLE

    msg = f"Ensemble ({ensemble_type}) trained & saved: {model_path.name} (rows: {len(X_train)})"
    return True, msg, metrics, len(X_train)


def _load_trained_feature_names(model_path: Path) -> Optional[List[str]]:
    """
    Return the ordered feature name list the model was trained on, or None.

    Priority:
    1. <model>.features.json  — written by train_and_save() (most reliable)
    2. model.named_steps['imputer'].feature_names_in_  — sklearn >= 1.0
    3. model.feature_names_in_  — some estimator types
    """
    # 1. JSON sidecar
    json_path = model_path.with_suffix(".features.json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                names = json.load(f)
            if isinstance(names, list) and names:
                return names
        except Exception as exc:
            logger.warning("Could not read %s: %s", json_path.name, exc)

    # 2. Extract from sklearn pipeline steps
    try:
        model = joblib.load(model_path)
        for step_name in ("imputer", "preprocessor", "scaler"):
            step = model.named_steps.get(step_name)
            if step is not None and hasattr(step, "feature_names_in_"):
                names = list(step.feature_names_in_)
                if names:
                    return names
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception as exc:
        logger.warning("Could not extract feature names from model: %s", exc)

    return None


def _align_to_trained_features(
    X: pd.DataFrame,
    trained_features: Optional[List[str]],
    context: str = "",
) -> pd.DataFrame:
    """
    Align X to exactly the columns the model was trained on.

    - Adds missing columns as NaN (the pipeline's imputer will fill them).
    - Drops any extra columns the model doesn't know (e.g. new celeb_* on old models).
    - Reorders columns to match training order.

    Logs a WARNING (not ERROR) when columns are dropped — expected when predicting
    with an old model before retraining with new features.
    """
    if trained_features is None:
        return X

    # Add missing columns as NaN
    for col in trained_features:
        if col not in X.columns:
            X = X.copy()
            X[col] = np.nan

    # Drop extra columns
    extra = [c for c in X.columns if c not in trained_features]
    if extra:
        logger.warning(
            "%sDropping %d feature(s) unknown to saved model: %s%s",
            f"[{context}] " if context else "",
            len(extra),
            extra[:5],
            "..." if len(extra) > 5 else "",
        )
        X = X.drop(columns=extra)

    # Reorder to training order
    return X[trained_features]


def predict_with_saved_model(
    df: pd.DataFrame,
    model_path: Path,
    charts_dir: Path,
    tag: str,
    title: str,
    feature_builder,
    skip_charts: bool = False,
    include_shap: bool = False,
    include_anomaly: bool = False,
    confidence_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, str], List[str]]:
    warnings: List[str] = []
    df = normalize_columns(df)

    # ── Strip label columns before feature extraction ─────────────────────────
    # Labels must never reach the model as features — this is the single
    # entry point for all callers, so stripping here covers every upload path.
    # We save the stripped series so they can be re-attached to the returned
    # dataframe (the result CSV must keep them for accuracy calculation).
    _stripped_labels: dict = {}
    for _lc in LABEL_COLUMNS:
        if _lc in df.columns:
            _stripped_labels[_lc] = df[_lc].copy()
            df = df.drop(columns=[_lc])
    if _stripped_labels:
        logger.info(
            "Stripped %d label column(s) from prediction input: %s",
            len(_stripped_labels),
            list(_stripped_labels.keys()),
        )
    # ── End label stripping ───────────────────────────────────────────────────

    if not model_path.exists():
        warnings.append("No saved model found. Train the model first (Persistent Learning).")
        df["prediction"] = "Legit"
        df["confidence"] = 0.0
        counts = {"Fake": 0, "Legit": len(df)}
        for _lc, _ls in _stripped_labels.items():
            df[_lc] = _ls.values
        return df, counts, {}, warnings

    model = joblib.load(model_path)
    X, df_norm = feature_builder(df)

    # Apply the same feature group weights used at training time
    X_w = apply_feature_group_weights(X)

    # ── Feature alignment — fixes celeb_* mismatch on old models ─────────
    # Old models (.pkl files trained before celeb_* features were added) will
    # raise ValueError if we send them columns they've never seen.
    # We align X_w (and X for SHAP) to exactly what the model was trained on.
    platform_name = platform_from_path(model_path)
    trained_features = _load_trained_feature_names(model_path)
    X_w = _align_to_trained_features(X_w, trained_features, context=platform_name)
    X   = _align_to_trained_features(X,   trained_features, context=platform_name)

    # Warn the user when a large fraction of expected features are NaN-filled
    # (indicates the uploaded CSV format doesn't match what the model was trained on)
    if trained_features:
        missing_feats = [c for c in trained_features if X_w[c].isna().all()]
        if len(missing_feats) > max(1, len(trained_features) * 0.25):
            warnings.append(
                f"{len(missing_feats)} of {len(trained_features)} feature columns "
                f"are missing from this CSV and were filled with training defaults "
                f"({', '.join(missing_feats[:4])}{'…' if len(missing_feats) > 4 else ''}). "
                f"The CSV format may not match what this model was trained on — "
                f"predictions may be less accurate. Consider retraining the model on "
                f"data in this format."
            )
    # ── End feature alignment ─────────────────────────────────────────────

    preds = model.predict(X_w)
    df_norm["prediction"] = pd.Series(preds, index=df_norm.index).map(
        lambda v: "Fake" if v == 1 else "Legit"
    )

    # Confidence: probability of the predicted class, calibrated if available
    try:
        from ML.confidence_calibrator import RobustConfidenceCalibrator
        calibrator = RobustConfidenceCalibrator.load(model_path)
    except Exception:
        calibrator = None

    try:
        proba = model.predict_proba(X_w)
        raw_fake_proba = proba[:, 1]
        if calibrator is not None:
            fake_proba = np.array([calibrator.calibrate(float(p)) for p in raw_fake_proba])
        else:
            fake_proba = raw_fake_proba
        confidence = np.where(preds == 1, fake_proba, 1 - fake_proba)
        df_norm["confidence"] = (confidence * 100).round(1)
    except Exception:
        df_norm["confidence"] = None

    # Apply confidence threshold — override low-confidence predictions
    if confidence_threshold > 0:
        try:
            conf_vals = df_norm["confidence"].fillna(0)
            low_conf_mask = conf_vals < confidence_threshold
            df_norm.loc[low_conf_mask, "prediction"] = "Uncertain"
        except Exception:
            pass

    # Anomaly detection
    chart_files: Dict[str, str] = {}
    if include_anomaly:
        try:
            anomaly_labels, anomaly_raw = detect_anomalies(X)
            df_norm["anomaly"] = pd.Series(anomaly_labels, index=df_norm.index).map(
                lambda v: "Anomaly" if v == -1 else "Normal"
            )
            df_norm["anomaly_score"] = (anomaly_raw * -100).round(1)  # higher = more anomalous
        except Exception as exc:
            logger.warning("Anomaly annotation failed: %s", exc)

    # SHAP explanations
    shap_chart = None
    shap_single_chart = None
    if include_shap:
        shap_vals, base_value = compute_shap_values(model, X)
        if shap_vals is not None:
            if len(X) > 1:
                shap_chart = save_shap_chart(
                    charts_dir, shap_vals, list(X.columns), platform_from_path(model_path), tag
                )
            else:
                shap_single_chart = save_shap_single_chart(
                    charts_dir, shap_vals[0], list(X.columns), base_value,
                    platform_from_path(model_path), tag
                )

    counts = df_norm["prediction"].value_counts().to_dict()
    counts.setdefault("Fake", 0)
    counts.setdefault("Legit", 0)

    if not skip_charts:
        chart_files = save_charts(
            charts_dir=charts_dir,
            tag=tag,
            title=title,
            fake_count=int(counts["Fake"]),
            legit_count=int(counts["Legit"]),
        )

    if shap_chart:
        chart_files["shap"] = shap_chart
    if shap_single_chart:
        chart_files["shap_single"] = shap_single_chart

    return df_norm, counts, chart_files, warnings


def platform_from_path(model_path: Path) -> str:
    return model_path.stem


def get_dataset_statistics(df: pd.DataFrame, feature_builder, platform: str) -> dict:
    """Compute descriptive statistics for a dataset before training."""
    X, _ = feature_builder(df)
    stats = {
        "rows": len(df),
        "features": len(X.columns),
        "feature_names": list(X.columns),
        "missing_pct": round(X.isnull().mean().mean() * 100, 1),
        "numeric_stats": {},
    }
    for col in X.select_dtypes(include=np.number).columns:
        s = X[col].dropna()
        stats["numeric_stats"][col] = {
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "median": round(float(s.median()), 3),
        }
    return stats


def compute_accuracy_on_csv(df_pred: pd.DataFrame, label_col: str) -> Optional[dict]:
    """
    Compare batch predictions in df_pred against true labels in label_col.

    df_pred must contain both a 'prediction' column ('Fake'/'Legit'/'Uncertain')
    and the true-label column identified by label_col.  'Uncertain' predictions
    are excluded from the accuracy calculation.

    Returns a dict with accuracy/precision/recall/F1 (as percentages 0-100),
    confusion matrix, and per-class counts — or None if not enough labeled rows.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )

    if "prediction" not in df_pred.columns or label_col not in df_pred.columns:
        return None

    y_true = coerce_label_binary(df_pred[label_col]).dropna()
    if len(y_true) < 2 or y_true.nunique() < 2:
        return None

    y_pred_str = df_pred.loc[y_true.index, "prediction"]
    y_pred = y_pred_str.map({"Fake": 1, "Legit": 0})  # Uncertain → NaN → dropped

    certain = y_pred.notna()
    y_true = y_true[certain].astype(int)
    y_pred = y_pred[certain].astype(int)

    if len(y_true) < 2:
        return None

    logging.getLogger(__name__).info(
        "Computing accuracy on %d labeled rows (true_fake=%d true_legit=%d)",
        len(y_true), int((y_true == 1).sum()), int((y_true == 0).sum()),
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy":              round(acc  * 100, 1),
        "precision":             round(prec * 100, 1),
        "recall":                round(rec  * 100, 1),
        "f1":                    round(f1   * 100, 1),
        "confusion_matrix":      cm,
        "total":                 len(y_true),
        "true_fake_count":       int((y_true == 1).sum()),
        "true_legit_count":      int((y_true == 0).sum()),
        "predicted_fake_count":  int((y_pred == 1).sum()),
        "predicted_legit_count": int((y_pred == 0).sum()),
    }
