"""
Train all 10 platform models from the generated datasets.
Run: python train_all.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure we run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from ML.persistent_common import (
    train_and_save, XGBOOST_AVAILABLE, SHAP_AVAILABLE,
    LIGHTGBM_AVAILABLE, IMBLEARN_AVAILABLE, OPTUNA_AVAILABLE, HISTGB_AVAILABLE,
)
from ML.features_instagram import build_instagram_features
from ML.features_facebook import build_facebook_features
from ML.features_x import build_x_features
from ML.features_linkedin import build_linkedin_features
from ML.features_github import build_github_features
from ML.features_discord import build_discord_features
from ML.features_youtube import build_youtube_features
from ML.features_tiktok import build_tiktok_features
from ML.features_reddit import build_reddit_features
from ML.features_snapchat import build_snapchat_features

CSV_DIR   = Path("csv")
MODEL_DIR = Path("models")
CHART_DIR = Path("static/charts")

MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

PLATFORM_CONFIG = [
    ("instagram", "instagram_train.csv", build_instagram_features),
    ("facebook",  "facebook_train.csv",  build_facebook_features),
    ("x",         "x_train.csv",         build_x_features),
    ("linkedin",  "linkedin_train.csv",  build_linkedin_features),
    ("github",    "github_train.csv",    build_github_features),
    ("discord",   "discord_train.csv",   build_discord_features),
    ("youtube",   "youtube_train.csv",   build_youtube_features),
    ("tiktok",    "tiktok_train.csv",    build_tiktok_features),
    ("reddit",    "reddit_train.csv",    build_reddit_features),
    ("snapchat",  "snapchat_train.csv",  build_snapchat_features),
]

parts = ["RF", "SVM"]
if XGBOOST_AVAILABLE:  parts.append("XGB")
if HISTGB_AVAILABLE:   parts.append("HGB")
if LIGHTGBM_AVAILABLE: parts.append("LGBM")
print(f"\nEnsemble: {' + '.join(parts)}")
print(f"SHAP: {'enabled' if SHAP_AVAILABLE else 'disabled'}")
print(f"SMOTE balancing: {'enabled' if IMBLEARN_AVAILABLE else 'disabled'}")
print(f"Optuna tuning: {'available' if OPTUNA_AVAILABLE else 'disabled'}")
print("=" * 60)

results = []
for platform, csv_file, builder in PLATFORM_CONFIG:
    csv_path = CSV_DIR / csv_file
    model_path = MODEL_DIR / f"{platform}.pkl"

    if not csv_path.exists():
        print(f"[SKIP] {platform}: {csv_file} not found")
        continue

    print(f"\nTraining {platform.upper()} ...", flush=True)
    ok, msg, metrics, rows = train_and_save(
        labeled_csv_paths=[csv_path],
        model_path=model_path,
        feature_builder=builder,
        min_rows=10,
        charts_dir=CHART_DIR,
        platform=platform,
        include_stats=True,
        balancing_strategy="smote" if IMBLEARN_AVAILABLE else "none",
        use_stacking=False,   # set True for higher accuracy (slower)
        use_optuna=False,     # set True for hyperparameter tuning
    )

    status = "OK" if ok else "FAIL"
    if ok and metrics:
        acc  = metrics.get("accuracy", "?")
        prec = metrics.get("precision", "?")
        rec  = metrics.get("recall", "?")
        f1   = metrics.get("f1", "?")
        cv   = metrics.get("cv_folds", "?")
        print(f"  [{status}] rows={rows}  acc={acc}  prec={prec}  rec={rec}  F1={f1}  CV={cv}")
    else:
        print(f"  [{status}] {msg}")

    results.append((platform, ok, metrics, rows))

    # Log to training_history.json (same format as app.py)
    if ok:
        hist_path = MODEL_DIR / "training_history.json"
        hist = {}
        if hist_path.exists():
            try:
                hist = json.loads(hist_path.read_text())
            except Exception:
                pass
        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": rows,
            **(metrics or {}),
        }
        hist.setdefault(platform, []).append(record)
        hist[platform] = hist[platform][-20:]
        hist_path.write_text(json.dumps(hist, indent=2))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_ok = True
for platform, ok, metrics, rows in results:
    if ok and metrics:
        f1 = metrics.get("f1", 0)
        acc = metrics.get("accuracy", 0)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {platform:12s}  [{bar}]  acc={acc:.3f}  F1={f1:.3f}  rows={rows}")
    elif ok:
        print(f"  {platform:12s}  TRAINED (no CV metrics)")
    else:
        print(f"  {platform:12s}  FAILED")
        all_ok = False

print("\nAll models saved to models/")
print("Feature importance + SHAP charts saved to static/charts/")
