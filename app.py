import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, Response, after_this_request, stream_with_context,
)
from flask_wtf.csrf import CSRFProtect
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from ML.persistent_common import (
    train_and_save,
    predict_with_saved_model,
    normalize_columns,
    get_dataset_statistics,
    detect_anomalies,
    save_dataset_stats_charts,
    find_label_col,
    coerce_label_binary,
    compute_accuracy_on_csv,
    FEATURE_DESCRIPTIONS,
    SHAP_AVAILABLE,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    IMBLEARN_AVAILABLE,
    OPTUNA_AVAILABLE,
    HISTGB_AVAILABLE,
)
from data_generator import generate_dataset, save_dataset, get_generation_stats
from feedback import init_feedback, save_feedback, get_pending_reviews, get_all_feedback, get_feedback_stats, mark_reviewed, get_feedback_for_retraining
from drift import (
    check_and_log_drift, get_drift_summary, get_all_platforms_drift_summary,
    init_drift, PREDICTION_COUNT_CHECK, PSI_THRESHOLD_WARN, PSI_THRESHOLD_ALERT,
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

# Optional live lookup
try:
    import requests as http_requests
    HTTP_REQUESTS_AVAILABLE = True
except ImportError:
    HTTP_REQUESTS_AVAILABLE = False

# Optional PDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# Optional Excel
try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = APP_DIR / "static"
CHARTS_DIR = STATIC_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
HISTORY_PATH = MODELS_DIR / "training_history.json"
DB_PATH = APP_DIR / "instance" / "users.db"
DB_PATH.parent.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 20
MIN_TRAIN_ROWS = 10
TABLE_PREVIEW_ROWS = 500
CLEANUP_AGE_HOURS = 2
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "0") == "1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

_secret = os.environ.get("SECRET_KEY")
if not _secret:
    _secret = os.urandom(32).hex()
    logger.warning("SECRET_KEY not set — using random key.")
app.secret_key = _secret
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["WTF_CSRF_TIME_LIMIT"] = 3600
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

csrf = CSRFProtect(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

# Rate limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per minute"],
        storage_uri="memory://",
    )
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False
    logger.warning("flask-limiter not installed — rate limiting disabled.")

# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


FeedbackEntry = init_feedback(db)
DriftLog = init_drift(db)


# ---------------------------------------------------------------------------
# New DB models (Section 8)
# ---------------------------------------------------------------------------

class AvatarHash(db.Model):
    """Perceptual hash registry for profile avatars — enables stock photo detection."""
    __tablename__ = "avatar_hash"
    id = db.Column(db.Integer, primary_key=True)
    phash = db.Column(db.String(16), index=True, nullable=False)
    profile_identifier = db.Column(db.String(255), nullable=False)
    platform = db.Column(db.String(50), nullable=False)
    first_seen = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    seen_count = db.Column(db.Integer, default=1, nullable=False)


class CrossPlatformCache(db.Model):
    """24-hour TTL cache for cross-platform username existence checks."""
    __tablename__ = "cross_platform_cache"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), index=True, nullable=False)
    platform = db.Column(db.String(50), nullable=False)
    exists = db.Column(db.Boolean, default=False)
    profile_data_json = db.Column(db.Text)
    checked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = db.Column(db.DateTime)


class PredictionLog(db.Model):
    """Audit log of all single-profile predictions with analysis depth tracking."""
    __tablename__ = "prediction_log"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), index=True)
    platform = db.Column(db.String(50), index=True, nullable=False)
    prediction = db.Column(db.String(10))   # 'Fake' or 'Legit'
    confidence = db.Column(db.Float)
    analysis_depth = db.Column(db.String(10))  # 'quick' or 'deep'
    features_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)


with app.app_context():
    db.create_all()

# Wire photo_analysis DB session factory (avoids circular import)
try:
    from features.photo_analysis import set_db_session_factory
    set_db_session_factory(db.session)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Platform registry
# ---------------------------------------------------------------------------

PLATFORMS = [
    "instagram", "facebook", "x", "linkedin", "github", "discord",
    "youtube", "tiktok", "reddit", "snapchat",
]

MODEL_PATHS = {p: MODELS_DIR / f"{p}.pkl" for p in PLATFORMS}

FEATURE_BUILDERS = {
    "instagram": build_instagram_features,
    "facebook":  build_facebook_features,
    "x":         build_x_features,
    "linkedin":  build_linkedin_features,
    "github":    build_github_features,
    "discord":   build_discord_features,
    "youtube":   build_youtube_features,
    "tiktok":    build_tiktok_features,
    "reddit":    build_reddit_features,
    "snapchat":  build_snapchat_features,
}

EXPECTED_COLUMNS: Dict[str, List[str]] = {
    "instagram": ["username", "followers", "following", "posts", "bio", "is_verified"],
    "facebook":  ["name", "friends", "followers", "posts", "bio", "is_verified"],
    "x":         ["username", "followers", "following", "tweets", "bio", "is_verified"],
    "linkedin":  ["name", "connections", "followers", "headline", "about"],
    "github":    ["username", "followers", "following", "public_repos", "public_gists", "account_age_days", "bio"],
    "discord": [
        "timestamp_utc", "guild_id", "guild_name", "user_id", "username",
        "created_at_utc", "joined_at_utc", "account_age_days", "has_avatar",
        "suspicion_score", "reasons", "action_taken",
    ],
    "youtube":  ["channel_name", "subscribers", "videos", "about"],
    "tiktok":   ["username", "followers", "following", "videos", "bio"],
    "reddit":   ["username", "karma", "created_at", "about"],
    "snapchat": ["username", "score", "bio"],
}

MANUAL_FIELDS: Dict[str, List[tuple]] = {
    "instagram": [
        ("username", "text"), ("followers", "number"), ("following", "number"),
        ("posts", "number"), ("bio", "text"), ("is_verified", "checkbox"),
    ],
    "facebook": [
        ("name", "text"), ("friends", "number"), ("followers", "number"),
        ("posts", "number"), ("bio", "text"), ("is_verified", "checkbox"),
    ],
    "x": [
        ("username", "text"), ("followers", "number"), ("following", "number"),
        ("tweets", "number"), ("bio", "text"), ("is_verified", "checkbox"),
    ],
    "linkedin": [
        ("name", "text"), ("connections", "number"), ("followers", "number"),
        ("headline", "text"), ("about", "text"),
    ],
    "github": [
        ("username", "text"), ("followers", "number"), ("following", "number"),
        ("public_repos", "number"), ("public_gists", "number"),
        ("account_age_days", "number"), ("bio", "text"),
    ],
    "discord": [
        ("username", "text"), ("account_age_days", "number"),
        ("has_avatar", "checkbox"), ("suspicion_score", "number"),
        ("created_at_utc", "text"), ("joined_at_utc", "text"),
    ],
    "youtube": [
        ("channel_name", "text"), ("subscribers", "number"),
        ("videos", "number"), ("about", "text"),
    ],
    "tiktok": [
        ("username", "text"), ("followers", "number"), ("following", "number"),
        ("videos", "number"), ("bio", "text"),
    ],
    "reddit": [
        ("username", "text"), ("karma", "number"),
        ("created_at", "text"), ("about", "text"),
    ],
    "snapchat": [
        ("username", "text"), ("score", "number"), ("bio", "text"),
    ],
}

# Platform icons (Bootstrap Icons)
PLATFORM_ICONS = {
    "instagram": "bi-instagram",
    "facebook": "bi-facebook",
    "x": "bi-twitter-x",
    "linkedin": "bi-linkedin",
    "github": "bi-github",
    "discord": "bi-discord",
    "youtube": "bi-youtube",
    "tiktok": "bi-tiktok",
    "reddit": "bi-reddit",
    "snapchat": "bi-camera-fill",
}

# Platforms that support live API lookup (extended in Section 1)
LIVE_LOOKUP_PLATFORMS = {
    "github", "reddit", "twitter", "x", "instagram",
    "linkedin", "youtube", "discord", "tiktok",
}

# ---------------------------------------------------------------------------
# Task progress tracking (in-memory queues for SSE)
# ---------------------------------------------------------------------------

_progress_queues: Dict[str, "queue.Queue"] = {}
import queue as _queue_module


def create_progress_task() -> str:
    import os
    task_id = os.urandom(8).hex()
    _progress_queues[task_id] = _queue_module.Queue()
    return task_id


def push_progress(task_id: str, pct: int, message: str, done: bool = False, **extra):
    q = _progress_queues.get(task_id)
    if q:
        q.put({"progress": pct, "message": message, "done": done, **extra})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_in_upload_dir(filename: str) -> Path | None:
    resolved = (UPLOAD_DIR / filename).resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve()) + os.sep) and \
            resolved != UPLOAD_DIR.resolve():
        return None
    return resolved


def _validate_expected_columns(platform: str, df: pd.DataFrame) -> List[str]:
    expected = [c.lower() for c in EXPECTED_COLUMNS.get(platform, [])]
    return [c for c in expected if c not in df.columns]


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete temp file %s", path)


def cleanup_old_files(max_age_hours: float = CLEANUP_AGE_HOURS) -> None:
    cutoff = time.time() - max_age_hours * 3600
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            _cleanup(f)
    for f in CHARTS_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            if f.name.startswith(("bar_", "pie_", "shap_", "stats_")):
                _cleanup(f)


def get_model_status(platform: str) -> dict:
    try:
        mtime = MODEL_PATHS[platform].stat().st_mtime
        trained_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        return {"exists": True, "trained_at": trained_at}
    except FileNotFoundError:
        return {"exists": False, "trained_at": None}


def load_history() -> dict:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    return {}


def log_training_event(platform: str, rows: int, metrics: dict) -> None:
    history = load_history()
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": rows,
        **metrics,
    }
    history.setdefault(platform, []).append(record)
    history[platform] = history[platform][-20:]
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def _safe_confidence(val) -> float | None:
    if val is not None and str(val) != "nan":
        return float(val)
    return None


def _build_single_row_df(platform: str, form) -> pd.DataFrame:
    row: dict = {}
    for field, ftype in MANUAL_FIELDS[platform]:
        if ftype == "checkbox":
            row[field] = 1 if form.get(field) else 0
        elif ftype == "number":
            raw = form.get(field, "") or "0"
            try:
                row[field] = float(raw)
            except ValueError:
                row[field] = 0.0
        else:
            row[field] = form.get(field, "")
    return pd.DataFrame([row])


def _auth_required(fn):
    """Decorator — enforces login only if REQUIRE_AUTH is enabled."""
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if REQUIRE_AUTH:
            return login_required(fn)(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

cleanup_old_files()

# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(request.args.get("next") or url_for("index"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    # Only allow registration if no users exist OR current user is admin
    user_count = User.query.count()
    if user_count > 0 and (not current_user.is_authenticated or not current_user.is_admin):
        flash("Registration is restricted to admins.", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password required.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
        else:
            user = User(username=username, is_admin=(user_count == 0))
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash(f"User '{username}' created{'  (admin)' if user_count == 0 else ''}.", "success")
            return redirect(url_for("login"))
    return render_template("register.html", first_user=(User.query.count() == 0))


# ---------------------------------------------------------------------------
# Main routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    model_status = {p: get_model_status(p) for p in PLATFORMS}
    history = load_history()
    platform_acc = {}
    for p in PLATFORMS:
        entries = history.get(p, [])
        if entries:
            last = entries[-1]
            platform_acc[p] = last.get("accuracy")
    return render_template(
        "index.html",
        platforms=PLATFORMS,
        model_status=model_status,
        platform_acc=platform_acc,
        platform_icons=PLATFORM_ICONS,
        live_lookup_platforms=LIVE_LOOKUP_PLATFORMS,
        shap_available=SHAP_AVAILABLE,
        xgboost_available=XGBOOST_AVAILABLE,
    )


@app.route("/train/<platform>", methods=["GET", "POST"])
@_auth_required
def train_platform(platform: str):
    platform = platform.lower()

    if platform not in FEATURE_BUILDERS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("index"))

    if request.method == "POST":
        files = request.files.getlist("file")
        if not files or all(f.filename == "" for f in files):
            flash("No file(s) uploaded.", "danger")
            return redirect(request.url)

        save_paths = []
        for f in files:
            if not f.filename or not f.filename.lower().endswith(".csv"):
                flash(f"Skipped '{f.filename}' — not a .csv file.", "warning")
                continue
            sp = UPLOAD_DIR / f"{platform}_train_{os.urandom(8).hex()}.csv"
            f.save(sp)
            save_paths.append(sp)

        if not save_paths:
            flash("No valid CSV files were uploaded.", "danger")
            return redirect(request.url)

        try:
            ok, msg, metrics, rows_used = train_and_save(
                labeled_csv_paths=save_paths,
                model_path=MODEL_PATHS[platform],
                feature_builder=FEATURE_BUILDERS[platform],
                min_rows=MIN_TRAIN_ROWS,
                charts_dir=CHARTS_DIR,
                platform=platform,
                include_stats=True,
                use_stacking=request.form.get("use_stacking") == "1",
                balancing_strategy=request.form.get("balancing_strategy", "none"),
                use_temporal_cv=request.form.get("use_temporal_cv") == "1",
                use_feature_selection=request.form.get("use_feature_selection") == "1",
                use_optuna=request.form.get("use_optuna") == "1",
                optuna_trials=int(request.form.get("optuna_trials", "40") or "40"),
            )
        finally:
            for sp in save_paths:
                _cleanup(sp)

        flash(msg, "success" if ok else "danger")

        if ok:
            log_training_event(platform, rows_used, metrics)
            if metrics:
                ens = metrics.get("ensemble_type", "?")
                optuna_note = " [Optuna-tuned]" if metrics.get("use_optuna") else ""
                flash(
                    f"CV metrics — Accuracy: {metrics.get('accuracy','?')}  "
                    f"Precision: {metrics.get('precision','?')}  "
                    f"Recall: {metrics.get('recall','?')}  "
                    f"F1: {metrics.get('f1','?')}  "
                    f"Ensemble: {ens}{optuna_note}",
                    "info",
                )
        return redirect(url_for("index"))

    return render_template("train_generic.html", platform=platform,
                           shap_available=SHAP_AVAILABLE,
                           xgboost_available=XGBOOST_AVAILABLE,
                           lightgbm_available=LIGHTGBM_AVAILABLE,
                           imblearn_available=IMBLEARN_AVAILABLE,
                           optuna_available=OPTUNA_AVAILABLE,
                           histgb_available=HISTGB_AVAILABLE)


@app.route("/upload/<platform>", methods=["GET", "POST"])
def upload(platform: str):
    platform = platform.lower()

    if platform not in PLATFORMS:
        flash("Unknown platform selected.", "danger")
        return redirect(url_for("index"))

    expected = EXPECTED_COLUMNS.get(platform, [])

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded.", "danger")
            return redirect(request.url)

        f = request.files["file"]
        if not f.filename or not f.filename.lower().endswith(".csv"):
            flash("Please upload a .csv file.", "danger")
            return redirect(request.url)

        save_path = UPLOAD_DIR / f"{platform}_{os.urandom(8).hex()}.csv"
        f.save(save_path)

        try:
            df = pd.read_csv(save_path)
        except Exception:
            _cleanup(save_path)
            flash("Could not read the uploaded file. Make sure it is a valid CSV.", "danger")
            return redirect(request.url)
        finally:
            _cleanup(save_path)

        df = normalize_columns(df)
        missing = _validate_expected_columns(platform, df)
        if missing:
            flash(f"Missing expected columns: {', '.join(missing)}", "warning")

        tag = os.urandom(8).hex()
        include_shap = request.form.get("include_shap") == "1"
        include_anomaly = request.form.get("include_anomaly") == "1"
        threshold = float(request.form.get("threshold", "0") or "0")

        df_pred, counts, chart_files, warnings = predict_with_saved_model(
            df=df,
            model_path=MODEL_PATHS[platform],
            charts_dir=CHARTS_DIR,
            tag=tag,
            title=f"{platform.capitalize()} Fake vs Legit",
            feature_builder=FEATURE_BUILDERS[platform],
            include_shap=include_shap,
            include_anomaly=include_anomaly,
            confidence_threshold=threshold,
        )

        for w in warnings:
            flash(w, "warning")

        result_path = UPLOAD_DIR / f"result_{platform}_{os.urandom(8).hex()}.csv"
        df_pred.to_csv(result_path, index=False)

        return redirect(url_for(
            "results",
            platform=platform,
            result_file=result_path.name,
            bar=chart_files.get("bar", ""),
            pie=chart_files.get("pie", ""),
            shap=chart_files.get("shap", ""),
        ))

    return render_template("upload.html", platform=platform, expected=expected,
                           shap_available=SHAP_AVAILABLE)


@app.route("/results/<platform>")
def results(platform: str):
    platform = platform.lower()

    result_file = request.args.get("result_file", "")
    if not result_file:
        flash("Result not found.", "danger")
        return redirect(url_for("index"))

    result_path = _safe_in_upload_dir(result_file)
    if result_path is None or not result_path.exists():
        flash("Result not found.", "danger")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(result_path)
    except Exception:
        _cleanup(result_path)
        flash("Could not read results.", "danger")
        return redirect(url_for("index"))

    df = normalize_columns(df)

    counts = df["prediction"].value_counts().to_dict()
    fake_count = int(counts.get("Fake", 0))
    legit_count = int(counts.get("Legit", 0))
    uncertain_count = int(counts.get("Uncertain", 0))

    # Compute accuracy metrics when the uploaded CSV contained true labels
    accuracy_metrics = None
    label_col = find_label_col(df)
    if label_col:
        try:
            accuracy_metrics = compute_accuracy_on_csv(df, label_col)
        except Exception as _acc_exc:
            logger.warning("Accuracy computation failed: %s", _acc_exc)

    rows = df.head(TABLE_PREVIEW_ROWS).to_dict(orient="records")
    columns = list(df.columns)
    total_rows = len(df)

    bar = request.args.get("bar", "")
    pie = request.args.get("pie", "")
    shap_chart = request.args.get("shap", "")

    importance_chart = f"charts/{platform}_importance.png"
    importance_path = STATIC_DIR / importance_chart
    importance_chart = importance_chart if importance_path.exists() else ""

    shap_available = SHAP_AVAILABLE and not shap_chart

    # Network graph data (top 50 rows as nodes)
    network_data = _build_network_data(df.head(50), platform)

    # Stats charts for this platform
    stats_dist = f"charts/stats_{platform}_distributions.png"
    stats_dist_path = STATIC_DIR / stats_dist
    stats_dist_chart = stats_dist if stats_dist_path.exists() else ""

    return render_template(
        "results.html",
        platform=platform,
        fake_count=fake_count,
        legit_count=legit_count,
        uncertain_count=uncertain_count,
        accuracy_metrics=accuracy_metrics,
        columns=columns,
        rows=rows,
        total_rows=total_rows,
        bar=bar,
        pie=pie,
        shap_chart=shap_chart,
        importance_chart=importance_chart,
        stats_dist_chart=stats_dist_chart,
        result_file=result_file,
        shap_available=shap_available,
        network_data=json.dumps(network_data),
        feature_descriptions=FEATURE_DESCRIPTIONS,
    )


def _build_network_data(df: pd.DataFrame, platform: str) -> dict:
    """Build vis.js-compatible nodes/edges for the profile network graph."""
    nodes = []
    edges = []
    has_prediction = "prediction" in df.columns
    has_followers = any(c in df.columns for c in ["followers", "subscribers", "karma", "score"])
    follower_col = next((c for c in ["followers", "subscribers", "karma", "score"] if c in df.columns), None)

    for i, row in df.iterrows():
        pred = row.get("prediction", "Legit")
        conf = row.get("confidence", 50)
        label = row.get("username", row.get("name", row.get("channel_name", f"Profile {i}")))
        color = "#dc3545" if pred == "Fake" else ("#ffc107" if pred == "Uncertain" else "#198754")
        size = 15
        if follower_col and pd.notna(row.get(follower_col)):
            try:
                v = float(row[follower_col])
                size = max(8, min(40, 8 + np.log1p(v) * 2))
            except Exception:
                pass
        nodes.append({
            "id": int(i),
            "label": str(label)[:20],
            "color": color,
            "size": round(size, 1),
            "title": f"{label}<br>Prediction: {pred}<br>Confidence: {conf}%",
        })

    # Add edges between "similar" profiles (same prediction cluster, limited edges)
    fake_ids = [n["id"] for n in nodes if n["color"] == "#dc3545"][:15]
    legit_ids = [n["id"] for n in nodes if n["color"] == "#198754"][:15]

    for group in [fake_ids, legit_ids]:
        for j in range(len(group) - 1):
            edges.append({"from": group[j], "to": group[j + 1], "color": {"opacity": 0.3}})

    return {"nodes": nodes, "edges": edges}


@app.route("/download/<filename>")
def download_result(filename: str):
    path = _safe_in_upload_dir(filename)
    if path is None:
        flash("Download file not found.", "warning")
        return redirect(url_for("index"))

    try:
        @after_this_request
        def _delete(response):
            _cleanup(path)
            return response
        return send_file(path, mimetype="text/csv", as_attachment=True, download_name=filename)
    except FileNotFoundError:
        flash("Download file not found or already downloaded.", "warning")
        return redirect(url_for("index"))


@app.route("/template/<platform>")
def download_template(platform: str):
    platform = platform.lower()
    if platform not in EXPECTED_COLUMNS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("index"))
    cols = EXPECTED_COLUMNS[platform]
    csv_content = ",".join(cols) + "\n"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={platform}_template.csv"},
    )


@app.route("/predict/<platform>", methods=["GET", "POST"])
def manual_predict(platform: str):
    platform = platform.lower()
    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("index"))

    fields = MANUAL_FIELDS[platform]
    result = None
    confidence = None
    confidence_lbl = None
    analysis_depth = None
    form_data: dict = {}
    shap_chart = None
    cross_platform = None
    username_features = None
    bio_features = None
    photo_features = None
    confidence_explanation = None
    override_applied = False
    override_reason = None
    is_known_figure = False

    # Check API key configuration for this platform
    try:
        from live_enrichment import PLATFORM_ENV_VARS
        env_var = PLATFORM_ENV_VARS.get(platform, "")
        api_key_configured = bool(os.environ.get(env_var)) if env_var else True
    except Exception:
        api_key_configured = False

    data_coverage = None
    fields_filled = None
    fields_total = None

    if request.method == "POST":
        from features.manual_feature_mapper import ManualFeatureMapper
        from features.missing_value_handler import handle_missing_features

        mapper = ManualFeatureMapper()
        mapped = mapper.map_form_to_features(request.form, platform)

        # Coverage: fields the user actually filled in (non-empty / non-None)
        # vs. the total set of mappable feature keys for this platform.
        countable = {
            k: v for k, v in mapped.items()
            if not k.endswith("_missing") and not k.endswith("_features_json")
        }
        fields_total = len(countable)
        fields_filled = sum(
            1 for v in countable.values()
            if v is not None and v != "" and v != 0
        )
        data_coverage = (fields_filled / fields_total) if fields_total else 0.0

        # Fill missing optional features with platform medians + add *_missing flags.
        filled = handle_missing_features(mapped, platform)

        # Build the single-row DataFrame the feature_builder can consume.
        # Numeric values stay numeric; strings stay strings; the builder picks
        # whichever columns it knows about.
        df = pd.DataFrame([filled])
        form_data = request.form.to_dict()
        tag = os.urandom(8).hex()

        df_pred, _, chart_files, warnings = predict_with_saved_model(
            df=df,
            model_path=MODEL_PATHS[platform],
            charts_dir=CHARTS_DIR,
            tag=tag,
            title="",
            feature_builder=FEATURE_BUILDERS[platform],
            skip_charts=False,
            include_shap=SHAP_AVAILABLE,
        )

        for w in warnings:
            flash(w, "warning")

        result = df_pred["prediction"].iloc[0]
        raw_conf = df_pred["confidence"].iloc[0] if "confidence" in df_pred.columns else None
        confidence = _safe_confidence(raw_conf)
        shap_chart = chart_files.get("shap_single", "")
        analysis_depth = "manual_comprehensive"

        # ── Celebrity / public figure safeguards ──────────────────────────────
        try:
            from prediction.celebrity_safeguard import (
                apply_celebrity_safeguard,
                apply_bot_safeguard,
                check_public_figure_allowlist,
            )
            uname_for_check = (
                mapped.get("username") or mapped.get("name")
                or mapped.get("channel_name") or ""
            )

            # 1. Allowlist check — highest priority
            is_known_figure, kf_conf = check_public_figure_allowlist(
                uname_for_check, platform
            )
            if is_known_figure and kf_conf is not None:
                result = "Legit"
                confidence = kf_conf
                override_applied = True
                override_reason = (
                    "This account is on the known public figures list for this platform."
                )
            else:
                # 2. Celebrity safeguard
                result, confidence, override_applied, override_reason = (
                    apply_celebrity_safeguard(filled, result, confidence or 50.0)
                )
                # 3. Bot safeguard (only if celebrity safeguard didn't fire)
                if not override_applied:
                    result, confidence, override_applied, override_reason = (
                        apply_bot_safeguard(filled, result, confidence or 50.0)
                    )
        except Exception as _sg_exc:
            logger.debug("Safeguard layer failed: %s", _sg_exc)

        if confidence is not None:
            try:
                from prediction.tiered_predictor import confidence_label as _conf_label
                confidence_lbl = _conf_label(confidence)
            except Exception:
                pass

        # Surface the rich forensic + photo signals computed by the mapper into
        # the result template (it already has display blocks for these).
        username_features = {
            k: v for k, v in mapped.items()
            if k.startswith("uname_") and v is not None
        } or None
        bio_features = {
            k: v for k, v in mapped.items()
            if k.startswith("bio_") and v is not None
        } or None
        photo_features = {
            k: v for k, v in mapped.items()
            if k.startswith("photo_") and v is not None
        } or None
        if photo_features and "photo_available" not in photo_features:
            photo_features["photo_available"] = 1

        # Confidence explanation
        if result is not None and confidence is not None:
            try:
                from features.confidence_explainer import (
                    generate_confidence_explanation,
                    adjust_confidence_for_display,
                )
                confidence = adjust_confidence_for_display(confidence, result, filled)
                confidence_explanation = generate_confidence_explanation(
                    result, confidence, filled, platform=platform
                )
                if confidence_explanation is not None:
                    confidence_explanation["data_coverage_pct"] = round(data_coverage * 100, 1)
            except Exception:
                pass

        # Audit log — override info stored in features_json
        try:
            uname = (
                mapped.get("username") or mapped.get("name")
                or mapped.get("channel_name") or ""
            )
            log_payload = {
                k: v for k, v in mapped.items()
                if not isinstance(v, (dict, list))
            }
            if override_applied:
                log_payload["_override_applied"] = True
                log_payload["_override_reason"] = override_reason or ""
                log_payload["_is_known_figure"] = is_known_figure
            db.session.add(PredictionLog(
                username=uname[:255],
                platform=platform,
                prediction=result,
                confidence=confidence,
                analysis_depth="manual_comprehensive",
                features_json=json.dumps(log_payload, default=str)[:65535],
            ))
            db.session.commit()
        except Exception as exc:
            logger.debug("PredictionLog write failed: %s", exc)
            db.session.rollback()

    return render_template(
        "manual_predict.html",
        platform=platform,
        fields=fields,
        result=result,
        confidence=confidence,
        confidence_label=confidence_lbl,
        analysis_depth=analysis_depth,
        form_data=form_data,
        shap_chart=shap_chart,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        shap_available=SHAP_AVAILABLE,
        live_lookup=platform in LIVE_LOOKUP_PLATFORMS,
        api_key_configured=api_key_configured,
        cross_platform=cross_platform,
        username_features=username_features,
        bio_features=bio_features,
        photo_features=photo_features,
        confidence_explanation=confidence_explanation,
        data_coverage=data_coverage,
        fields_filled=fields_filled,
        fields_total=fields_total,
        override_applied=override_applied,
        override_reason=override_reason,
        is_known_figure=is_known_figure,
    )


@app.route("/history")
def history():
    hist = load_history()
    return render_template("history.html", history=hist, platforms=PLATFORMS)


# ---------------------------------------------------------------------------
# Platform comparison dashboard
# ---------------------------------------------------------------------------

@app.route("/comparison")
def comparison():
    hist = load_history()
    comparison_data = []
    for p in PLATFORMS:
        status = get_model_status(p)
        entries = hist.get(p, [])
        latest = entries[-1] if entries else {}
        comparison_data.append({
            "platform": p,
            "trained": status["exists"],
            "trained_at": status.get("trained_at"),
            "accuracy": latest.get("accuracy"),
            "precision": latest.get("precision"),
            "recall": latest.get("recall"),
            "f1": latest.get("f1"),
            "rows": latest.get("rows"),
            "icon": PLATFORM_ICONS.get(p, "bi-question"),
        })

    # Generate comparison chart
    trained = [d for d in comparison_data if d["accuracy"] is not None]
    comp_chart = ""
    if trained:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        platforms = [d["platform"].capitalize() for d in trained]
        metrics_names = ["accuracy", "precision", "recall", "f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
        colors = ["#0d6efd", "#198754", "#ffc107", "#dc3545"]

        x = np.arange(len(platforms))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(8, len(platforms) * 1.2), 5))
        for i, (metric, label, color) in enumerate(zip(metrics_names, metric_labels, colors)):
            vals = [d.get(metric, 0) or 0 for d in trained]
            bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        ax.set_xlabel("Platform")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison Across Platforms", fontweight="bold", fontsize=13)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(platforms, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        comp_chart_path = CHARTS_DIR / "comparison_metrics.png"
        plt.savefig(comp_chart_path, dpi=120, bbox_inches="tight")
        plt.close()
        comp_chart = "charts/comparison_metrics.png"

    return render_template(
        "comparison.html",
        comparison_data=comparison_data,
        comp_chart=comp_chart,
        platforms=PLATFORMS,
    )


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

@app.route("/stats/<platform>", methods=["GET", "POST"])
def dataset_stats(platform: str):
    platform = platform.lower()
    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("index"))

    stats = None
    stats_charts = {}

    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.", "danger")
            return redirect(request.url)

        save_path = UPLOAD_DIR / f"stats_{platform}_{os.urandom(8).hex()}.csv"
        f.save(save_path)
        try:
            df = pd.read_csv(save_path)
            df = normalize_columns(df)
            stats = get_dataset_statistics(df, FEATURE_BUILDERS[platform], platform)

            from ML.persistent_common import find_label_col, coerce_label_binary
            label_col = find_label_col(df)
            if label_col:
                y = coerce_label_binary(df[label_col]).dropna().astype(int)
                X, _ = FEATURE_BUILDERS[platform](df)
                stats["class_counts"] = y.value_counts().to_dict()
                stats_charts = save_dataset_stats_charts(CHARTS_DIR, X, y, platform)
            else:
                stats["class_counts"] = {}
        except Exception as exc:
            flash(f"Error analyzing file: {exc}", "danger")
        finally:
            _cleanup(save_path)

    return render_template(
        "stats.html",
        platform=platform,
        stats=stats,
        stats_charts=stats_charts,
        feature_descriptions=FEATURE_DESCRIPTIONS,
    )


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

@app.route("/benchmark", methods=["GET", "POST"])
def benchmark():
    hist = load_history()
    our_metrics = {}
    for p in PLATFORMS:
        entries = hist.get(p, [])
        if entries:
            our_metrics[p] = entries[-1]

    baseline_result = None
    bench_chart = ""

    if request.method == "POST":
        platform = request.form.get("platform", "").lower()
        baseline_name = request.form.get("baseline_name", "Published Baseline")
        try:
            baseline = {
                "accuracy": float(request.form.get("acc", 0) or 0),
                "precision": float(request.form.get("prec", 0) or 0),
                "recall": float(request.form.get("rec", 0) or 0),
                "f1": float(request.form.get("f1", 0) or 0),
            }
        except ValueError:
            flash("Please enter valid numeric metric values.", "danger")
            return redirect(request.url)

        our = our_metrics.get(platform, {})

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics_names = ["accuracy", "precision", "recall", "f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
        our_vals = [our.get(m, 0) or 0 for m in metrics_names]
        base_vals = [baseline.get(m, 0) for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - width / 2, our_vals, width, label="Our Ensemble Model",
                    color="#0d6efd", alpha=0.85)
        b2 = ax.bar(x + width / 2, base_vals, width, label=baseline_name,
                    color="#6c757d", alpha=0.85)
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"{platform.capitalize()} — Model vs Baseline Comparison", fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        bench_chart_path = CHARTS_DIR / f"benchmark_{platform}.png"
        plt.savefig(bench_chart_path, dpi=120)
        plt.close()
        bench_chart = f"charts/benchmark_{platform}.png"

        baseline_result = {
            "platform": platform,
            "our": our,
            "baseline": baseline,
            "baseline_name": baseline_name,
        }

    return render_template(
        "benchmark.html",
        platforms=PLATFORMS,
        our_metrics=our_metrics,
        baseline_result=baseline_result,
        bench_chart=bench_chart,
    )


# ---------------------------------------------------------------------------
# A/B Model Testing
# ---------------------------------------------------------------------------

@app.route("/abtest/<platform>", methods=["GET", "POST"])
def abtest(platform: str):
    platform = platform.lower()
    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("index"))

    ab_results = None

    if request.method == "POST":
        model_a_file = request.files.get("model_a")
        model_b_file = request.files.get("model_b")
        data_file = request.files.get("data")

        if not data_file or not data_file.filename.lower().endswith(".csv"):
            flash("Please upload a CSV data file.", "danger")
            return redirect(request.url)

        model_a_path = None
        model_b_path = None
        data_path = UPLOAD_DIR / f"abtest_{os.urandom(8).hex()}.csv"

        try:
            data_file.save(data_path)
            df = pd.read_csv(data_path)
            df = normalize_columns(df)

            # Use saved model if no file uploaded for A
            if model_a_file and model_a_file.filename:
                model_a_path = UPLOAD_DIR / f"model_a_{os.urandom(8).hex()}.pkl"
                model_a_file.save(model_a_path)
            else:
                model_a_path = MODEL_PATHS.get(platform)

            if model_b_file and model_b_file.filename:
                model_b_path = UPLOAD_DIR / f"model_b_{os.urandom(8).hex()}.pkl"
                model_b_file.save(model_b_path)

            if not model_a_path or not model_a_path.exists():
                flash(f"No model available for {platform}. Train first.", "danger")
                return redirect(request.url)

            df_a, counts_a, _, warnings_a = predict_with_saved_model(
                df=df.copy(), model_path=model_a_path,
                charts_dir=CHARTS_DIR, tag="", title="",
                feature_builder=FEATURE_BUILDERS[platform], skip_charts=True,
            )

            if model_b_path and model_b_path.exists():
                df_b, counts_b, _, warnings_b = predict_with_saved_model(
                    df=df.copy(), model_path=model_b_path,
                    charts_dir=CHARTS_DIR, tag="", title="",
                    feature_builder=FEATURE_BUILDERS[platform], skip_charts=True,
                )
            else:
                df_b, counts_b = None, None
                warnings_b = ["No Model B uploaded — showing Model A only."]

            for w in warnings_a + (warnings_b or []):
                flash(w, "warning")

            # Agreement rate
            agreement = None
            if df_b is not None:
                agree = (df_a["prediction"] == df_b["prediction"]).mean() * 100
                agreement = round(float(agree), 1)

            # Build comparison table
            compare_rows = []
            for i in range(min(200, len(df_a))):
                name = str(df.iloc[i].get("username", df.iloc[i].get("name", f"#{i}")))
                row = {
                    "profile": name,
                    "pred_a": df_a.iloc[i].get("prediction", "?"),
                    "conf_a": _safe_confidence(df_a.iloc[i].get("confidence")),
                }
                if df_b is not None:
                    row["pred_b"] = df_b.iloc[i].get("prediction", "?")
                    row["conf_b"] = _safe_confidence(df_b.iloc[i].get("confidence"))
                    row["agree"] = row["pred_a"] == row["pred_b"]
                compare_rows.append(row)

            ab_results = {
                "counts_a": counts_a,
                "counts_b": counts_b,
                "agreement": agreement,
                "rows": compare_rows,
            }

        except Exception as exc:
            flash(f"Error during A/B test: {exc}", "danger")
            logger.exception("A/B test error")
        finally:
            _cleanup(data_path)
            if model_a_path and model_a_path != MODEL_PATHS.get(platform):
                _cleanup(model_a_path)
            if model_b_path:
                _cleanup(model_b_path)

    return render_template(
        "abtest.html",
        platform=platform,
        ab_results=ab_results,
        model_exists=MODEL_PATHS[platform].exists(),
    )


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

@app.route("/anomaly/<platform>", methods=["GET", "POST"])
def anomaly_detection(platform: str):
    platform = platform.lower()
    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("index"))

    anomaly_rows = None
    counts = {}

    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.", "danger")
            return redirect(request.url)

        save_path = UPLOAD_DIR / f"anomaly_{platform}_{os.urandom(8).hex()}.csv"
        f.save(save_path)
        try:
            df = pd.read_csv(save_path)
            df = normalize_columns(df)
            X, df_norm = FEATURE_BUILDERS[platform](df)
            labels, raw_scores = detect_anomalies(X)
            df_norm["anomaly_flag"] = pd.Series(labels).map(lambda v: "Anomaly" if v == -1 else "Normal")
            df_norm["anomaly_score"] = (raw_scores * -100).round(1)
            df_norm = df_norm.sort_values("anomaly_score", ascending=False)
            counts = df_norm["anomaly_flag"].value_counts().to_dict()
            anomaly_rows = df_norm.head(200).to_dict(orient="records")
        except Exception as exc:
            flash(f"Error: {exc}", "danger")
        finally:
            _cleanup(save_path)

    return render_template(
        "anomaly.html",
        platform=platform,
        anomaly_rows=anomaly_rows,
        counts=counts,
        columns=list(anomaly_rows[0].keys()) if anomaly_rows else [],
    )


# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------

@app.route("/export/pdf/<platform>/<result_file>")
def export_pdf(platform: str, result_file: str):
    if not FPDF_AVAILABLE:
        flash("PDF export requires fpdf2. Install it: pip install fpdf2", "danger")
        return redirect(url_for("index"))

    result_path = _safe_in_upload_dir(result_file)
    if result_path is None or not result_path.exists():
        flash("Result file not found.", "warning")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(result_path)
        df = normalize_columns(df)
    except Exception:
        flash("Could not read result file.", "danger")
        return redirect(url_for("index"))

    counts = df["prediction"].value_counts().to_dict()
    fake_count = counts.get("Fake", 0)
    legit_count = counts.get("Legit", 0)
    total = len(df)
    fake_rate = round(fake_count / total * 100, 1) if total > 0 else 0

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Fake Profile Detection Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Platform: {platform.capitalize()}   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(5)

    # Summary metrics
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_fill_color(248, 249, 250)
    pdf.cell(60, 8, "Total Profiles Analyzed:", fill=True)
    pdf.cell(0, 8, str(total), ln=True, fill=True)
    pdf.cell(60, 8, "Fake Profiles Detected:", fill=True)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(0, 8, f"{fake_count} ({fake_rate}%)", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(60, 8, "Legitimate Profiles:", fill=True)
    pdf.set_text_color(25, 135, 84)
    pdf.cell(0, 8, str(legit_count), ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Charts
    bar_path = CHARTS_DIR / f"bar_{result_file.replace('result_', '').replace('.csv', '')}.png"
    pie_path = CHARTS_DIR / f"pie_{result_file.replace('result_', '').replace('.csv', '')}.png"
    importance_path = STATIC_DIR / f"charts/{platform}_importance.png"

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Distribution Charts", ln=True)
    for chart_path, label in [(bar_path, "Bar Chart"), (pie_path, "Distribution")]:
        if chart_path.exists():
            try:
                pdf.image(str(chart_path), w=85)
                pdf.ln(2)
            except Exception:
                pass

    if importance_path.exists():
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Feature Importance", ln=True)
        try:
            pdf.image(str(importance_path), w=170)
        except Exception:
            pass

    # Top fake profiles table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Sample Predictions (Top 30)", ln=True)
    pdf.set_font("Helvetica", "B", 9)

    show_cols = ["username", "name", "channel_name", "prediction", "confidence"]
    show_cols = [c for c in show_cols if c in df.columns][:5]
    col_w = 170 // max(len(show_cols), 1)

    for col in show_cols:
        pdf.cell(col_w, 7, col.replace("_", " ").title(), border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for _, row in df.head(30).iterrows():
        for col in show_cols:
            val = str(row.get(col, ""))[:20]
            pdf.cell(col_w, 6, val, border=1)
        pdf.ln()

    # Footer
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Generated by Fake Profile Detector — Final Year Engineering Project", ln=True, align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"fake_profile_report_{platform}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    )


# ---------------------------------------------------------------------------
# Excel Export
# ---------------------------------------------------------------------------

@app.route("/export/excel/<platform>/<result_file>")
def export_excel(platform: str, result_file: str):
    if not OPENPYXL_AVAILABLE:
        flash("Excel export requires openpyxl. Install it: pip install openpyxl", "danger")
        return redirect(url_for("index"))

    result_path = _safe_in_upload_dir(result_file)
    if result_path is None or not result_path.exists():
        flash("Result file not found.", "warning")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(result_path)
        df = normalize_columns(df)
    except Exception:
        flash("Could not read result file.", "danger")
        return redirect(url_for("index"))

    wb = openpyxl.Workbook()

    # Sheet 1: Summary
    ws_sum = wb.active
    ws_sum.title = "Summary"
    title_font = Font(bold=True, size=14)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="0D6EFD")
    fake_fill = PatternFill("solid", fgColor="FFEBEE")
    legit_fill = PatternFill("solid", fgColor="E8F5E9")

    ws_sum["A1"] = f"Fake Profile Detection Report — {platform.capitalize()}"
    ws_sum["A1"].font = title_font
    ws_sum["A2"] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ws_sum.merge_cells("A1:D1")

    counts = df["prediction"].value_counts().to_dict()
    total = len(df)
    summary_data = [
        ("Metric", "Value"),
        ("Platform", platform.capitalize()),
        ("Total Profiles", total),
        ("Fake Profiles", counts.get("Fake", 0)),
        ("Legit Profiles", counts.get("Legit", 0)),
        ("Fake Rate (%)", round(counts.get("Fake", 0) / total * 100, 1) if total > 0 else 0),
        ("Model Ensemble", "RF + XGBoost + SVM" if XGBOOST_AVAILABLE else "RF + SVM"),
    ]
    for row_idx, row_data in enumerate(summary_data, start=4):
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws_sum.cell(row=row_idx, column=col_idx, value=val)
            if row_idx == 4:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
    ws_sum.column_dimensions["A"].width = 25
    ws_sum.column_dimensions["B"].width = 20

    # Sheet 2: Predictions data
    ws_data = wb.create_sheet("Predictions")
    cols = list(df.columns)
    for col_idx, col_name in enumerate(cols, start=1):
        cell = ws_data.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        for col_idx, col_name in enumerate(cols, start=1):
            val = row.get(col_name, "")
            cell = ws_data.cell(row=row_idx, column=col_idx, value=val)
            if col_name == "prediction":
                if val == "Fake":
                    cell.fill = fake_fill
                elif val == "Legit":
                    cell.fill = legit_fill

    for col_letter in ws_data.column_dimensions:
        ws_data.column_dimensions[col_letter].width = 15

    # Sheet 3: Feature stats
    ws_feat = wb.create_sheet("Feature Statistics")
    ws_feat["A1"] = "Feature Statistics"
    ws_feat["A1"].font = title_font
    stat_headers = ["Feature", "Mean", "Std Dev", "Min", "Max", "Missing %"]
    for col_idx, h in enumerate(stat_headers, start=1):
        cell = ws_feat.cell(row=2, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill

    num_cols = df.select_dtypes(include=np.number).columns
    for row_idx, col_name in enumerate(num_cols, start=3):
        s = df[col_name].dropna()
        missing_pct = round(df[col_name].isna().mean() * 100, 1)
        row_vals = [
            col_name,
            round(float(s.mean()), 3) if len(s) > 0 else "",
            round(float(s.std()), 3) if len(s) > 0 else "",
            round(float(s.min()), 3) if len(s) > 0 else "",
            round(float(s.max()), 3) if len(s) > 0 else "",
            missing_pct,
        ]
        for col_idx, val in enumerate(row_vals, start=1):
            ws_feat.cell(row=row_idx, column=col_idx, value=val)
    for i in range(1, 7):
        ws_feat.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 18

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"fake_profiles_{platform}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    )


# ---------------------------------------------------------------------------
# Live API Lookup
# ---------------------------------------------------------------------------

@csrf.exempt
@app.route("/api/live/github/<username>")
def live_github(username: str):
    if not HTTP_REQUESTS_AVAILABLE:
        return jsonify({"error": "requests library not installed"}), 500
    try:
        headers = {"User-Agent": "FakeProfileDetector/1.0", "Accept": "application/vnd.github.v3+json"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        resp = http_requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers, timeout=8
        )
        if resp.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        if resp.status_code != 200:
            return jsonify({"error": f"GitHub API error: {resp.status_code}"}), resp.status_code
        data = resp.json()
        # Compute account age from created_at
        created_at_str = data.get("created_at", "")
        account_age_days = 0
        if created_at_str:
            try:
                from datetime import datetime, timezone
                created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                account_age_days = (datetime.now(timezone.utc) - created).days
            except Exception:
                pass
        return jsonify({
            "username": data.get("login", ""),
            "followers": data.get("followers", 0),
            "following": data.get("following", 0),
            "public_repos": data.get("public_repos", 0),
            "public_gists": data.get("public_gists", 0),
            "account_age_days": account_age_days,
            "bio": data.get("bio") or "",
            "created_at": created_at_str,
            "avatar_url": data.get("avatar_url", ""),
            "name": data.get("name", ""),
        })
    except http_requests.Timeout:
        return jsonify({"error": "GitHub API timeout"}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/live/reddit/<username>")
def live_reddit(username: str):
    if not HTTP_REQUESTS_AVAILABLE:
        return jsonify({"error": "requests library not installed"}), 500
    try:
        headers = {"User-Agent": "FakeProfileDetector:v1.0 (educational project)"}
        resp = http_requests.get(
            f"https://www.reddit.com/user/{username}/about.json",
            headers=headers, timeout=8
        )
        if resp.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        if resp.status_code != 200:
            return jsonify({"error": f"Reddit API error: {resp.status_code}"}), resp.status_code
        data = resp.json().get("data", {})
        created_ts = data.get("created_utc", 0)
        created_str = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d") if created_ts else ""
        return jsonify({
            "username": data.get("name", ""),
            "karma": data.get("total_karma", 0),
            "created_at": created_str,
            "about": data.get("subreddit", {}).get("public_description", "") or "",
            "icon_img": data.get("icon_img", ""),
            "is_verified": data.get("verified", False),
        })
    except http_requests.Timeout:
        return jsonify({"error": "Reddit API timeout"}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Extended live API routes (Section 1)
# ---------------------------------------------------------------------------

@csrf.exempt
@app.route("/api/live/twitter/<username>")
@app.route("/api/live/x/<username>")
@app.route("/api/v1/live/twitter/<username>")
@app.route("/api/v1/live/x/<username>")
def live_twitter(username: str):
    """Live Twitter/X profile enrichment. Requires TWITTER_BEARER_TOKEN."""
    try:
        from live_enrichment.twitter_enricher import enrich
        result = enrich(username)
        if "error" in result:
            return jsonify(result), 404 if "not found" in result["error"].lower() else 503
        return jsonify(result["raw_data"] | {"features": result["features"],
                                              "completeness": result["data_completeness_score"]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/live/instagram/<username>")
@app.route("/api/v1/live/instagram/<username>")
def live_instagram(username: str):
    """Live Instagram profile enrichment. Requires INSTAGRAM_ACCESS_TOKEN or instaloader."""
    try:
        from live_enrichment.instagram_enricher import enrich
        result = enrich(username)
        if "error" in result and result.get("data_completeness_score", 0) == 0:
            return jsonify(result), 503
        rd = result.get("raw_data", {})
        return jsonify({
            "username": rd.get("username", username),
            "followers": result["features"].get("followers", 0),
            "following": result["features"].get("following", 0),
            "posts": result["features"].get("posts", 0),
            "bio": result["features"].get("bio", ""),
            "avatar_url": result["features"].get("avatar_url", ""),
            "is_verified": result["features"].get("is_verified", 0),
            "features": result["features"],
            "completeness": result["data_completeness_score"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/live/youtube/<channel_id>")
@app.route("/api/v1/live/youtube/<channel_id>")
def live_youtube(channel_id: str):
    """Live YouTube channel enrichment. Requires YOUTUBE_API_KEY."""
    try:
        from live_enrichment.youtube_enricher import enrich
        result = enrich(channel_id)
        if "error" in result and result.get("data_completeness_score", 0) == 0:
            return jsonify(result), 503
        return jsonify({
            "channel_name": result["features"].get("channel_name", ""),
            "subscribers": result["features"].get("subscribers", 0),
            "videos": result["features"].get("videos", 0),
            "about": result["features"].get("about", ""),
            "features": result["features"],
            "completeness": result["data_completeness_score"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/live/linkedin/<username>")
@app.route("/api/v1/live/linkedin/<username>")
def live_linkedin(username: str):
    """LinkedIn profile enrichment. Requires LINKEDIN_ACCESS_TOKEN."""
    try:
        from live_enrichment.linkedin_enricher import enrich
        result = enrich(username)
        if "error" in result and result.get("data_completeness_score", 0) == 0:
            return jsonify(result), 503
        return jsonify({
            "name": username,
            "connections": result["features"].get("connections", 0),
            "headline": result["features"].get("headline", ""),
            "bio": result["features"].get("bio", ""),
            "features": result["features"],
            "completeness": result["data_completeness_score"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/live/discord/<user_id>")
@app.route("/api/v1/live/discord/<user_id>")
def live_discord(user_id: str):
    """Discord profile enrichment. Requires DISCORD_BOT_TOKEN."""
    try:
        from live_enrichment.discord_enricher import enrich
        result = enrich(user_id)
        if "error" in result and result.get("data_completeness_score", 0) == 0:
            return jsonify(result), 503
        return jsonify({
            "username": result["features"].get("username", ""),
            "account_age_days": result["features"].get("account_age_days", 0),
            "has_avatar": result["features"].get("has_avatar", 0),
            "suspicion_score": result["features"].get("suspicion_score", 0),
            "features": result["features"],
            "completeness": result["data_completeness_score"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/live/tiktok/<username>")
@app.route("/api/v1/live/tiktok/<username>")
def live_tiktok(username: str):
    """TikTok profile enrichment. Requires TIKTOK_API_KEY."""
    try:
        from live_enrichment.tiktok_enricher import enrich
        result = enrich(username)
        if "error" in result and result.get("data_completeness_score", 0) == 0:
            return jsonify(result), 503
        return jsonify({
            "username": username,
            "followers": result["features"].get("followers", 0),
            "following": result["features"].get("following", 0),
            "videos": result["features"].get("videos", 0),
            "bio": result["features"].get("bio", ""),
            "features": result["features"],
            "completeness": result["data_completeness_score"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/v1/live/<platform>/<identifier>")
def live_generic_v1(platform: str, identifier: str):
    """
    Generic versioned live enrichment endpoint.
    ---
    tags:
      - Live Enrichment API
    parameters:
      - name: platform
        in: path
        type: string
        required: true
        enum: [github, reddit, twitter, instagram, youtube, linkedin, discord, tiktok]
      - name: identifier
        in: path
        type: string
        required: true
    responses:
      200:
        description: Enrichment result with raw_data, features, and completeness score.
      503:
        description: API key not configured or enrichment failed.
    """
    platform = platform.lower()
    try:
        from live_enrichment import enrich_profile
        result = enrich_profile(platform, identifier)
        status = 200 if result.get("data_completeness_score", 0) > 0 else 503
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/v1/analyze/username/<username>")
def analyze_username_api(username: str):
    """
    Analyze username forensics.
    ---
    tags:
      - Analysis API
    """
    try:
        from features.username_forensics import analyze_username
        return jsonify(analyze_username(username))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/v1/analyze/bio", methods=["POST"])
def analyze_bio_api():
    """
    Bio forensics: language, spam score, template similarity, emojis,
    invisible chars, URL/phone presence, excessive caps.
    ---
    tags:
      - Analysis API
    parameters:
      - in: body
        name: body
        schema:
          properties:
            bio: { type: string }
            platform: { type: string }
    """
    data = request.get_json(silent=True) or {}
    bio = (data.get("bio") or "").strip()
    platform = (data.get("platform") or "").lower()
    if not bio:
        return jsonify({"error": "No bio provided"}), 400
    try:
        from features.bio_forensics import analyze_bio
        return jsonify(analyze_bio(bio, platform))
    except Exception as exc:
        logger.exception("analyze_bio failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/v1/analyze/avatar", methods=["POST"])
def analyze_avatar_api():
    """
    Profile photo analysis: default avatar detection, stock photo match,
    AI-generated heuristic, color entropy, EXIF.
    Accepts either multipart file upload (avatar_file) or JSON {avatar_url, platform}.
    ---
    tags:
      - Analysis API
    """
    import tempfile
    try:
        from features.photo_analysis import analyze_avatar
    except Exception as exc:
        return jsonify({"error": f"Photo analysis unavailable: {exc}"}), 503

    platform = (request.form.get("platform") or
                (request.get_json(silent=True) or {}).get("platform") or "").lower()
    identifier = (request.form.get("identifier") or
                  (request.get_json(silent=True) or {}).get("identifier") or "")

    if "avatar_file" in request.files:
        f = request.files["avatar_file"]
        if not f.filename:
            return jsonify({"error": "Empty file"}), 400
        ext = (f.filename.rsplit(".", 1)[-1].lower()
               if "." in f.filename else "jpg")
        if ext not in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
            return jsonify({"error": f"Unsupported image type: {ext}"}), 400
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        try:
            f.save(tmp.name)
            tmp.close()
            return jsonify(analyze_avatar(tmp.name, platform, identifier))
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass

    body = request.get_json(silent=True) or {}
    url = (body.get("avatar_url") or "").strip()
    if url:
        return jsonify(analyze_avatar(url, platform, identifier))

    return jsonify({"error": "No image provided. Upload avatar_file or pass avatar_url."}), 400


@csrf.exempt
@app.route("/api/v1/analyze/posts", methods=["POST"])
def analyze_posts_api():
    """
    NLP analysis on a paste of recent posts/captions/comments.
    Returns lexical diversity, sentiment variance, duplicate ratio,
    hashtag/URL/emoji ratios, average length.
    ---
    tags:
      - Analysis API
    """
    data = request.get_json(silent=True) or {}
    posts_text = (data.get("posts") or "").strip()
    if not posts_text:
        return jsonify({"error": "No posts provided"}), 400
    posts_list = [p.strip() for p in posts_text.split("\n") if p.strip()]
    if not posts_list:
        return jsonify({"error": "No posts provided"}), 400
    try:
        from features.manual_feature_mapper import analyze_pasted_posts
        result = analyze_pasted_posts(posts_list)
        result["post_count_analyzed"] = len(posts_list)
        return jsonify(result)
    except Exception as exc:
        logger.exception("analyze_posts failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@csrf.exempt
@app.route("/api/v1/cross-platform/<username>")
def cross_platform_api(username: str):
    """
    Cross-platform identity consistency check.
    ---
    tags:
      - Analysis API
    parameters:
      - name: username
        in: path
        type: string
        required: true
      - name: primary
        in: query
        type: string
        description: Primary platform (default: github)
    """
    primary = request.args.get("primary", "github").lower()
    try:
        from features.cross_platform import cross_platform_check
        result = cross_platform_check(username, primary)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# API Keys Configuration page (admin only — Section 1)
# ---------------------------------------------------------------------------

@app.route("/admin/api-keys", methods=["GET", "POST"])
@login_required
def api_keys_config():
    """Admin page for configuring platform API keys (stored as env vars in session)."""
    if not current_user.is_admin:
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))

    from live_enrichment import PLATFORM_ENV_VARS, get_platform_api_status

    if request.method == "POST":
        for platform, env_var in PLATFORM_ENV_VARS.items():
            if not env_var:
                continue
            val = request.form.get(f"key_{platform}", "").strip()
            if val:
                os.environ[env_var] = val
                flash(f"{platform.capitalize()} API key updated.", "success")
            elif request.form.get(f"clear_{platform}"):
                os.environ.pop(env_var, None)
                flash(f"{platform.capitalize()} API key cleared.", "warning")
        return redirect(url_for("api_keys_config"))

    api_status = get_platform_api_status()
    return render_template(
        "api_keys.html",
        platform_env_vars=PLATFORM_ENV_VARS,
        api_status=api_status,
    )


# ---------------------------------------------------------------------------
# Tiered single-profile prediction route (Section 6)
# ---------------------------------------------------------------------------

@app.route("/predict-deep/<platform>", methods=["POST"])
@csrf.exempt
def predict_deep(platform: str):
    """
    Deep analysis prediction for a single profile.
    Returns JSON with tiered prediction result.
    ---
    tags:
      - Prediction API
    """
    platform = platform.lower()
    if platform not in PLATFORMS:
        return jsonify({"error": f"Unknown platform: {platform}"}), 404
    if not MODEL_PATHS[platform].exists():
        return jsonify({"error": f"No trained model for {platform}. Train first."}), 503

    if request.is_json:
        profile_data = request.get_json(force=True) or {}
    else:
        profile_data = request.form.to_dict()

    enable_deep = request.args.get("deep", "1") != "0"

    try:
        from prediction.tiered_predictor import predict_single_profile
        result = predict_single_profile(
            profile_data=profile_data,
            platform=platform,
            model_path=MODEL_PATHS[platform],
            feature_builder=FEATURE_BUILDERS[platform],
            enable_deep_analysis=enable_deep,
        )

        # Log prediction
        try:
            log_entry = PredictionLog(
                username=profile_data.get("username") or profile_data.get("name") or "",
                platform=platform,
                prediction=result.get("label"),
                confidence=result.get("confidence"),
                analysis_depth=result.get("analysis_depth"),
                features_json=json.dumps({k: str(v) for k, v in profile_data.items()}),
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception:
            pass

        return jsonify(result)
    except Exception as exc:
        logger.exception("predict_deep error [%s]: %s", platform, exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# JSON API (with Swagger-style docstrings for flasgger)
# ---------------------------------------------------------------------------

@csrf.exempt
@app.route("/api/predict/<platform>", methods=["POST"])
def api_predict(platform: str):
    """
    Predict fake vs legit for a profile or batch of profiles.
    ---
    tags:
      - Prediction API
    parameters:
      - name: platform
        in: path
        type: string
        required: true
        enum: [instagram, facebook, x, linkedin, github, discord, youtube, tiktok, reddit, snapchat]
      - name: body
        in: body
        required: true
        schema:
          type: object
          example:
            username: "bot123"
            followers: 5
            following: 900
            posts: 0
            bio: ""
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: string
              enum: [Fake, Legit]
            confidence:
              type: number
    """
    platform = platform.lower()
    if platform not in PLATFORMS:
        return jsonify({"error": "Unknown platform"}), 404

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    batch = payload if isinstance(payload, list) else [payload]

    try:
        df = pd.DataFrame(batch)
    except Exception as exc:
        return jsonify({"error": f"Could not parse payload: {exc}"}), 400

    df_pred, _, _, warnings = predict_with_saved_model(
        df=df,
        model_path=MODEL_PATHS[platform],
        charts_dir=CHARTS_DIR,
        tag="",
        title="",
        feature_builder=FEATURE_BUILDERS[platform],
        skip_charts=True,
    )

    results = []
    for _, row in df_pred.iterrows():
        entry: dict = {"prediction": row.get("prediction", "Legit")}
        conf = _safe_confidence(row.get("confidence"))
        if conf is not None:
            entry["confidence"] = conf
        results.append(entry)

    response_data = results[0] if not isinstance(payload, list) else results
    return jsonify(response_data)


@csrf.exempt
@app.route("/api/stats")
def api_stats():
    """
    Get model status and training metrics for all platforms.
    ---
    tags:
      - System API
    responses:
      200:
        description: Platform status summary
    """
    hist = load_history()
    result = {}
    for p in PLATFORMS:
        status = get_model_status(p)
        entries = hist.get(p, [])
        latest = entries[-1] if entries else {}
        result[p] = {
            "model_trained": status["exists"],
            "trained_at": status.get("trained_at"),
            "latest_metrics": latest,
        }
    return jsonify({
        "platforms": result,
        "ensemble": "RF+XGBoost+SVM" if XGBOOST_AVAILABLE else "RF+SVM",
        "shap_enabled": SHAP_AVAILABLE,
    })


# ---------------------------------------------------------------------------
# Swagger / API Docs
# ---------------------------------------------------------------------------

try:
    from flasgger import Swagger
    swagger_config = {
        "headers": [],
        "specs": [{"endpoint": "apispec", "route": "/apispec.json"}],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/docs",
    }
    swagger_template = {
        "info": {
            "title": "Fake Profile Detector API",
            "description": "REST API for detecting fake social media profiles using ML ensemble (RF + XGBoost + SVM)",
            "version": "2.0.0",
        }
    }
    Swagger(app, config=swagger_config, template=swagger_template)
    logger.info("Swagger UI available at /api/docs")
except ImportError:
    logger.warning("flasgger not installed — /api/docs unavailable")


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

@app.route("/generate-data", methods=["GET", "POST"])
@_auth_required
def generate_data():
    generated_stats = None
    preview_rows = []
    preview_columns = []
    download_file = None
    generation_error = None

    if request.method == "POST":
        platform = request.form.get("platform", "instagram").lower()
        try:
            total_count = int(request.form.get("total_count", 1000))
            fake_ratio = float(request.form.get("fake_ratio", 0.4))
            add_noise = request.form.get("add_noise") == "1"
            action = request.form.get("action", "generate")

            total_count = max(100, min(total_count, 10000))
            fake_ratio = max(0.1, min(fake_ratio, 0.9))

            df = generate_dataset(platform, total_count=total_count,
                                  fake_ratio=fake_ratio, add_noise=add_noise)
            generated_stats = get_generation_stats(df)
            generated_stats["platform"] = platform

            # Save to upload dir for download
            result_name = f"generated_{platform}_{os.urandom(6).hex()}.csv"
            result_path = UPLOAD_DIR / result_name
            df.to_csv(result_path, index=False)
            download_file = result_name

            preview_columns = list(df.columns)
            preview_rows = df.head(20).to_dict(orient="records")

            flash(f"Generated {len(df)} profiles for {platform.capitalize()} "
                  f"(fake={generated_stats['fake_count']} legit={generated_stats['legit_count']})", "success")

            # Optionally feed directly into training pipeline
            if action == "generate_train":
                try:
                    train_path = UPLOAD_DIR / f"generated_train_{platform}_{os.urandom(6).hex()}.csv"
                    df.to_csv(train_path, index=False)
                    ok, msg, metrics, rows_used = train_and_save(
                        labeled_csv_paths=[train_path],
                        model_path=MODEL_PATHS[platform],
                        feature_builder=FEATURE_BUILDERS[platform],
                        min_rows=MIN_TRAIN_ROWS,
                        charts_dir=CHARTS_DIR,
                        platform=platform,
                        include_stats=True,
                    )
                    _cleanup(train_path)
                    flash(msg, "success" if ok else "danger")
                    if ok:
                        log_training_event(platform, rows_used, metrics)
                except Exception as exc:
                    flash(f"Training failed: {exc}", "danger")
                    logger.exception("generate_train error")

        except Exception as exc:
            generation_error = str(exc)
            logger.exception("Data generation error")

    return render_template(
        "generate_data.html",
        platforms=PLATFORMS,
        generated_stats=generated_stats,
        preview_rows=preview_rows,
        preview_columns=preview_columns,
        download_file=download_file,
        generation_error=generation_error,
    )


# ---------------------------------------------------------------------------
# Feedback System
# ---------------------------------------------------------------------------

@app.route("/feedback/flag", methods=["POST"])
def feedback_flag():
    """Flag a prediction as incorrect (AJAX or form POST)."""
    platform = request.form.get("platform", "")
    predicted_label = request.form.get("predicted_label", "")
    corrected_label = request.form.get("corrected_label") or None
    confidence_raw = request.form.get("confidence")
    profile_json = request.form.get("profile_data_json", "{}")

    try:
        confidence = float(confidence_raw) if confidence_raw else None
        profile_data = json.loads(profile_json)
        user_id = current_user.id if current_user.is_authenticated else None
        save_feedback(
            db, FeedbackEntry, platform, profile_data,
            predicted_label, corrected_label, confidence, user_id
        )
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "ok", "message": "Feedback saved"})
        flash("Thank you — feedback recorded.", "success")
    except Exception as exc:
        logger.error("feedback_flag error: %s", exc)
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": str(exc)}), 500
        flash(f"Could not save feedback: {exc}", "danger")

    return redirect(request.referrer or url_for("index"))


@app.route("/feedback/review")
@login_required
def feedback_review():
    """Admin page to review all feedback entries."""
    if REQUIRE_AUTH and not current_user.is_admin:
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    entries = get_all_feedback(db, FeedbackEntry)
    stats = get_feedback_stats(db, FeedbackEntry)
    return render_template(
        "feedback_review.html",
        feedback_entries=entries,
        feedback_stats=stats,
        platforms=PLATFORMS,
    )


@app.route("/feedback/review/<int:entry_id>/mark", methods=["POST"])
@login_required
def feedback_mark_reviewed(entry_id: int):
    """Mark a feedback entry as reviewed."""
    mark_reviewed(db, FeedbackEntry, entry_id)
    flash("Marked as reviewed.", "success")
    return redirect(url_for("feedback_review"))


@app.route("/feedback/retrain/<platform>", methods=["POST"])
@login_required
def feedback_retrain(platform: str):
    """Retrain the model for a platform using corrected feedback labels."""
    platform = platform.lower()
    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("feedback_review"))

    feedback_data = get_feedback_for_retraining(db, FeedbackEntry, platform)
    if len(feedback_data) < MIN_TRAIN_ROWS:
        flash(f"Not enough corrected feedback entries (need >= {MIN_TRAIN_ROWS}, "
              f"have {len(feedback_data)}).", "warning")
        return redirect(url_for("feedback_review"))

    try:
        # Build corrected DataFrame from feedback
        rows = []
        for entry in feedback_data:
            profile = entry.get("profile_data", {})
            profile["label"] = 1 if entry["corrected_label"] == "Fake" else 0
            rows.append(profile)
        df_corrected = pd.DataFrame(rows)

        # Merge with original training CSV if it exists
        train_csv = APP_DIR / "csv" / f"{platform}_train.csv"
        save_path = UPLOAD_DIR / f"feedback_train_{platform}_{os.urandom(6).hex()}.csv"

        if train_csv.exists():
            df_orig = pd.read_csv(train_csv)
            df_merged = pd.concat([df_orig, df_corrected], ignore_index=True)
        else:
            df_merged = df_corrected

        df_merged.to_csv(save_path, index=False)

        ok, msg, metrics, rows_used = train_and_save(
            labeled_csv_paths=[save_path],
            model_path=MODEL_PATHS[platform],
            feature_builder=FEATURE_BUILDERS[platform],
            min_rows=MIN_TRAIN_ROWS,
            charts_dir=CHARTS_DIR,
            platform=platform,
            include_stats=True,
        )
        _cleanup(save_path)
        flash(msg, "success" if ok else "danger")
        if ok:
            log_training_event(platform, rows_used, metrics)
            flash(f"Model retrained with {len(feedback_data)} feedback corrections.", "info")
    except Exception as exc:
        flash(f"Retrain failed: {exc}", "danger")
        logger.exception("feedback_retrain error")

    return redirect(url_for("feedback_review"))


# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------

@app.route("/drift-monitor")
def drift_monitor():
    """Drift monitoring dashboard showing PSI per platform."""
    drift_summaries = get_all_platforms_drift_summary()
    # Fill in platforms with no data
    platforms_with_data = {s["platform"] for s in drift_summaries}
    for p in PLATFORMS:
        if p not in platforms_with_data:
            drift_summaries.append({
                "platform": p, "max_psi": 0.0, "drift_level": "ok",
                "drifted_features": [], "confidence_trend": [],
                "retrain_recommended": False, "latest_check": None, "prediction_count": 0,
            })
    drift_summaries.sort(key=lambda s: s.get("max_psi", 0.0), reverse=True)
    return render_template(
        "drift_monitor.html",
        drift_summaries=drift_summaries,
        platforms=PLATFORMS,
        check_interval=PREDICTION_COUNT_CHECK,
    )


@app.route("/drift-monitor/<platform>", methods=["POST"])
def drift_monitor_platform(platform: str):
    """Manually trigger drift check for a specific platform."""
    platform = platform.lower()
    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("drift_monitor"))

    train_csv = APP_DIR / "csv" / f"{platform}_train.csv"
    if not train_csv.exists():
        flash(f"No training CSV found at csv/{platform}_train.csv.", "warning")
        return redirect(url_for("drift_monitor"))

    # Load recent predictions from uploads dir
    recent_files = sorted(
        [f for f in UPLOAD_DIR.glob(f"result_{platform}_*.csv") if f.is_file()],
        key=lambda f: f.stat().st_mtime, reverse=True
    )
    if not recent_files:
        flash(f"No recent prediction results found for {platform.capitalize()}.", "warning")
        return redirect(url_for("drift_monitor"))

    try:
        df_recent = pd.read_csv(recent_files[0])
        df_recent = normalize_columns(df_recent)
        confidence_vals = pd.to_numeric(df_recent.get("confidence", pd.Series(dtype=float)), errors="coerce").dropna()
        avg_conf = float(confidence_vals.mean()) if len(confidence_vals) > 0 else 50.0

        summary = check_and_log_drift(platform, train_csv, df_recent, avg_conf)
        drift_level = summary.get("drift_level", "ok")
        max_psi = summary.get("max_psi", 0.0)
        flash(
            f"Drift check for {platform.capitalize()}: level={drift_level.upper()} max_psi={max_psi:.4f}",
            "success" if drift_level == "ok" else ("warning" if drift_level == "warn" else "danger")
        )
    except Exception as exc:
        flash(f"Drift check failed: {exc}", "danger")
        logger.exception("drift_monitor_platform error")

    return redirect(url_for("drift_monitor"))


# ---------------------------------------------------------------------------
# Real Dataset Integration
# ---------------------------------------------------------------------------

@app.route("/real-datasets", methods=["GET"])
def real_datasets():
    """Show available real datasets and local auto-discovered files."""
    from datasets.real_datasets import discover_local_datasets, KNOWN_DATASETS
    local = discover_local_datasets()
    return render_template(
        "real_datasets.html",
        known_datasets=KNOWN_DATASETS,
        local_datasets=local,
        platforms=PLATFORMS,
        platform_icons=PLATFORM_ICONS,
    )


@app.route("/real-datasets/load", methods=["POST"])
@_auth_required
@csrf.exempt
def load_real_dataset_route():
    """
    Load a real dataset from a local path or URL, adapt it, and optionally train.
    POST params: platform, path_or_url, adapter, do_train (checkbox)
    """
    from datasets.real_datasets import load_real_dataset, save_for_training

    platform     = request.form.get("platform", "").lower()
    path_or_url  = request.form.get("path_or_url", "").strip()
    adapter_name = request.form.get("adapter", "generic")
    do_train     = request.form.get("do_train") == "1"

    if platform not in PLATFORMS:
        flash("Unknown platform.", "danger")
        return redirect(url_for("real_datasets"))

    if not path_or_url:
        flash("Please provide a file path or URL.", "danger")
        return redirect(url_for("real_datasets"))

    df, msg = load_real_dataset(path_or_url, adapter_name)
    if df.empty:
        flash(f"Failed to load dataset: {msg}", "danger")
        return redirect(url_for("real_datasets"))

    flash(msg, "info")

    if do_train:
        out_path = save_for_training(df, platform)
        try:
            ok, train_msg, metrics, rows_used = train_and_save(
                labeled_csv_paths=[out_path],
                model_path=MODEL_PATHS[platform],
                feature_builder=FEATURE_BUILDERS[platform],
                min_rows=MIN_TRAIN_ROWS,
                charts_dir=CHARTS_DIR,
                platform=platform,
                include_stats=True,
                use_stacking=False,
                balancing_strategy="smote" if IMBLEARN_AVAILABLE else "none",
            )
            flash(train_msg, "success" if ok else "danger")
            if ok and metrics:
                log_training_event(platform, rows_used, metrics)
                flash(
                    f"Accuracy: {metrics.get('accuracy','?')}  "
                    f"F1: {metrics.get('f1','?')}  "
                    f"Ensemble: {metrics.get('ensemble_type','?')}",
                    "info",
                )
        except Exception as exc:
            flash(f"Training error: {exc}", "danger")

    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# JSON REST API  (/api/v1/predict)
# ---------------------------------------------------------------------------

@app.route("/api/v1/predict/<platform>", methods=["POST"])
@csrf.exempt
def api_predict_v1(platform: str):
    """
    REST API endpoint — predict one or more profiles as JSON.

    POST body (JSON or form-data):
      Single profile:  { "username": "...", "followers": 1000, ... }
      Multiple:        { "profiles": [ { ... }, { ... } ] }

    Returns:
      { "platform": "...", "results": [ { "prediction": "Fake|Legit",
                                          "confidence": 87.3, ... } ] }
    """
    platform = platform.lower()
    if platform not in PLATFORMS:
        return jsonify({"error": f"Unknown platform: {platform}"}), 404

    if not MODEL_PATHS[platform].exists():
        return jsonify({"error": f"No trained model for {platform}. Train first."}), 503

    # Accept JSON or form data
    if request.is_json:
        body = request.get_json(force=True) or {}
    else:
        body = request.form.to_dict()

    profiles = body.get("profiles", None)
    if profiles is None:
        profiles = [body]

    if not isinstance(profiles, list) or len(profiles) == 0:
        return jsonify({"error": "Provide 'profiles' as a list of profile dicts."}), 400

    df = pd.DataFrame(profiles)
    tag = os.urandom(6).hex()

    try:
        df_pred, counts, _, warnings = predict_with_saved_model(
            df=df,
            model_path=MODEL_PATHS[platform],
            charts_dir=CHARTS_DIR,
            tag=tag,
            title=f"{platform} API prediction",
            feature_builder=FEATURE_BUILDERS[platform],
            skip_charts=True,
            include_shap=False,
            include_anomaly=False,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    results = []
    for _, row in df_pred.iterrows():
        results.append({
            "prediction": row.get("prediction", "Unknown"),
            "confidence": row.get("confidence", None),
            "is_fake": row.get("prediction") == "Fake",
        })

    return jsonify({
        "platform": platform,
        "count": len(results),
        "summary": counts,
        "results": results,
        "warnings": warnings,
    })


@app.route("/api/v1/status", methods=["GET"])
@csrf.exempt
def api_status():
    """Return model status and platform info as JSON."""
    model_status = {p: get_model_status(p) for p in PLATFORMS}
    return jsonify({
        "platforms": PLATFORMS,
        "models": model_status,
        "capabilities": {
            "xgboost": XGBOOST_AVAILABLE,
            "lightgbm": LIGHTGBM_AVAILABLE,
            "shap": SHAP_AVAILABLE,
            "optuna": OPTUNA_AVAILABLE,
            "histgb": HISTGB_AVAILABLE,
        },
    })


@app.route("/admin/confidence-monitor")
@login_required
def confidence_monitor():
    """
    Dashboard showing per-platform confidence distribution from prediction history.
    Helps identify platforms where the model is systematically under- or over-confident.
    """
    hist = load_history()
    stats = {}
    for p in PLATFORMS:
        entries = hist.get(p, [])
        confs = []
        for e in entries:
            try:
                c = float(e.get("confidence", 0) or 0)
                if c > 0:
                    confs.append(c)
            except (TypeError, ValueError):
                pass

        if confs:
            import statistics
            low_conf = [c for c in confs if c < 60]
            stats[p] = {
                "count": len(confs),
                "mean": round(statistics.mean(confs), 1),
                "median": round(statistics.median(confs), 1),
                "low_conf_count": len(low_conf),
                "low_conf_pct": round(len(low_conf) / len(confs) * 100, 1),
                "model_exists": MODEL_PATHS[p].exists(),
            }
        else:
            stats[p] = {
                "count": 0, "mean": 0, "median": 0,
                "low_conf_count": 0, "low_conf_pct": 0,
                "model_exists": MODEL_PATHS[p].exists(),
            }

    return render_template("admin_confidence_monitor.html", stats=stats, platforms=PLATFORMS)


@app.route("/admin/recalibrate/<platform>", methods=["POST"])
@login_required
def admin_recalibrate(platform: str):
    """
    Re-calibrate the confidence calibrator for a platform using stored feedback.
    Requires at least 20 corrected FeedbackEntry rows for the platform.
    """
    platform = platform.lower()
    if platform not in PLATFORMS:
        return jsonify({"error": "Unknown platform"}), 400

    try:
        from ML.confidence_calibrator import recalibrate_from_feedback
        calibrator = recalibrate_from_feedback(MODEL_PATHS[platform], platform)
        if calibrator is None:
            return jsonify({
                "ok": False,
                "message": f"Insufficient feedback data for {platform} (need ≥20 corrected entries).",
            })
        return jsonify({
            "ok": True,
            "platform": platform,
            "method": calibrator.method,
            "message": f"Calibrator updated for {platform} using method={calibrator.method}.",
        })
    except Exception as exc:
        logger.exception("Recalibration failed for %s", platform)
        return jsonify({"ok": False, "message": str(exc)}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(500)
def internal_error(error):
    import traceback
    tb = traceback.format_exc()
    logger.error("500 error: %s", tb)
    if app.debug:
        return (
            f'<div style="font-family:monospace;padding:20px;background:#1e1e1e;'
            f'color:#f44;max-width:900px;margin:20px auto;border-radius:8px">'
            f'<h2 style="color:#ff6b6b">&#9888; Internal Server Error</h2>'
            f'<pre style="color:#ffa;white-space:pre-wrap;font-size:12px">{tb}</pre></div>',
            500,
        )
    return render_template(
        "error.html",
        error_title="Something went wrong",
        error_message="An internal error occurred. Check server logs for details.",
        error_hint="If this happened during prediction, retrain the model at /train/<platform>.",
    ), 500


@app.errorhandler(ValueError)
def value_error_handler(error):
    """Catch sklearn feature-mismatch errors with a user-friendly message."""
    import traceback
    tb = traceback.format_exc()
    logger.error("ValueError in prediction: %s", error)

    error_str = str(error)
    if "feature names should match" in error_str.lower():
        hint = (
            "The saved model was trained with a different feature set. "
            "Retrain the model at /train/<platform> to pick up the latest features."
        )
    else:
        hint = error_str

    if app.debug:
        return (
            f'<div style="font-family:monospace;padding:20px;background:#1e1e1e;'
            f'color:#f44;max-width:900px;margin:20px auto;border-radius:8px">'
            f'<h2 style="color:#ff6b6b">&#9888; Feature Mismatch</h2>'
            f'<p style="color:#aaa">{hint}</p>'
            f'<pre style="color:#ffa;white-space:pre-wrap;font-size:12px">{tb}</pre></div>',
            500,
        )
    return render_template(
        "error.html",
        error_title="Model Feature Mismatch",
        error_message=hint,
        error_hint="Go to /train/<platform> and retrain the model.",
    ), 500


# ---------------------------------------------------------------------------
# Admin: Known Public Figures Allowlist
# ---------------------------------------------------------------------------

@app.route("/admin/public-figures", methods=["GET", "POST"])
@login_required
def admin_public_figures():
    """Edit the known public figures allowlist per platform."""
    if not current_user.is_admin:
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))

    from prediction.celebrity_safeguard import load_allowlist, save_allowlist

    allowlist = load_allowlist()

    # Ensure every platform key exists
    for p in PLATFORMS:
        allowlist.setdefault(p, [])

    if request.method == "POST":
        action   = request.form.get("action", "")
        platform = request.form.get("platform", "").lower()
        username = (request.form.get("username") or "").strip().lower()

        if platform not in PLATFORMS:
            flash("Unknown platform.", "danger")
        elif not username:
            flash("Username cannot be empty.", "danger")
        elif action == "add":
            if username not in [u.lower() for u in allowlist.get(platform, [])]:
                allowlist.setdefault(platform, []).append(username)
                save_allowlist(allowlist)
                flash(f"Added '{username}' to {platform} allowlist.", "success")
            else:
                flash(f"'{username}' is already in the {platform} allowlist.", "info")
        elif action == "remove":
            allowlist[platform] = [
                u for u in allowlist.get(platform, [])
                if u.lower() != username
            ]
            save_allowlist(allowlist)
            flash(f"Removed '{username}' from {platform} allowlist.", "success")
        else:
            flash("Unknown action.", "danger")

        return redirect(url_for("admin_public_figures"))

    return render_template(
        "admin_public_figures.html",
        allowlist=allowlist,
        platforms=PLATFORMS,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
