"""
datasets/real_datasets.py — Real public dataset integration for fake profile detection.

Supports:
  - Local dataset auto-discovery (Dataset/ folder)
  - URL-based downloads (CSV/ZIP from GitHub, academic repos, etc.)
  - Kaggle dataset downloads (requires ~/.kaggle/kaggle.json)
  - Format adapters for common dataset schemas

Supported adapters:
  - instagram_kaggle    : Kaggle Instagram Fake Profile dataset (pre-computed features)
  - twibot20            : TwiBot-20 Twitter bot detection dataset
  - cresci17            : Cresci-2017 Twitter social spambots
  - generic_twitter     : Any labeled Twitter CSV with follower/following/tweet cols
  - generic_labeled     : Any labeled CSV with a 'label'/'fake'/'is_bot' column
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Known public dataset sources
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_DATASET_DIR = Path(__file__).parent.parent / "Dataset"
CSV_OUT_DIR       = Path(__file__).parent.parent / "csv"

# Pre-configured public datasets with known stable URLs
KNOWN_DATASETS: Dict[str, Dict[str, Any]] = {
    "instagram_local": {
        "platform":    "instagram",
        "name":        "Instagram Fake Profile Dataset (Local — 5,000 profiles)",
        "description": "5,000 Instagram profiles (2,500 fake / 2,500 real) with pre-computed features. Already present in Dataset/ folder.",
        "adapter":     "instagram_kaggle",
        "local_path":  "Dataset/instagram_bot_detection/Instagram_fake_profile_dataset.csv",
        "url":         None,
        "label_col":   "fake",
        "download_instructions": None,
    },
    "twibot22": {
        "platform":    "x",
        "name":        "TwiBot-22 (Twitter Bot Benchmark — 1M users, CC BY 4.0)",
        "description": "Largest Twitter bot benchmark: 1,000,000 accounts across 8 domains. Direct download from Zenodo (no login required).",
        "adapter":     "twibot20",
        "url":         None,
        "label_col":   "label",
        "download_instructions": "Direct download at https://zenodo.org/record/6950806 (CC BY 4.0, no login needed). Extract label.csv + user.json.",
    },
    "cresci17": {
        "platform":    "x",
        "name":        "Cresci-2017 Twitter Social Spambots",
        "description": "Classic benchmark: genuine accounts + 3 types of social spambots (~14,000 labeled accounts). Free ZIP download.",
        "adapter":     "cresci17",
        "url":         None,
        "label_col":   "label",
        "download_instructions": "Download ZIP from http://mib.projects.iit.cnr.it/dataset.html (no login). Extract genuine_accounts/ and social_spambots_*/ CSVs.",
    },
    "bodegha_github": {
        "platform":    "github",
        "name":        "BoDeGHa — GitHub Bot Detection Dataset (~5,700 accounts)",
        "description": "Ground-truth labeled GitHub bots vs humans. CSV files are directly in the GitHub repo — no auth needed.",
        "adapter":     "generic",
        "url":         None,
        "label_col":   "type",
        "download_instructions": "Clone or download CSV from https://github.com/mehdigolzadeh/BoDeGHa — label column is 'type' (Bot/Human).",
    },
    "osome_bot_repo": {
        "platform":    "x",
        "name":        "OSoMe Bot Repository (Indiana University)",
        "description": "Aggregates 20+ labeled Twitter bot datasets. Many are direct ZIP downloads with no login.",
        "adapter":     "generic",
        "url":         None,
        "label_col":   "label",
        "download_instructions": "Browse individual datasets at https://botometer.osome.iu.edu/bot-repository/datasets.html",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Format adapters — convert various dataset schemas → our standard format
# ─────────────────────────────────────────────────────────────────────────────

class DatasetAdapter:
    """Base adapter — override `transform(df) -> pd.DataFrame`."""

    label_col: str = "label"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _ensure_label(self, df: pd.DataFrame, src_col: str) -> pd.DataFrame:
        """Normalise label column to binary int (0=legit, 1=fake)."""
        if src_col in df.columns:
            df = df.rename(columns={src_col: "label"})
        if "label" not in df.columns:
            raise ValueError("No label column found.")
        # Convert textual labels
        truthy  = {"1", "true", "yes", "fake", "bot", "spam", "1.0"}
        falsey  = {"0", "false", "no", "legit", "real", "human", "genuine", "0.0"}
        df["label"] = df["label"].apply(
            lambda v: 1 if str(v).strip().lower() in truthy
                      else (0 if str(v).strip().lower() in falsey else None)
        )
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df


class InstagramKaggleAdapter(DatasetAdapter):
    """
    Adapter for the pre-computed Instagram Fake Profile dataset.
    Columns: profile pic, nums/length username, fullname words,
             nums/length fullname, name==username, description length,
             external URL, private, #posts, #followers, #follows, fake
    """
    label_col = "fake"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        # Map to our standard format so the feature builder can use it
        rename = {
            "#followers":          "followers",
            "#follows":            "following",
            "#posts":              "posts",
            "profile pic":         "has_profile_pic",
            "description length":  "bio_len_raw",
            "external url":        "bio_has_url",
            "private":             "is_private",
            "name==username":      "name_eq_username",
            "fullname words":      "fullname_words",
            "nums/length username": "uname_digit_ratio",
            "nums/length fullname": "fullname_digit_ratio",
        }
        df = df.rename(columns=rename)
        # Ensure label
        df = self._ensure_label(df, "fake")
        return df


class TwiBot20Adapter(DatasetAdapter):
    """
    Adapter for TwiBot-20 dataset format.
    Expected columns: id, username, followers_count, friends_count,
                      statuses_count, verified, created_at, description, label
    """
    label_col = "label"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        rename = {
            "followers_count": "followers",
            "friends_count":   "following",
            "statuses_count":  "tweets",
            "screen_name":     "username",
            "verified":        "is_verified",
            "description":     "bio",
        }
        df = df.rename(columns=rename)
        df = self._ensure_label(df, "label")
        return df


class Cresci17Adapter(DatasetAdapter):
    """
    Adapter for Cresci-2017 dataset format.
    Handles the genuine_accounts.csv / social_spambots_*.csv files.
    """
    label_col = "label"

    def transform(self, df: pd.DataFrame, is_bot: int = 0) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        rename = {
            "followers_count": "followers",
            "friends_count":   "following",
            "statuses_count":  "tweets",
            "screen_name":     "username",
            "verified":        "is_verified",
            "description":     "bio",
            "listed_count":    "listed_count",
        }
        df = df.rename(columns=rename)
        df["label"] = is_bot  # caller sets 0 for genuine, 1 for bots
        return df


class GenericLabeledAdapter(DatasetAdapter):
    """
    Fallback adapter for any labeled CSV.
    Tries to auto-detect label columns and map common field names.
    """
    _LABEL_CANDIDATES = ["label", "fake", "is_fake", "is_bot", "bot", "target"]
    _FOLLOWER_CANDIDATES = ["followers", "followers_count", "#followers", "follower_count"]
    _FOLLOWING_CANDIDATES = ["following", "friends_count", "#follows", "friends"]
    _TWEET_CANDIDATES = ["tweets", "statuses_count", "tweet_count", "posts", "videos"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        # Find label column
        label_src = next((c for c in self._LABEL_CANDIDATES if c in df.columns), None)
        if label_src is None:
            raise ValueError(
                f"No label column found. Expected one of: {self._LABEL_CANDIDATES}. "
                f"Found: {list(df.columns)}"
            )
        # Auto-rename common fields
        renames: dict = {}
        for src in self._FOLLOWER_CANDIDATES:
            if src in df.columns and "followers" not in df.columns:
                renames[src] = "followers"
                break
        for src in self._FOLLOWING_CANDIDATES:
            if src in df.columns and "following" not in df.columns:
                renames[src] = "following"
                break
        for src in self._TWEET_CANDIDATES:
            if src in df.columns and "tweets" not in df.columns:
                renames[src] = "tweets"
                break
        if renames:
            df = df.rename(columns=renames)
        df = self._ensure_label(df, label_src)
        return df


ADAPTERS: Dict[str, DatasetAdapter] = {
    "instagram_kaggle": InstagramKaggleAdapter(),
    "twibot20":         TwiBot20Adapter(),
    "cresci17":         Cresci17Adapter(),
    "generic":          GenericLabeledAdapter(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_url(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a URL to dest path. Returns True on success."""
    try:
        req = Request(url, headers={"User-Agent": "FakeProfileAI/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        logger.info("Downloaded %s → %s", url, dest)
        return True
    except Exception as exc:
        logger.warning("Download failed (%s): %s", url, exc)
        return False


def download_zip(url: str, extract_to: Path, timeout: int = 60) -> List[Path]:
    """Download a ZIP from url, extract to extract_to, return extracted file paths."""
    extracted: List[Path] = []
    try:
        req = Request(url, headers={"User-Agent": "FakeProfileAI/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(extract_to)
            extracted = [extract_to / name for name in zf.namelist()]
        logger.info("Extracted %d files from ZIP to %s", len(extracted), extract_to)
    except Exception as exc:
        logger.warning("ZIP download/extract failed (%s): %s", url, exc)
    return extracted


def _try_kaggle_download(dataset_slug: str, dest_dir: Path) -> List[Path]:
    """
    Download a Kaggle dataset using the Kaggle API.
    Requires ~/.kaggle/kaggle.json with API credentials.
    dataset_slug format: 'owner/dataset-name'
    """
    try:
        import kaggle  # type: ignore
        dest_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(
            dataset_slug, path=str(dest_dir), unzip=True, quiet=False
        )
        return list(dest_dir.glob("*.csv"))
    except ImportError:
        logger.warning("kaggle package not installed. Run: pip install kaggle")
    except Exception as exc:
        logger.warning("Kaggle download failed (%s): %s", dataset_slug, exc)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Auto-discovery of local datasets
# ─────────────────────────────────────────────────────────────────────────────

def discover_local_datasets() -> Dict[str, Dict]:
    """
    Scan Dataset/ and csv/ folders and return a dict of discovered datasets with metadata.
    """
    found: Dict[str, Dict] = {}
    search_dirs = [LOCAL_DATASET_DIR, CSV_OUT_DIR]

    seen = set()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for csv_file in search_dir.rglob("*.csv"):
            if str(csv_file) in seen:
                continue
            seen.add(str(csv_file))
            try:
                df_sample = pd.read_csv(csv_file, nrows=5)
                cols = set(c.strip().lower() for c in df_sample.columns)

                # Detect label column
                label_col = next(
                    (c for c in ["fake", "label", "is_fake", "is_bot", "bot", "target"] if c in cols),
                    None,
                )
                if label_col is None:
                    continue

                # Detect platform hint from folder / filename
                try:
                    rel = str(csv_file.relative_to(search_dir)).lower()
                except ValueError:
                    rel = csv_file.name.lower()
                platform = "unknown"
                for p in ["instagram", "twitter", "facebook", "github", "reddit",
                          "youtube", "tiktok", "linkedin", "discord", "snapchat"]:
                    if p in rel:
                        platform = p
                        break
                if "x_train" in rel or "twitter" in rel:
                    platform = "x"

                # Detect adapter
                adapter = "generic"
                if {"#followers", "#follows", "description length"} & cols:
                    adapter = "instagram_kaggle"
                    platform = "instagram"
                elif {"followers_count", "friends_count", "statuses_count"} & cols:
                    adapter = "twibot20" if "label" in cols else "cresci17"
                    platform = "x"

                found[str(csv_file)] = {
                    "path":      str(csv_file),
                    "filename":  csv_file.name,
                    "platform":  platform,
                    "adapter":   adapter,
                    "label_col": label_col,
                    "rows":      None,  # lazy-loaded
                }
            except Exception as exc:
                logger.debug("Could not inspect %s: %s", csv_file, exc)

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — load, adapt, and return ready-to-train DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def load_real_dataset(
    path_or_url: str,
    adapter_name: str = "generic",
    label_col: str = "label",
    download_cache_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Load a real labeled dataset from a local path or URL.

    Returns:
        (df, message) where df is the adapted DataFrame ready for training.
    """
    cache_dir = download_cache_dir or (Path(__file__).parent.parent / "Dataset" / "_cache")

    # Handle URL
    if path_or_url.startswith("http"):
        dest = cache_dir / Path(path_or_url.split("?")[0]).name
        if not dest.exists():
            ok = download_url(path_or_url, dest)
            if not ok:
                return pd.DataFrame(), f"Failed to download from {path_or_url}"

        # Handle ZIP
        if dest.suffix.lower() == ".zip":
            extracted = download_zip(path_or_url, cache_dir / "extracted")
            csv_files = [f for f in extracted if f.suffix.lower() == ".csv"]
            if not csv_files:
                return pd.DataFrame(), "No CSV found in downloaded ZIP."
            dest = csv_files[0]

        path_or_url = str(dest)

    # Load CSV
    try:
        df = pd.read_csv(path_or_url)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read CSV: {exc}"

    if df.empty:
        return pd.DataFrame(), "Dataset is empty."

    # Apply adapter
    adapter = ADAPTERS.get(adapter_name, ADAPTERS["generic"])
    try:
        df = adapter.transform(df)
    except Exception as exc:
        # Fallback to generic adapter
        try:
            df = ADAPTERS["generic"].transform(df)
        except Exception as exc2:
            return pd.DataFrame(), f"Adapter error: {exc} | Fallback error: {exc2}"

    rows, fakes = len(df), int(df["label"].sum()) if "label" in df.columns else 0
    legit = rows - fakes
    msg = f"Loaded {rows:,} rows — Fake: {fakes:,} / Legit: {legit:,}"
    return df, msg


def get_dataset_info(path_or_url: str) -> Dict[str, Any]:
    """Return basic metadata for a dataset without fully loading it."""
    try:
        if path_or_url.startswith("http"):
            return {"status": "url", "url": path_or_url}
        df = pd.read_csv(path_or_url, nrows=1000)
        cols = list(df.columns)
        label_col = next(
            (c for c in ["fake", "label", "is_fake", "is_bot"] if c in [x.lower() for x in cols]),
            None,
        )
        return {
            "columns": cols,
            "sample_rows": len(df),
            "has_label": label_col is not None,
            "label_col": label_col,
        }
    except Exception as exc:
        return {"error": str(exc)}


def save_for_training(df: pd.DataFrame, platform: str) -> Path:
    """
    Save an adapted DataFrame to csv/<platform>_real.csv for use in training.
    Returns the saved path.
    """
    CSV_OUT_DIR.mkdir(exist_ok=True)
    out_path = CSV_OUT_DIR / f"{platform}_real.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved real dataset to %s (%d rows)", out_path, len(df))
    return out_path
