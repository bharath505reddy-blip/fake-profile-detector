"""
features/consistency.py — Profile consistency and coherence feature extraction.

Checks for mismatches between different profile fields (username vs display name,
bio language vs post language, profile completeness, etc.)

All third-party imports are optional and fail gracefully.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from langdetect import detect as langdetect_detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    Returns ISO 639-1 language code or 'unknown'.
    """
    if not text or len(text.split()) < 3:
        return "unknown"
    if LANGDETECT_AVAILABLE:
        try:
            return langdetect_detect(text)
        except Exception:
            return "unknown"
    return "unknown"


def _language_consistency(bio: str, posts: List[str]) -> float:
    """
    Compare detected language of bio vs majority language of posts.
    Returns 1.0 if same, 0.0 if different, 0.5 if undetermined.
    """
    bio_lang = _detect_language(bio)
    if not posts:
        return 0.5

    post_text = " ".join(posts[:10])  # Use first 10 posts for efficiency
    post_lang = _detect_language(post_text)

    if bio_lang == "unknown" or post_lang == "unknown":
        return 0.5
    return 1.0 if bio_lang == post_lang else 0.0


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.
    Uses jellyfish if available, else falls back to a simple LCS-based ratio.
    """
    if not s1 or not s2:
        return 0.0
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if s1 == s2:
        return 1.0

    if JELLYFISH_AVAILABLE:
        try:
            return round(jellyfish.jaro_winkler_similarity(s1, s2), 4)
        except Exception:
            pass

    # Fallback: longest common subsequence ratio
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    return round(2 * lcs_len / (m + n), 4)


def _name_username_similarity(display_name: str, username: str) -> float:
    """
    Jaro-Winkler similarity between display name and username (both normalized).
    Bots often have random mismatches between these fields.
    Returns 0-1 float.
    """
    # Strip common username separators and digits for comparison
    clean_name = re.sub(r"[^a-zA-Z ]", " ", display_name or "").strip()
    clean_uname = re.sub(r"[^a-zA-Z ]", " ", username or "").strip()
    if not clean_name or not clean_uname:
        return 0.5  # Unknown — neutral
    return _jaro_winkler_similarity(clean_name, clean_uname)


def _profile_completeness_score(
    has_bio: bool,
    has_avatar: bool,
    has_location: bool,
    has_url: bool,
    has_display_name: bool,
) -> float:
    """
    Weighted completeness score based on presence of profile fields.
    Weights: bio=0.30, avatar=0.25, location=0.15, url=0.15, display_name=0.15
    Returns 0-1 float.
    """
    score = (
        int(bool(has_bio)) * 0.30 +
        int(bool(has_avatar)) * 0.25 +
        int(bool(has_location)) * 0.15 +
        int(bool(has_url)) * 0.15 +
        int(bool(has_display_name)) * 0.15
    )
    return round(score, 4)


def _bio_post_topic_similarity(bio: str, posts: List[str]) -> float:
    """
    TF-IDF cosine similarity between bio and aggregated post text.
    Fakes often have mismatched bio vs actual post content.
    Returns 0-1 float.
    """
    if not bio or not posts:
        return 0.0
    post_text = " ".join(posts)
    if not post_text.strip():
        return 0.0

    if SKLEARN_AVAILABLE:
        try:
            vec = TfidfVectorizer(min_df=1)
            mat = vec.fit_transform([bio, post_text])
            sim = cosine_similarity(mat[0], mat[1])[0][0]
            return round(float(sim), 4)
        except Exception as exc:
            logger.debug("bio_post_topic_similarity TF-IDF error: %s", exc)

    # Jaccard fallback
    bio_set = set(re.findall(r"\w+", bio.lower()))
    post_set = set(re.findall(r"\w+", post_text.lower()))
    inter = len(bio_set & post_set)
    union = len(bio_set | post_set)
    return round(inter / max(union, 1), 4)


def extract_consistency_features(profile_data: dict) -> dict:
    """
    Extract profile consistency and coherence features.

    Expected profile_data keys (all optional):
      bio              — str
      username         — str
      display_name     — str
      location         — str
      posts            — list of str
      has_bio          — bool/int
      has_avatar       — bool/int
      has_location     — bool/int
      has_url          — bool/int
      has_display_name — bool/int

    Returns dict with keys:
      language_consistency, name_username_similarity,
      profile_completeness_score, bio_post_topic_similarity
    """
    features: Dict[str, float] = {}

    try:
        bio = str(profile_data.get("bio") or "").strip()
        username = str(profile_data.get("username") or "").strip()
        display_name = str(profile_data.get("display_name") or
                           profile_data.get("name") or "").strip()

        raw_posts = profile_data.get("posts") or []
        if isinstance(raw_posts, str):
            raw_posts = [raw_posts]
        posts: List[str] = [str(p) for p in raw_posts if p and str(p).strip()]

        # Profile presence fields — support both bool and int input
        has_bio = bool(profile_data.get("has_bio", bool(bio)))
        has_avatar = bool(profile_data.get("has_avatar", 0))
        has_location = bool(profile_data.get("has_location",
                                             bool(profile_data.get("location", ""))))
        has_url = bool(profile_data.get("has_url", 0))
        has_display_name = bool(profile_data.get("has_display_name", bool(display_name)))

        features["language_consistency"] = _language_consistency(bio, posts)
        features["name_username_similarity"] = _name_username_similarity(display_name, username)
        features["profile_completeness_score"] = _profile_completeness_score(
            has_bio, has_avatar, has_location, has_url, has_display_name
        )
        features["bio_post_topic_similarity"] = _bio_post_topic_similarity(bio, posts)

    except Exception as exc:
        logger.warning("extract_consistency_features error: %s", exc)

    # Replace NaN/inf with 0.0
    for k, v in features.items():
        try:
            if not math.isfinite(float(v)):
                features[k] = 0.0
        except (TypeError, ValueError):
            features[k] = 0.0

    return features
