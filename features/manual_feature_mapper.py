"""
features/manual_feature_mapper.py

Maps the comprehensive manual prediction form (from templates/manual_predict.html)
into a feature dict that the per-platform model can consume. This lets a fully
filled manual form produce predictions of the same quality as a successful
Live Lookup.

Design:
- Required Section-1 fields (followers, following, posts, bio, ...) become the
  base columns the existing feature_builder expects.
- Optional Section 2-6 fields (dropdowns, multi-selects, dates, text) are
  translated to the same numeric features the live enrichers would produce.
- Pre-computed photo/posts features arrive as JSON in hidden form fields.
- Empty optional fields become None (NOT 0) so missing_value_handler can fill
  platform medians and mark a *_missing flag.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lookup tables - convert dropdown text values to numeric feature signals.
# ---------------------------------------------------------------------------

POSTING_PATTERN_MAP = {
    "Regular/Scheduled":          {"interval_entropy": 0.30, "interval_std": 0.20, "burst_score": 0.05},
    "Regular":                    {"interval_entropy": 0.30, "interval_std": 0.20, "burst_score": 0.05},
    "Irregular/Random":           {"interval_entropy": 0.75, "interval_std": 0.65, "burst_score": 0.25},
    "Irregular":                  {"interval_entropy": 0.75, "interval_std": 0.65, "burst_score": 0.25},
    "Burst (many then silence)": {"interval_entropy": 0.95, "interval_std": 0.95, "burst_score": 0.90},
    "Burst":                      {"interval_entropy": 0.95, "interval_std": 0.95, "burst_score": 0.90},
    "Very Infrequent":            {"interval_entropy": 0.50, "interval_std": 0.85, "burst_score": 0.40},
    "Infrequent":                 {"interval_entropy": 0.50, "interval_std": 0.85, "burst_score": 0.40},
    "Bursts then silence":        {"interval_entropy": 0.95, "interval_std": 0.95, "burst_score": 0.90},
    "Burst joins then silent":    {"interval_entropy": 0.95, "interval_std": 0.95, "burst_score": 0.90},
}

ENGAGEMENT_LEVEL_MAP = {
    "High":                                 {"avg_engagement": 0.08, "zero_engagement_ratio": 0.05},
    "High (lots of likes/comments)":        {"avg_engagement": 0.08, "zero_engagement_ratio": 0.05},
    "High upvotes":                         {"avg_engagement": 0.08, "zero_engagement_ratio": 0.05},
    "Medium":                               {"avg_engagement": 0.04, "zero_engagement_ratio": 0.20},
    "Low":                                  {"avg_engagement": 0.01, "zero_engagement_ratio": 0.55},
    "Mostly downvoted":                     {"avg_engagement": 0.005, "zero_engagement_ratio": 0.20},
    "Zero":                                 {"avg_engagement": 0.0, "zero_engagement_ratio": 0.95},
    "Zero (no engagement)":                 {"avg_engagement": 0.0, "zero_engagement_ratio": 0.95},
}

POST_LENGTH_MAP = {
    "Very Short (1-2 words)":     {"avg_length": 12,  "length_variance": 8},
    "Very short (1-2 words)":     {"avg_length": 12,  "length_variance": 8},
    "Short (sentence)":           {"avg_length": 50,  "length_variance": 18},
    "Short":                      {"avg_length": 50,  "length_variance": 18},
    "Medium (paragraph)":         {"avg_length": 150, "length_variance": 50},
    "Medium":                     {"avg_length": 150, "length_variance": 50},
    "Long":                       {"avg_length": 350, "length_variance": 90},
    "Long (near 280)":            {"avg_length": 250, "length_variance": 30},
    "Mixed":                      {"avg_length": 150, "length_variance": 130},
}

FOLLOWER_QUALITY_MAP = {
    "Mostly real-looking accounts": 0.85,
    "Mostly real":                  0.85,
    "Mostly real professionals":    0.85,
    "Active developers":            0.85,
    "Appears organic growth":       0.85,
    "Mix of real and suspicious":   0.50,
    "Mixed":                        0.50,
    "Mostly bots/empty profiles":   0.10,
    "Mostly bots":                  0.10,
    "Mostly empty profiles":        0.10,
    "Suspicious growth pattern":    0.15,
    "Can't tell":                   None,
    "Unknown":                      None,
}

MUTUAL_FOLLOW_MAP = {
    "High (most follow back)": 0.80,
    "High":                    0.80,
    "Medium":                  0.45,
    "Low (one-directional)":   0.10,
    "Low":                     0.10,
    "Unknown":                 None,
}

CROSS_PLATFORM_CONSISTENCY_MAP = {
    "Same name/bio/photo everywhere":      0.95,
    "Same everywhere":                     0.95,
    "Similar but not identical":           0.65,
    "Similar":                             0.65,
    "Very different profiles":             0.20,
    "Very different":                      0.20,
    "Only exists on this platform":        0.0,
    "Single platform":                     0.0,
    "Single platform only":                0.0,
    "Didn't check":                        None,
}

CROSS_PLATFORM_AGES_MAP = {
    "Yes, similar age across platforms":   0.90,
    "No, created at very different times": 0.30,
    "Some platforms much newer":           0.50,
    "Didn't check":                        None,
}

FOLLOWER_GROWTH_MAP = {
    "Gradual organic growth":      {"growth_score": 0.85, "spike_score": 0.05},
    "Gradual":                     {"growth_score": 0.85, "spike_score": 0.05},
    "Sudden spike":                {"growth_score": 0.30, "spike_score": 0.95},
    "Steady decline":              {"growth_score": 0.40, "spike_score": 0.10},
    "Decline":                     {"growth_score": 0.40, "spike_score": 0.10},
    "Stagnant/dead":               {"growth_score": 0.20, "spike_score": 0.05},
    "No change":                   {"growth_score": 0.50, "spike_score": 0.05},
    "Gradual natural growth":      {"growth_score": 0.85, "spike_score": 0.05},
    "Sudden suspicious spike":     {"growth_score": 0.20, "spike_score": 0.95},
    "Unknown":                     {"growth_score": None, "spike_score": None},
}

UPLOAD_FREQ_MAP = {
    "Multiple daily": 5.0,
    "Daily":          1.0,
    "Weekly":         0.15,
    "Monthly":        0.04,
    "Sporadic":       0.02,
    "Rarely":         0.01,
    "Never":          0.0,
    "Inactive":       0.0,
    "Unknown":        None,
    "Today":          1.0,
    "This week":      0.5,
    "This month":     0.1,
    "Months ago":     0.01,
    "Years ago":      0.0,
}

VIDEO_LENGTH_MAP = {
    "< 1 min (Shorts)":   30,
    "< 15 sec":           10,
    "15-60 sec":          30,
    "1-3 min":            120,
    "1-5 min":            180,
    "3-10 min":           360,
    "5-15 min":           600,
    "15-60 min":          1800,
    "60+ min":            3900,
    "Mixed":              600,
}

CAPTION_LENGTH_MAP = POST_LENGTH_MAP

KARMA_RATIO_MAP = {
    "High karma, low posts (quality)":     0.85,
    "Balanced":                            0.55,
    "Low karma, high posts (spam-like)":   0.10,
    "Unknown":                             None,
}

LIKES_FOLLOWER_RATIO_MAP = {
    "High (organic)":         0.10,
    "Balanced":               0.05,
    "Suspiciously low":       0.005,
    "Unknown":                None,
}

CONTROVERSIAL_MAP = {
    "None":                  0.0,
    "Few":                   0.10,
    "Many":                  0.40,
    "Mostly controversial":  0.75,
    "Unknown":               None,
}

CONNECTIONS_BUCKET_MAP = {
    "< 50":     30,
    "50-150":   100,
    "150-500":  300,
    "500+":     500,
}

SUBSCRIBER_QUALITY_MAP = {
    "Appears organic growth":       0.85,
    "Suspicious growth pattern":    0.15,
    "Can't tell":                   None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    try:
        return int(round(f))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> int:
    """Form checkbox: present and truthy → 1, otherwise 0."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    s = str(value).strip().lower()
    return 1 if s in ("1", "true", "yes", "on", "y", "t") else 0


def _parse_date(value: str) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _days_since(d: Optional[date]) -> Optional[int]:
    if not d:
        return None
    return (date.today() - d).days


def _safe_json_load(value: str) -> Dict:
    if not value:
        return {}
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def _hhi(values: List[float]) -> float:
    """Herfindahl-Hirschman Index: 1.0 = single category, 0 = uniform."""
    total = sum(v for v in values if v > 0)
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in values)


# ---------------------------------------------------------------------------
# Posts text NLP
# ---------------------------------------------------------------------------

_HASHTAG_RE = re.compile(r"#\w+")
_URL_RE     = re.compile(r"https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|me)\b", re.I)
_EMOJI_RE   = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF]",
    flags=re.UNICODE,
)


def analyze_pasted_posts(posts_list: List[str]) -> Dict[str, Any]:
    """
    Lightweight NLP over a list of post strings. Returns features that mirror
    those produced by features/behavioral.py from a real timeline.
    """
    posts = [p.strip() for p in posts_list if p and p.strip()]
    n = len(posts)
    out: Dict[str, Any] = {"behav_post_count_analyzed": n}

    if n == 0:
        return out

    lengths       = [len(p) for p in posts]
    word_counts   = [len(p.split()) for p in posts]
    hashtag_cnts  = [len(_HASHTAG_RE.findall(p)) for p in posts]
    url_cnts      = [len(_URL_RE.findall(p)) for p in posts]
    emoji_cnts    = [len(_EMOJI_RE.findall(p)) for p in posts]

    avg_len = sum(lengths) / n
    var_len = (sum((x - avg_len) ** 2 for x in lengths) / n) if n > 1 else 0
    out["behav_avg_post_length"]      = round(avg_len, 2)
    out["behav_post_length_variance"] = round(var_len, 2)

    total_words = sum(word_counts) or 1
    unique_tokens = set()
    for p in posts:
        unique_tokens.update(t.lower().strip(".,!?:;") for t in p.split() if t)
    out["behav_lexical_diversity"] = round(len(unique_tokens) / total_words, 4)

    # Duplicate ratio: fraction of posts that exactly match another
    seen: Dict[str, int] = {}
    for p in posts:
        key = p.lower().strip()
        seen[key] = seen.get(key, 0) + 1
    dups = sum(1 for c in seen.values() if c > 1)
    out["behav_duplicate_post_ratio"] = round(dups / n, 4)

    out["behav_hashtag_per_post"] = round(sum(hashtag_cnts) / n, 3)
    out["behav_url_post_ratio"]   = round(sum(1 for c in url_cnts if c > 0) / n, 3)
    out["behav_emoji_per_post"]   = round(sum(emoji_cnts) / n, 3)

    # Sentiment variance: very rough heuristic via positive/negative word ratio
    pos_words = {"good", "great", "love", "amazing", "best", "happy", "win", "awesome",
                 "beautiful", "perfect", "thank", "thanks", "excited", "blessed"}
    neg_words = {"bad", "hate", "worst", "terrible", "awful", "sad", "angry", "tired",
                 "horrible", "disappointed", "fail", "failed"}
    sentiments = []
    for p in posts:
        toks = [t.lower().strip(".,!?:;") for t in p.split()]
        pos = sum(1 for t in toks if t in pos_words)
        neg = sum(1 for t in toks if t in neg_words)
        total = pos + neg
        sentiments.append((pos - neg) / total if total else 0.0)
    s_mean = sum(sentiments) / n
    out["behav_sentiment_mean"]     = round(s_mean, 3)
    out["behav_sentiment_variance"] = round(sum((s - s_mean) ** 2 for s in sentiments) / n, 3) if n > 1 else 0.0

    return out


# ---------------------------------------------------------------------------
# Multi-select / topic mapping
# ---------------------------------------------------------------------------

_TOPIC_RISK = {
    "Crypto/Finance": 0.85, "Crypto": 0.85, "Crypto promo": 0.95,
    "Adult content": 0.85, "NSFW": 0.85, "Spam": 0.95,
    "Promotions/Ads": 0.70, "Promo": 0.70,
    "Politics": 0.40, "Random content": 0.50,
    "Personal/Mixed": 0.10, "Personal": 0.10, "Lifestyle": 0.10,
    "Food": 0.10, "Travel": 0.10, "Tech": 0.10, "Music": 0.10,
    "Sports": 0.10, "Education": 0.05, "Gaming": 0.10, "Vlogs": 0.10,
    "Beauty": 0.10, "Comedy": 0.10, "Dance": 0.10, "News": 0.10,
    "Memes": 0.20, "Advice": 0.10, "Other": 0.30,
    "Community chat": 0.05, "Support": 0.05, "Bot commands": 0.30,
}


def _topic_features(topics: List[str]) -> Dict[str, Any]:
    if not topics:
        return {"behav_topic_count": None, "behav_topic_hhi": None, "behav_topic_risk": None}
    risks = [_TOPIC_RISK.get(t, 0.30) for t in topics]
    weights = [1.0] * len(topics)
    return {
        "behav_topic_count": len(topics),
        "behav_topic_hhi":   round(_hhi(weights), 3),  # uniform → low HHI
        "behav_topic_risk":  round(sum(risks) / len(risks), 3),
    }


def _content_type_features(content_types: List[str]) -> Dict[str, Any]:
    if not content_types:
        return {}
    ct = {c.lower() for c in content_types}
    has_original = any("original" in c for c in ct)
    has_repost   = any(("repost" in c) or ("share" in c) or ("retweet" in c) for c in ct)
    has_url      = any(("link" in c) or ("url" in c) for c in ct)
    return {
        "behav_original_ratio": 1.0 if has_original and not has_repost else (0.5 if has_original else 0.1),
        "behav_repost_ratio":   0.7 if has_repost   and not has_original else (0.3 if has_repost   else 0.0),
        "behav_url_post_ratio": 0.4 if has_url else 0.0,
    }


def _cross_platform_features(checked_platforms: List[str]) -> Dict[str, Any]:
    n = len(checked_platforms)
    return {
        "cross_platform_count":          n,
        "cross_platform_presence_ratio": round(n / 9.0, 3),  # 9 = max checkboxes shown
    }


# ---------------------------------------------------------------------------
# Per-platform mappers
# ---------------------------------------------------------------------------

def _common_account_age(form, out: Dict[str, Any]) -> None:
    """Resolve account_age_days from either explicit field or creation_date."""
    explicit = _to_int(form.get("account_age_days"))
    if explicit is not None and explicit >= 0:
        out["account_age_days"] = explicit
        return
    for date_key in ("creation_date", "cake_day", "joined_at_utc", "created_at_utc", "created_at"):
        d = _parse_date(form.get(date_key, ""))
        if d:
            out["account_age_days"] = _days_since(d)
            return
    out["account_age_days"] = None


def _instagram(form, out: Dict[str, Any]) -> None:
    out["username"]    = form.get("username", "").strip()
    out["followers"]   = _to_int(form.get("followers"))
    out["following"]   = _to_int(form.get("following"))
    out["posts"]       = _to_int(form.get("posts"))
    out["bio"]         = form.get("bio", "")
    out["is_verified"] = _to_bool(form.get("verified") or form.get("is_verified"))

    out["is_private"]        = _to_bool(form.get("is_private"))
    out["is_business"]       = _to_bool(form.get("is_business"))
    out["display_name"]      = form.get("display_name", "")
    out["location"]          = form.get("location", "")
    out["website_url"]       = form.get("website_url", "")
    out["profile_category"]  = form.get("profile_category", "")
    out["has_highlights"]    = _to_bool(form.get("has_highlights"))
    out["has_active_story"]  = _to_bool(form.get("has_active_story"))
    out["has_profile_pic"]   = _to_bool(form.get("has_custom_avatar")) or 1
    out["bio_has_url"]       = 1 if (form.get("website_url") or "").strip() else 0


def _facebook(form, out: Dict[str, Any]) -> None:
    out["name"]        = form.get("name", "").strip()
    out["friends"]     = _to_int(form.get("friends"))
    out["followers"]   = _to_int(form.get("followers"))
    out["posts"]       = _to_int(form.get("posts"))
    out["bio"]         = form.get("bio", "")
    out["is_verified"] = _to_bool(form.get("verified") or form.get("is_verified"))

    out["location"]            = form.get("location", "")
    out["workplace"]           = form.get("workplace", "")
    out["education"]           = form.get("education", "")
    out["relationship_listed"] = _to_bool(form.get("relationship_listed"))
    out["profile_url_slug"]    = form.get("profile_url_slug", "")
    out["has_cover_photo"]     = _to_bool(form.get("has_cover_photo"))
    out["mutual_friends"]      = _to_int(form.get("mutual_friends"))
    out["group_count"]         = _to_int(form.get("group_count"))


def _x(form, out: Dict[str, Any]) -> None:
    out["username"]    = form.get("username", "").strip()
    out["followers"]   = _to_int(form.get("followers"))
    out["following"]   = _to_int(form.get("following"))
    out["tweets"]      = _to_int(form.get("tweets"))
    out["bio"]         = form.get("bio", "")
    out["is_verified"] = _to_bool(form.get("verified") or form.get("is_verified"))

    out["display_name"]    = form.get("display_name", "")
    out["location"]        = form.get("location", "")
    out["website_url"]     = form.get("website_url", "")
    out["listed_count"]    = _to_int(form.get("listed_count"))
    out["default_header"]  = _to_bool(form.get("default_header"))
    out["tweet_source"]    = form.get("tweet_source", "")
    out["is_protected"]    = _to_bool(form.get("is_protected"))
    out["bot_client"]      = 1 if "third-party" in (form.get("tweet_source") or "").lower() or "bot" in (form.get("tweet_source") or "").lower() else 0


def _linkedin(form, out: Dict[str, Any]) -> None:
    out["name"]      = form.get("name", "").strip()
    conn_raw = form.get("connections", "")
    out["connections"] = CONNECTIONS_BUCKET_MAP.get(conn_raw, _to_int(conn_raw))
    out["followers"] = _to_int(form.get("followers"))
    out["headline"]  = form.get("headline", "")
    out["about"]     = form.get("about", "")

    out["work_experiences"]  = _to_int(form.get("work_experiences"))
    out["education_count"]   = _to_int(form.get("education_count"))
    out["skills_count"]      = _to_int(form.get("skills_count"))
    out["endorsement_count"] = _to_int(form.get("endorsement_count"))
    out["recommendations"]   = _to_int(form.get("recommendations"))
    out["has_summary"]       = _to_bool(form.get("has_summary"))
    out["has_photo"]         = _to_bool(form.get("has_photo")) or 1
    out["has_banner"]        = _to_bool(form.get("has_banner"))
    out["certifications"]    = _to_int(form.get("certifications"))
    out["volunteer_count"]   = _to_int(form.get("volunteer_count"))
    out["post_count"]        = _to_int(form.get("post_count"))
    out["mutual_connections"] = _to_int(form.get("mutual_connections"))
    out["headline_buzzwords"] = form.get("headline_buzzwords", "")

    # Map career consistency dropdowns
    consistency = form.get("job_dates_sequential", "")
    out["career_logical"] = (
        1.0 if consistency.startswith("Yes")
        else 0.0 if consistency.startswith("No")
        else None
    )


def _github(form, out: Dict[str, Any]) -> None:
    out["username"]      = form.get("username", "").strip()
    out["followers"]     = _to_int(form.get("followers"))
    out["following"]     = _to_int(form.get("following"))
    out["public_repos"]  = _to_int(form.get("public_repos"))
    out["bio"]           = form.get("bio", "")
    out["public_gists"]  = _to_int(form.get("public_gists"))

    out["display_name"]       = form.get("display_name", "")
    out["location"]           = form.get("location", "")
    out["website_url"]        = form.get("website_url", "")
    out["company"]            = form.get("company", "")
    out["is_hireable"]        = _to_bool(form.get("is_hireable"))
    out["total_stars"]        = _to_int(form.get("total_stars"))
    out["forked_repos"]       = _to_int(form.get("forked_repos"))
    out["has_readme_profile"] = _to_bool(form.get("has_readme_profile"))
    out["top_language"]       = form.get("top_language", "")
    out["unique_languages"]   = _to_int(form.get("unique_languages"))
    out["contributions_year"] = _to_int(form.get("contributions_year"))
    out["has_pinned_repos"]   = _to_bool(form.get("has_pinned_repos"))
    out["uses_identicon"]     = _to_bool(form.get("uses_identicon"))
    out["issues_prs_opened"]  = _to_int(form.get("issues_prs_opened"))
    out["org_count"]          = _to_int(form.get("org_count"))


def _discord(form, out: Dict[str, Any]) -> None:
    out["username"]        = form.get("username", "").strip()
    out["user_id"]         = form.get("user_id", "")
    out["bio"]             = form.get("bio", "")

    out["display_name"]     = form.get("display_name", "")
    out["has_avatar"]       = _to_bool(form.get("has_custom_avatar"))
    out["has_banner"]       = _to_bool(form.get("has_banner"))
    out["has_nitro"]        = _to_bool(form.get("has_nitro"))
    out["custom_status"]    = form.get("custom_status", "")
    out["pronouns"]         = form.get("pronouns", "")
    out["server_count"]     = _to_int(form.get("server_count"))
    out["days_first_message"] = _to_int(form.get("days_first_message"))
    out["role_count"]       = _to_int(form.get("role_count"))
    out["sends_unsolicited_dms"] = _to_bool(form.get("sends_unsolicited_dms"))
    out["mentions_everyone"] = _to_bool(form.get("mentions_everyone"))
    out["posts_invite_links"] = _to_bool(form.get("posts_invite_links"))
    out["suspicion_score"]  = _to_int(form.get("suspicion_score"))


def _youtube(form, out: Dict[str, Any]) -> None:
    out["channel_name"]  = form.get("channel_name", "").strip()
    out["subscribers"]   = _to_int(form.get("subscribers"))
    out["videos"]        = _to_int(form.get("total_videos"))
    out["about"]         = form.get("about", "")

    out["total_views"]      = _to_int(form.get("total_views"))
    out["country"]          = form.get("country", "")
    out["channel_handle"]   = form.get("channel_handle", "")
    out["channel_type"]     = form.get("channel_type", "")
    out["is_verified"]      = _to_bool(form.get("is_verified"))
    out["comments_enabled"] = _to_bool(form.get("comments_enabled"))
    out["has_custom_avatar"] = _to_bool(form.get("has_custom_avatar"))
    out["has_custom_banner"] = _to_bool(form.get("has_custom_banner"))
    out["avg_views"]         = _to_int(form.get("avg_views"))
    out["featured_channels"] = _to_int(form.get("featured_channels"))
    out["avg_video_length_s"] = VIDEO_LENGTH_MAP.get(form.get("avg_video_length", ""))


def _tiktok(form, out: Dict[str, Any]) -> None:
    out["username"]    = form.get("username", "").strip()
    out["followers"]   = _to_int(form.get("followers"))
    out["following"]   = _to_int(form.get("following"))
    out["videos"]      = _to_int(form.get("total_videos"))
    out["bio"]         = form.get("bio", "")
    out["is_verified"] = _to_bool(form.get("verified") or form.get("is_verified"))

    out["display_name"]      = form.get("display_name", "")
    out["total_likes"]       = _to_int(form.get("total_likes"))
    out["website_url"]       = form.get("website_url", "")
    out["is_private"]        = _to_bool(form.get("is_private"))
    out["category"]          = form.get("category", "")
    out["avg_views"]         = _to_int(form.get("avg_views"))
    out["avg_likes_video"]   = _to_int(form.get("avg_likes_video"))
    out["video_length_s"]    = VIDEO_LENGTH_MAP.get(form.get("video_length", ""))
    out["has_custom_avatar"] = _to_bool(form.get("has_custom_avatar"))


def _reddit(form, out: Dict[str, Any]) -> None:
    out["username"]      = form.get("username", "").strip()
    out["post_karma"]    = _to_int(form.get("post_karma"))
    out["comment_karma"] = _to_int(form.get("comment_karma"))
    out["about"]         = form.get("about", "")
    pk = out["post_karma"] or 0
    ck = out["comment_karma"] or 0
    out["karma"] = (pk + ck) if (out["post_karma"] is not None or out["comment_karma"] is not None) else None

    out["trophy_count"]      = _to_int(form.get("trophy_count"))
    out["has_premium"]       = _to_bool(form.get("has_premium"))
    out["has_custom_avatar"] = _to_bool(form.get("has_custom_avatar"))
    out["has_custom_banner"] = _to_bool(form.get("has_custom_banner"))
    out["display_name"]      = form.get("display_name", "")
    out["subreddits_active"] = _to_int(form.get("subreddits_active"))
    out["controversial_ratio"] = CONTROVERSIAL_MAP.get(form.get("controversial_ratio", ""))
    out["karma_activity"]    = KARMA_RATIO_MAP.get(form.get("karma_activity_ratio", ""))


def _snapchat(form, out: Dict[str, Any]) -> None:
    out["username"] = form.get("username", "").strip()
    out["score"]    = _to_int(form.get("snap_score"))
    out["bio"]      = form.get("bio", "")

    out["display_name"]      = form.get("display_name", "")
    out["has_bitmoji"]       = _to_bool(form.get("has_bitmoji"))
    out["snap_map_location"] = form.get("snap_map_location", "")
    out["is_public"]         = _to_bool(form.get("is_public"))
    out["has_public_story"]  = _to_bool(form.get("has_public_story"))
    out["sends_unsolicited"] = _to_bool(form.get("sends_unsolicited"))


_PLATFORM_MAPPERS = {
    "instagram": _instagram,
    "facebook":  _facebook,
    "x":         _x,
    "twitter":   _x,
    "linkedin":  _linkedin,
    "github":    _github,
    "discord":   _discord,
    "youtube":   _youtube,
    "tiktok":    _tiktok,
    "reddit":    _reddit,
    "snapchat":  _snapchat,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ManualFeatureMapper:
    """
    Convert a Werkzeug-style form (request.form) for a given platform into a
    feature dict. Empty optional fields stay None (so the missing-value
    handler can fill platform medians).
    """

    def map_form_to_features(self, form, platform: str) -> Dict[str, Any]:
        platform = (platform or "").lower()
        out: Dict[str, Any] = {}

        # Always include the platform key for downstream tools
        out["platform"] = platform

        # Platform-specific basics + extras
        mapper = _PLATFORM_MAPPERS.get(platform)
        if mapper:
            mapper(form, out)
        else:
            logger.warning("ManualFeatureMapper: unknown platform %s", platform)

        # Common: account age
        _common_account_age(form, out)

        # Behavioral signals from dropdowns
        out.update(self._behavioral_features(form, platform))

        # Cross-platform
        cross_platforms = form.getlist("cross_platforms") if hasattr(form, "getlist") else []
        out.update(_cross_platform_features(cross_platforms))
        consistency = form.get("cross_platform_consistency", "")
        out["cross_platform_consistency_score"] = CROSS_PLATFORM_CONSISTENCY_MAP.get(consistency)
        ages = form.get("cross_platform_ages", "")
        out["cross_platform_age_consistency"] = CROSS_PLATFORM_AGES_MAP.get(ages)

        # Forensics: username + bio (live, fast, no network)
        try:
            from features.username_forensics import analyze_username
            uname = out.get("username") or out.get("name") or out.get("channel_name") or ""
            if uname:
                out.update({k: v for k, v in analyze_username(uname).items() if v is not None})
        except Exception as exc:
            logger.debug("Username forensics skipped: %s", exc)

        try:
            from features.bio_forensics import analyze_bio
            bio_text = (
                out.get("bio") or out.get("about") or out.get("headline") or ""
            )
            if bio_text:
                out.update({k: v for k, v in analyze_bio(bio_text, platform).items() if v is not None})
        except Exception as exc:
            logger.debug("Bio forensics skipped: %s", exc)

        # Pre-computed photo features (from AJAX hidden field)
        photo_features = _safe_json_load(form.get("photo_features_json", ""))
        if photo_features:
            out.update(photo_features)
        else:
            avatar_url = form.get("avatar_url", "").strip()
            if avatar_url:
                try:
                    from features.photo_analysis import analyze_avatar
                    out.update({
                        k: v for k, v in analyze_avatar(avatar_url, platform).items()
                        if v is not None
                    })
                except Exception as exc:
                    logger.debug("Avatar analysis skipped: %s", exc)

        # Pre-computed posts NLP features (from AJAX hidden field)
        posts_features = _safe_json_load(form.get("posts_features_json", ""))
        if posts_features:
            out.update(posts_features)
        else:
            for key in ("recent_posts", "recent_comments", "recent_titles",
                        "recent_messages", "recent_captions"):
                txt = form.get(key, "")
                if txt and txt.strip():
                    posts_list = [p.strip() for p in txt.split("\n") if p.strip()]
                    if posts_list:
                        out.update(analyze_pasted_posts(posts_list))
                    break

        # Derived ratios
        out.update(self._derived_features(out))

        return out

    # ------------------------------------------------------------------
    # Behavioral / dropdown mapping
    # ------------------------------------------------------------------
    def _behavioral_features(self, form, platform: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        pp = form.get("posting_pattern", "")
        info = POSTING_PATTERN_MAP.get(pp)
        if info:
            out["behav_posting_interval_entropy"] = info["interval_entropy"]
            out["behav_posting_interval_std"]     = info["interval_std"]
            out["behav_burst_score"]              = info["burst_score"]

        eng = form.get("engagement_level", "") or form.get("engagement_received", "")
        info = ENGAGEMENT_LEVEL_MAP.get(eng)
        if info:
            out["behav_avg_engagement"]        = info["avg_engagement"]
            out["behav_zero_engagement_ratio"] = info["zero_engagement_ratio"]

        cl = (form.get("caption_length", "") or form.get("comment_length", "")
              or form.get("tweet_length", ""))
        info = POST_LENGTH_MAP.get(cl)
        if info:
            out["behav_avg_post_length"]      = info["avg_length"]
            out["behav_post_length_variance"] = info["length_variance"]

        # Topic concentration / risk
        topics = form.getlist("topics") if hasattr(form, "getlist") else []
        if not topics:
            topics = form.getlist("primary_activity") if hasattr(form, "getlist") else []
        out.update(_topic_features(topics))

        # Content type signals
        ct = form.getlist("content_types") if hasattr(form, "getlist") else []
        out.update(_content_type_features(ct))

        # Avg posts per day
        appd = _to_float(form.get("avg_posts_per_day")) or _to_float(form.get("avg_tweets_per_day"))
        if appd is not None:
            out["behav_posts_per_day"] = appd

        # Follower quality / mutual follow
        fq = form.get("follower_quality", "") or form.get("friend_quality", "") or form.get("connection_quality", "")
        score = FOLLOWER_QUALITY_MAP.get(fq)
        if score is not None:
            out["follower_quality_score"] = score

        mf = form.get("mutual_follow_rate", "")
        score = MUTUAL_FOLLOW_MAP.get(mf)
        if score is not None:
            out["mutual_follow_rate"] = score

        out["follows_celebrities"] = _to_bool(form.get("follows_celebrities"))

        fg = form.get("follower_growth", "") or form.get("score_growth", "")
        info = FOLLOWER_GROWTH_MAP.get(fg)
        if info and info.get("growth_score") is not None:
            out["follower_growth_score"] = info["growth_score"]
            out["follower_spike_score"]  = info["spike_score"]

        # YouTube/TikTok upload frequency
        uf = form.get("upload_frequency", "") or form.get("posting_frequency", "")
        v = UPLOAD_FREQ_MAP.get(uf)
        if v is not None:
            out["uploads_per_day"] = v

        # Likes/follower ratio (TikTok)
        lf = form.get("likes_follower_ratio", "")
        v = LIKES_FOLLOWER_RATIO_MAP.get(lf)
        if v is not None:
            out["likes_follower_ratio"] = v

        # Subscriber quality (YouTube)
        sq = form.get("subscriber_quality", "")
        v = SUBSCRIBER_QUALITY_MAP.get(sq)
        if v is not None:
            out["follower_quality_score"] = v

        return out

    # ------------------------------------------------------------------
    # Derived ratios
    # ------------------------------------------------------------------
    def _derived_features(self, f: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        followers = f.get("followers")
        following = f.get("following")
        posts = f.get("posts") or f.get("tweets") or f.get("videos")
        age = f.get("account_age_days")

        if followers is not None and following is not None:
            denom = following if following > 0 else 1
            out["ff_ratio"] = round(followers / denom, 4)

        if posts is not None and followers is not None:
            denom = followers if followers > 0 else 1
            out["posts_per_100_followers"] = round(posts / denom * 100, 4)

        if posts is not None and age is not None and age > 0:
            out["posts_per_day"] = round(posts / age, 4)

        if followers is not None:
            out["log_followers"] = round(math.log1p(max(followers, 0)), 4)

        if followers is not None:
            if followers < 1000: tier = 0
            elif followers < 100_000: tier = 1
            elif followers < 1_000_000: tier = 2
            else: tier = 3
            out["followers_tier"] = tier

        # Profile completeness 0-1: fraction of important fields filled
        important = [
            "bio", "username", "name", "display_name", "location",
            "website_url", "creation_date", "has_profile_pic", "is_verified",
        ]
        present = sum(1 for k in important if f.get(k) not in (None, "", 0))
        out["profile_completeness"] = round(present / len(important), 3)

        return out


# Module-level convenience
def map_form(form, platform: str) -> Dict[str, Any]:
    return ManualFeatureMapper().map_form_to_features(form, platform)
