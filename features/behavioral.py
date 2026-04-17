"""
features/behavioral.py — Behavioral fingerprinting from post/activity history.

Computes timing, content, and engagement pattern features from a list of posts
and timestamps. All features handle missing/sparse data gracefully.
"""
from __future__ import annotations

import logging
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Time gap bins for entropy computation (in hours)
_GAP_BINS = [
    (0, 1/60),        # < 1 min
    (1/60, 5/60),     # 1-5 min
    (5/60, 0.5),      # 5-30 min
    (0.5, 2),         # 30min-2hr
    (2, 6),           # 2-6hr
    (6, 24),          # 6-24hr
    (24, 72),         # 1-3 days
    (72, 168),        # 3-7 days
]


def _bin_gap(gap_hours: float) -> int:
    """Return bin index for a time gap in hours."""
    for i, (lo, hi) in enumerate(_GAP_BINS):
        if lo <= gap_hours < hi:
            return i
    return len(_GAP_BINS)  # > 7 days


def _shannon_entropy(counts: List[int]) -> float:
    """Shannon entropy of a frequency distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def _hhi(counts: List[int]) -> float:
    """Herfindahl-Hirschman Index (market concentration). 0=diverse, 1=monopoly."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in counts)


def _parse_timestamps(raw: List) -> List[float]:
    """Convert list of timestamps (unix float or ISO string) to sorted unix floats."""
    result = []
    for t in raw:
        if t is None:
            continue
        try:
            result.append(float(t))
            continue
        except (TypeError, ValueError):
            pass
        try:
            from dateutil import parser as dp
            result.append(dp.parse(str(t)).timestamp())
        except Exception:
            pass
    return sorted(result)


def _type_token_ratio(texts: List[str]) -> float:
    """Type-token ratio (lexical diversity) across concatenated post texts."""
    all_words = []
    for t in texts:
        words = re.findall(r"\b[a-z]+\b", t.lower())
        all_words.extend(words)
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _topic_counts(texts: List[str]) -> Dict[str, int]:
    """
    Simple keyword-based topic categorisation.
    Returns {topic_name: post_count}.
    """
    TOPIC_KEYWORDS = {
        "crypto":  ["bitcoin", "btc", "eth", "crypto", "nft", "defi", "token", "hodl"],
        "promo":   ["giveaway", "airdrop", "promo", "free", "win", "offer"],
        "politics":["election", "president", "vote", "democrat", "republican", "government"],
        "sports":  ["football", "soccer", "basketball", "nba", "nfl", "game", "score"],
        "tech":    ["code", "software", "python", "javascript", "api", "github", "developer"],
        "food":    ["recipe", "food", "eat", "restaurant", "cook", "delicious"],
        "travel":  ["travel", "trip", "flight", "hotel", "vacation", "country"],
        "other":   [],
    }

    topic_counts: Dict[str, int] = {k: 0 for k in TOPIC_KEYWORDS}
    for text in texts:
        t = text.lower()
        matched = False
        for topic, keywords in TOPIC_KEYWORDS.items():
            if topic == "other":
                continue
            if any(kw in t for kw in keywords):
                topic_counts[topic] += 1
                matched = True
                break
        if not matched:
            topic_counts["other"] += 1

    return topic_counts


def _is_repost(text: str) -> bool:
    """Heuristic: does the text look like a repost/retweet/share?"""
    lower = text.lower().strip()
    return (
        lower.startswith("rt ") or
        lower.startswith("retweet") or
        lower.startswith("shared ") or
        lower.startswith("via @") or
        bool(re.match(r"^(rt|repost|share)\b", lower))
    )


def _is_self_reply(post: dict) -> bool:
    """Check if a post is a reply to the same author."""
    if not isinstance(post, dict):
        return False
    referenced = post.get("referenced_tweets") or post.get("in_reply_to_user_id")
    if isinstance(referenced, list):
        return any(r.get("type") == "replied_to" for r in referenced)
    return False


# ---------------------------------------------------------------------------
# Default result structure
# ---------------------------------------------------------------------------

_DEFAULT_RESULT: Dict[str, object] = {
    # Timing
    "behav_posting_interval_entropy": None,
    "behav_posting_interval_mean_hr": None,
    "behav_posting_interval_std_hr": None,
    "behav_burst_post_count": None,
    "behav_hourly_distribution_skew": None,
    # Content
    "behav_topic_hhi": None,
    "behav_original_ratio": None,
    "behav_avg_post_length": None,
    "behav_post_length_variance": None,
    "behav_url_post_ratio": None,
    "behav_self_reply_ratio": None,
    "behav_lexical_diversity": None,
    # Engagement
    "behav_avg_engagement": None,
    "behav_engagement_variance": None,
    "behav_zero_engagement_ratio": None,
    # Meta
    "behav_post_count": None,
    "behav_low_confidence": 1,
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_behavioral_fingerprint(
    posts: Optional[List],
    timestamps: Optional[List],
    platform: str = "",
) -> Dict[str, object]:
    """
    Compute behavioral fingerprint features from a list of posts and timestamps.

    Args:
        posts: List of post dicts (from enricher output). May be None.
        timestamps: List of timestamp values (unix floats or ISO strings).
        platform: Platform name for logging context.

    Returns:
        Dict of feature_name -> value. behav_low_confidence=1 when < 5 posts available.

    Key features:
        behav_posting_interval_entropy  — entropy of time-gap distribution
        behav_posting_interval_mean_hr  — mean gap in hours
        behav_posting_interval_std_hr   — std dev of gaps (near-zero = bot)
        behav_burst_post_count          — posts within 60-second windows
        behav_hourly_distribution_skew  — skewness of hour-of-day distribution
        behav_topic_hhi                 — topic concentration (HHI)
        behav_original_ratio            — original posts / total
        behav_avg_post_length           — mean word count
        behav_post_length_variance      — variance in word count
        behav_url_post_ratio            — posts with URLs / total
        behav_self_reply_ratio          — self-reply posts / total
        behav_lexical_diversity         — type-token ratio across all posts
        behav_avg_engagement            — avg likes+comments per post
        behav_engagement_variance       — variance in engagement
        behav_zero_engagement_ratio     — posts with 0 engagement / total
        behav_post_count                — number of posts analysed
        behav_low_confidence            — 1 if fewer than 5 posts
    """
    result = dict(_DEFAULT_RESULT)

    if not posts and not timestamps:
        return result

    posts = posts or []
    timestamps = timestamps or []

    # Parse texts from posts
    texts: List[str] = []
    for p in posts:
        if isinstance(p, dict):
            text = p.get("text") or p.get("body") or p.get("selftext") or p.get("title") or ""
            texts.append(str(text))
        elif isinstance(p, str):
            texts.append(p)

    post_count = max(len(posts), len(texts))
    result["behav_post_count"] = post_count
    result["behav_low_confidence"] = int(post_count < 5)

    # ---- Timing features ----
    ts = _parse_timestamps(timestamps or [
        p.get("created_utc", p.get("created_at", None))
        for p in posts if isinstance(p, dict)
    ])

    if len(ts) >= 2:
        gaps_secs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        gaps_hrs = [g / 3600 for g in gaps_secs]

        # Bin gaps for entropy
        bin_counts = Counter(_bin_gap(g) for g in gaps_hrs)
        entropy_counts = [bin_counts.get(i, 0) for i in range(len(_GAP_BINS) + 1)]
        result["behav_posting_interval_entropy"] = _shannon_entropy(entropy_counts)
        result["behav_posting_interval_mean_hr"] = sum(gaps_hrs) / len(gaps_hrs)

        try:
            result["behav_posting_interval_std_hr"] = statistics.stdev(gaps_hrs)
        except statistics.StatisticsError:
            result["behav_posting_interval_std_hr"] = 0.0

        # Burst detection: posts within 60s
        burst = sum(1 for g in gaps_secs if g < 60)
        result["behav_burst_post_count"] = burst

        # Hour of day distribution skewness
        hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts]
        if len(hours) >= 3:
            try:
                result["behav_hourly_distribution_skew"] = statistics.mean(
                    [(h - statistics.mean(hours)) ** 3 for h in hours]
                ) / max(statistics.stdev(hours) ** 3, 1e-9)
            except Exception:
                result["behav_hourly_distribution_skew"] = 0.0

    # ---- Content features ----
    if texts:
        word_counts = [len(t.split()) for t in texts]
        url_posts = sum(1 for t in texts if _URL_RE.search(t))
        reposts = sum(1 for t in texts if _is_repost(t))
        self_replies = sum(1 for p in posts if isinstance(p, dict) and _is_self_reply(p))

        result["behav_avg_post_length"] = sum(word_counts) / len(word_counts)
        try:
            result["behav_post_length_variance"] = statistics.variance(word_counts)
        except statistics.StatisticsError:
            result["behav_post_length_variance"] = 0.0

        result["behav_original_ratio"] = 1.0 - (reposts / max(len(texts), 1))
        result["behav_url_post_ratio"] = url_posts / len(texts)
        result["behav_self_reply_ratio"] = self_replies / max(len(posts), 1)
        result["behav_lexical_diversity"] = _type_token_ratio(texts)

        # Topic concentration
        topic_counts = _topic_counts(texts)
        result["behav_topic_hhi"] = _hhi(list(topic_counts.values()))

    # ---- Engagement features ----
    engagements: List[float] = []
    for p in posts:
        if not isinstance(p, dict):
            continue
        metrics = p.get("public_metrics") or {}
        likes = (metrics.get("like_count") or p.get("ups") or
                 p.get("score") or p.get("likes") or 0)
        comments = (metrics.get("reply_count") or p.get("num_comments") or
                    p.get("comments") or 0)
        total = (likes or 0) + (comments or 0)
        engagements.append(float(total))

    if engagements:
        result["behav_avg_engagement"] = sum(engagements) / len(engagements)
        try:
            result["behav_engagement_variance"] = statistics.variance(engagements)
        except statistics.StatisticsError:
            result["behav_engagement_variance"] = 0.0
        result["behav_zero_engagement_ratio"] = sum(1 for e in engagements if e == 0) / len(engagements)

    if post_count > 0:
        logger.info(
            "[%s] Behavioral fingerprint: %d posts, interval_entropy=%.2f, topic_hhi=%.2f",
            platform, post_count,
            result.get("behav_posting_interval_entropy") or 0,
            result.get("behav_topic_hhi") or 0,
        )

    return result
