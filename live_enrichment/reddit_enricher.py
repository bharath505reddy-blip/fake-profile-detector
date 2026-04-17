"""
live_enrichment/reddit_enricher.py — Reddit live profile enrichment.

Fetches public Reddit user data via the JSON API (no auth required).
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_HEADERS = {"User-Agent": "FakeProfileDetector:v1.0 (educational project)"}
_TIMEOUT = 8


def _get_json(url: str) -> Optional[dict]:
    if not REQUESTS_AVAILABLE:
        return None
    try:
        r = _http.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        logger.debug("Reddit API %s → %d", url, r.status_code)
        return None
    except Exception as exc:
        logger.debug("Reddit API error: %s", exc)
        return None


def _shannon_entropy(dist: Dict[str, int]) -> float:
    total = sum(dist.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in dist.values() if v > 0)


def _hour_entropy(timestamps: List[float]) -> float:
    """Shannon entropy of hour-of-day distribution."""
    if not timestamps:
        return 0.0
    hour_dist: Dict[int, int] = {}
    for ts in timestamps:
        h = int(datetime.fromtimestamp(ts, tz=timezone.utc).hour)
        hour_dist[h] = hour_dist.get(h, 0) + 1
    return _shannon_entropy({str(k): v for k, v in hour_dist.items()})


def enrich(username: str) -> Dict:
    """
    Fetch Reddit profile data and compute enrichment features.

    Returns a standardised dict with:
      raw_data, features, data_completeness_score, error (optional)
    """
    if not REQUESTS_AVAILABLE:
        return {"error": "requests not installed", "data_completeness_score": 0.0,
                "raw_data": {}, "features": {}}

    about = _get_json(f"https://www.reddit.com/user/{username}/about.json")
    if about is None:
        return {"error": f"Reddit user '{username}' not found or API error",
                "data_completeness_score": 0.0, "raw_data": {}, "features": {}}

    data = about.get("data", {})
    fields_fetched = 1
    fields_total = 5

    comment_karma = data.get("comment_karma", 0) or 0
    link_karma = data.get("link_karma", 0) or 0
    total_karma = data.get("total_karma", 0) or (comment_karma + link_karma)
    created_ts = data.get("created_utc", 0) or 0
    trophies = 0  # trophy count requires separate call, skip for speed
    icon_img = data.get("icon_img", "") or ""
    about_text = data.get("subreddit", {}).get("public_description", "") or ""

    account_age_days = 0.0
    created_str = ""
    if created_ts:
        created_str = datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        account_age_days = (datetime.now(timezone.utc).timestamp() - created_ts) / 86400
        fields_fetched += 1

    # ---- Fetch recent submissions and comments ----
    overview = _get_json(
        f"https://www.reddit.com/user/{username}/overview.json?limit=50"
    )
    posts: List[dict] = []
    comments: List[dict] = []
    timestamps: List[float] = []
    subreddits: set = set()
    controversial_count = 0
    comment_lengths: List[int] = []

    if overview:
        fields_fetched += 1
        children = (overview.get("data", {}).get("children") or [])
        for child in children:
            kind = child.get("kind", "")
            cd = child.get("data", {})
            ts = cd.get("created_utc")
            if ts:
                timestamps.append(float(ts))
            sr = cd.get("subreddit")
            if sr:
                subreddits.add(sr)

            if kind == "t1":  # comment
                comments.append(cd)
                body = cd.get("body", "") or ""
                comment_lengths.append(len(body.split()))
                if cd.get("controversiality", 0):
                    controversial_count += 1
            elif kind == "t3":  # link/post
                posts.append(cd)

    fields_fetched += 1  # mark activity as fetched

    total_comments = max(len(comments), 1)
    total_posts = max(len(posts), 1)
    total_activity = total_comments + total_posts

    karma_per_day = total_karma / max(account_age_days, 1)
    comment_to_post_ratio = len(comments) / total_posts
    subreddit_diversity = len(subreddits) / max(total_activity, 1)
    avg_comment_length = sum(comment_lengths) / max(len(comment_lengths), 1)
    controversial_ratio = controversial_count / total_comments
    posting_time_entropy = _hour_entropy(timestamps)
    karma_to_activity_ratio = total_karma / total_activity

    features = {
        "karma": total_karma,
        "comment_karma": comment_karma,
        "link_karma": link_karma,
        "account_age_days": account_age_days,
        "about": about_text,
        "avatar_url": icon_img,
        "karma_per_day": karma_per_day,
        "comment_to_post_ratio": comment_to_post_ratio,
        "subreddit_diversity": subreddit_diversity,
        "avg_comment_length": avg_comment_length,
        "controversial_ratio": controversial_ratio,
        "posting_time_entropy": posting_time_entropy,
        "karma_to_activity_ratio": karma_to_activity_ratio,
        "total_recent_posts": len(posts),
        "total_recent_comments": len(comments),
        "unique_subreddits": len(subreddits),
    }

    logger.info(
        "Reddit enrichment for '%s': completeness=%.2f karma=%d age=%dd",
        username, fields_fetched / fields_total, total_karma, int(account_age_days),
    )

    return {
        "platform": "reddit",
        "username": username,
        "raw_data": {
            "name": data.get("name", ""),
            "karma": total_karma,
            "comment_karma": comment_karma,
            "link_karma": link_karma,
            "created_at": created_str,
            "about": about_text,
            "icon_img": icon_img,
            "is_verified": data.get("verified", False),
        },
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
        "posts": posts,
        "comments": comments,
        "timestamps": timestamps,
    }
